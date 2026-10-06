// Optional AI layer using Hugging Face Inference Providers (OpenAI-compatible). The user pastes their own free token; it is
// kept in memory (and in this browser's localStorage only if they tick "remember"). It is sent only to router.huggingface.co.
// Every failure resolves to null so callers fall back to templates; nothing here throws.

const ENDPOINT = 'https://router.huggingface.co/v1/chat/completions';
export const MODELS = ['Qwen/Qwen3-4B-Instruct-2507', 'google/gemma-3-4b-it', 'meta-llama/Llama-3.1-8B-Instruct'];
const KEY = 'crc-hf-token';

let session = null; // { token, model }

export const llmLive = () => session !== null;
export const llmModel = () => (session ? session.model : '');
export const disconnect = () => { session = null; try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ } };
export const savedToken = () => { try { return localStorage.getItem(KEY) || ''; } catch (e) { return ''; } };

async function call(token, model, messages, maxTokens, timeoutMs, temperature = 0.2) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature, stream: false }),
      signal: ctl.signal,
    });
    if (!res.ok) return { ok: false, status: res.status };
    const json = await res.json();
    const text = String((json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content) || '')
      .replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
    return text ? { ok: true, text } : { ok: false, status: 0 };
  } catch (e) {
    return { ok: false, status: 0 };
  } finally {
    clearTimeout(timer);
  }
}

async function complete(messages, maxTokens, timeoutMs = 30000, temperature = 0.2) {
  if (!session) return null;
  const order = [session.model, ...MODELS.filter((m) => m !== session.model)];
  for (const m of order) {
    const r = await call(session.token, m, messages, maxTokens, timeoutMs, temperature);
    if (r.ok) { session.model = m; return r.text; }
    if ([401, 402, 403, 429].includes(r.status)) return null;
  }
  return null;
}

/** Validate a token with a tiny request. Returns {ok, reason}. */
export async function connect(token, remember = false) {
  const t = String(token || '').trim();
  if (t.length < 10) return { ok: false, reason: 'Please paste a Hugging Face token.' };
  let last = { status: 0 };
  for (const m of MODELS) {
    const r = await call(t, m, [{ role: 'user', content: 'Reply with the word OK.' }], 40, 25000);
    if (r.ok) {
      session = { token: t, model: m };
      try { if (remember) localStorage.setItem(KEY, t); else localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
      return { ok: true, reason: '' };
    }
    last = r;
    if (r.status === 401 || r.status === 403) break;
  }
  session = null;
  if (last.status === 401 || last.status === 403) return { ok: false, reason: 'That token was not accepted. Create one that can call Inference Providers.' };
  if (last.status === 402 || last.status === 429) return { ok: false, reason: 'The free quota for this token is used up right now. Try again later.' };
  return { ok: false, reason: 'Could not reach the AI service. Check your connection and try again.' };
}

/** Every meaningful number in the text must already appear in the facts (bare small integers such as list numbers are ignored). */
export function numbersGrounded(text, facts) {
  const hay = String(facts).replace(/,/g, '');
  const nums = String(text).replace(/,/g, '').match(/\d+(?:\.\d+)?/g) || [];
  return nums.every((n) => (!n.includes('.') && Number(n) <= 12) || hay.includes(n));
}


/** Claims a number check cannot catch: saying everything passed when the fact sheet records warnings or failures. */
export function contradicts(text, ctx = {}) {
  const t = String(text).toLowerCase();
  const issues = (ctx.warn || 0) + (ctx.fail || 0);
  const allPassed = /\ball\b[^.]{0,40}\b(checks?|tests?|validations?|controls?)\b[^.]{0,20}\b(pass|passed|passes|met|green)\b/;
  if (issues > 0 && (allPassed.test(t) || /\bno (warnings?|concerns?|issues?|flags?)\b/.test(t) || /\bpasses all\b/.test(t))) return true;
  if ((ctx.fail || 0) > 0 && /\b(ready for deployment|approve for deployment|safe to deploy)\b/.test(t)) return true;
  return false;
}

const clean = (t) => t.replace(/\*\*/g, '').replace(/^#+\s*/gm, '').trim();
const NL = '\n';

export async function writeMemo(facts, ctx = {}) {
  if (!session) return null;
  const system = [
    'You are a senior model risk analyst writing a one-page memo for a credit risk committee.',
    'Use ONLY the facts provided. Quote every number exactly as written in the facts; never compute, round or invent numbers.',
    'Quote the "Governance verdict" and "Recommendation" facts faithfully. Never say all checks passed unless there are 0 warnings and 0 failures; name each check that has a warning.',
    'Structure: "Summary" (2 sentences), "Performance", "Business value", "Risks and controls", "Recommendation" (1 sentence). Plain text, no markdown, under 230 words.',
    '', 'FACTS:', ...facts,
  ].join(NL);
  const sheet = facts.join(NL);
  for (const temperature of [0, 0.3, 0.3]) {
    const text = await complete([{ role: 'system', content: system }, { role: 'user', content: 'Write the memo.' }], 520, 45000, temperature);
    if (text && numbersGrounded(text, sheet) && !contradicts(text, ctx)) return clean(text);
  }
  return null;
}

export async function answer(question, facts, ctx = {}) {
  if (!session) return null;
  const system = [
    'You answer questions about one credit risk model for a business audience, using ONLY the facts below.',
    'Quote numbers exactly as written; never compute or invent numbers. If the facts do not contain the answer, say so briefly.',
    'For deployment questions, repeat the "Governance verdict" and name the checks that have warnings. Never say all checks passed unless there are 0 warnings and 0 failures.',
    'Answer in at most four sentences, plain text.', '', 'FACTS:', ...facts,
  ].join(NL);
  const sheet = facts.join(NL);
  for (const temperature of [0, 0.3]) {
    const text = await complete([{ role: 'system', content: system }, { role: 'user', content: String(question).slice(0, 300) }], 300, 30000, temperature);
    if (!text) return null;
    const c = clean(text);
    if (numbersGrounded(c, sheet) && !contradicts(c, ctx)) return c;
  }
  return null;
}
