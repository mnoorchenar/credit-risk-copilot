// Optional AI layer using Hugging Face Inference Providers (OpenAI-compatible). The user pastes their own free token; it is
// kept in memory (and in this browser's localStorage only if they tick "remember"). It is sent only to router.huggingface.co.
// Every failure resolves to null so callers fall back to templates; nothing here throws.

const ENDPOINT = 'https://router.huggingface.co/v1/chat/completions';
// Curated list: each was checked to answer through Hugging Face Inference Providers. Which ones work for you depends on the
// providers enabled on your account, so a custom model id can be typed in as well.
export const MODEL_OPTIONS = [
  { id: 'Qwen/Qwen3-4B-Instruct-2507', label: 'Qwen3 4B Instruct', note: 'small, follows instructions well' },
  { id: 'google/gemma-3-4b-it', label: 'Gemma 3 4B', note: 'small and fast' },
  { id: 'meta-llama/Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B', note: 'fast, solid all-rounder' },
  { id: 'meta-llama/Llama-3.3-70B-Instruct', label: 'Llama 3.3 70B', note: 'larger, better writing' },
  { id: 'Qwen/Qwen2.5-72B-Instruct', label: 'Qwen 2.5 72B', note: 'larger, strong reasoning' },
  { id: 'Qwen/Qwen3-235B-A22B-Instruct-2507', label: 'Qwen3 235B (MoE)', note: 'largest, highest quality' },
  { id: 'deepseek-ai/DeepSeek-V3', label: 'DeepSeek V3', note: 'large, strong writing' },
];
export const MODELS = MODEL_OPTIONS.map((m) => m.id);
const KEY = 'crc-hf-token';
const MODEL_KEY = 'crc-hf-model';

let session = null; // { token, model }

export const llmLive = () => session !== null;
export const llmModel = () => (session ? session.model : '');
export const disconnect = () => { session = null; try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ } };
export const savedModel = () => { try { return localStorage.getItem(MODEL_KEY) || ''; } catch (e) { return ''; } };
export function setModel(id) {
  const m = String(id || '').trim();
  if (!m || !session) return false;
  session.model = m; session.preferred = m;
  try { localStorage.setItem(MODEL_KEY, m); } catch (e) { /* ignore */ }
  return true;
}
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
  const first = session.preferred || session.model;
  const order = [first, ...MODELS.filter((m) => m !== first)];
  for (const m of order) {
    const r = await call(session.token, m, messages, maxTokens, timeoutMs, temperature);
    if (r.ok) { session.model = m; return r.text; }
    if ([401, 402, 403, 429].includes(r.status)) return null;
  }
  return null;
}

/** Validate a token with a tiny request. Returns {ok, reason}. */
export async function connect(token, remember = false, model = '') {
  const tk = String(token || '').trim();
  if (tk.length < 10) return { ok: false, reason: 'Please paste a Hugging Face token.' };
  const want = String(model || savedModel() || MODELS[0]).trim();
  const tryList = [want, ...MODELS.filter((m) => m !== want)];
  let last = { status: 0 };
  for (const m of tryList) {
    const r = await call(tk, m, [{ role: 'user', content: 'Reply with the word OK.' }], 40, 25000);
    if (r.ok) {
      session = { token: tk, model: m, preferred: want };
      try { if (remember) localStorage.setItem(KEY, tk); else localStorage.removeItem(KEY); localStorage.setItem(MODEL_KEY, want); } catch (e) { /* ignore */ }
      return { ok: true, reason: '', usedFallback: m !== want, model: m };
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
export const SCOPE = 'I can only answer questions about this model\'s data, performance, explanations, decision cutoffs and validation results. Try: "Which features drive defaults?" or "Should we deploy this model?"';

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

export async function answer(question, facts, ctx = {}, opts = {}) {
  if (!session) return null;
  const page = opts.page || 'this page';
  const scope = opts.scope || SCOPE;
  const system = [
    `You are the assistant on the "${page}" page of a credit risk model studio, answering for a business audience using ONLY the facts below. The facts are ordered with the most relevant to the question first.`,
    'Quote numbers exactly as written; never compute, round or invent numbers. If the facts do not contain the answer, say so briefly.',
    'Scope: you can only discuss this model, its data, training, performance, explanations, decisions and validation as covered by the facts. For anything else (about yourself, general knowledge, other topics) reply exactly: "' + scope + '"',
    'Mention the "Governance verdict" and the checks with warnings ONLY when the question is about deployment, approval, risk or validation. Never say all checks passed unless there are 0 warnings and 0 failures.',
    'Answer in at most four sentences, plain text. Use earlier messages only to resolve follow-up questions.',
    '', 'FACTS:', ...facts,
  ].join(NL);
  const sheet = facts.join(NL);
  const turns = (opts.history || []).filter((m) => m.text).map((m) => ({ role: m.role === 'user' ? 'user' : 'assistant', content: String(m.text).slice(0, 600) }));
  for (const temperature of [0, 0.3]) {
    const text = await complete([{ role: 'system', content: system }, ...turns, { role: 'user', content: String(question).slice(0, 300) }], 320, 30000, temperature);
    if (!text) return null;
    const c = clean(text);
    if (numbersGrounded(c, sheet) && !contradicts(c, ctx)) return c;
  }
  return null;
}
