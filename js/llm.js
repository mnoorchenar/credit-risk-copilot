// Optional AI layer using Hugging Face Inference Providers (OpenAI-compatible). The user pastes their own free token; it is
// kept in memory (and only in this browser's localStorage if they tick "remember"). It is never sent anywhere except
// router.huggingface.co. Every failure resolves to null so the caller falls back to the rules engine; nothing here throws.
import { METRICS, GROUPS, validateQuery, numbersGrounded } from './engine.js';

const ENDPOINT = 'https://router.huggingface.co/v1/chat/completions';
export const MODELS = ['Qwen/Qwen3-4B-Instruct-2507', 'google/gemma-3-4b-it', 'meta-llama/Llama-3.1-8B-Instruct'];
const KEY = 'crc-hf-token';

let session = null; // { token, model }

export const llmLive = () => session !== null;
export const llmModel = () => (session ? session.model : '');
export const disconnect = () => { session = null; try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ } };
export const savedToken = () => { try { return localStorage.getItem(KEY) || ''; } catch (e) { return ''; } };

async function call(token, model, messages, maxTokens, timeoutMs) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ model, messages, max_tokens: maxTokens, temperature: 0, stream: false }),
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

/** Try the preferred model first, then the others. Returns text or null. */
async function complete(messages, maxTokens, timeoutMs) {
  if (!session) return null;
  const order = [session.model, ...MODELS.filter((m) => m !== session.model)];
  for (const m of order) {
    const r = await call(session.token, m, messages, maxTokens, timeoutMs);
    if (r.ok) { session.model = m; return r.text; }
    if (r.status === 401 || r.status === 403 || r.status === 402 || r.status === 429) return null;
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

function extractJson(text) {
  if (!text) return null;
  const a = text.indexOf('{');
  const b = text.lastIndexOf('}');
  if (a < 0 || b <= a) return null;
  try { return JSON.parse(text.slice(a, b + 1)); } catch (e) { return null; }
}

function interpretPrompt(model, ctx) {
  const metrics = Object.entries(METRICS).map(([k, m]) => `${k}: ${m.blurb}`).join('\n');
  const names = model.base.map((r) => r.short).join(', ');
  const groups = Object.entries(GROUPS).map(([k, g]) => `${k} = ${g.label}`).join('; ');
  return [
    'You convert a question about a residential secured lending portfolio in Ontario into ONE JSON query. Output JSON only, no prose, no code fences.',
    '',
    'Schema:',
    '{"intent": "rank|profile|compare|stress|drivers|trend|overview|reset|help",',
    ' "metric": one metric key (default "risk"),',
    ' "order": "worst|best|high|low" (worst = riskiest first),',
    ' "n": number of regions to return (1-10, default 5),',
    ' "filters": [{"metric": key, "op": ">|<|worse_third|better_third", "value": number or null}],',
    ' "regions": [region names mentioned],',
    ' "scope": null or one of the group keys,',
    ' "scenario": {"rate_bps": number, "unemp_pp": number, "hpi_pct": number}  (stress only; hpi_pct negative = price fall)}',
    '',
    'Metric keys:',
    metrics,
    '',
    `Regions: ${names}`,
    `Groups: ${groups}`,
    '',
    'Rules: "rising/deteriorating delinquency" = filter dpd_trend > 0. "high X" = filter worse_third on X. "why/explain" = drivers. "trend/over time" = trend. "what if/stress/shock" = stress. Use the selected region when the user says "here" or "this region".',
    ctx.regionName ? `Selected region: ${ctx.regionName}.` : 'No region is selected.',
    '',
    'Examples:',
    'Q: which 3 areas have the worst credit risk? -> {"intent":"rank","metric":"risk","order":"worst","n":3,"filters":[],"regions":[],"scope":null}',
    'Q: high LTV and rising delinquency -> {"intent":"rank","metric":"risk","order":"worst","n":5,"filters":[{"metric":"ltv","op":"worse_third","value":null},{"metric":"dpd_trend","op":">","value":0}],"regions":[],"scope":null}',
    'Q: compare Toronto and Ottawa -> {"intent":"compare","regions":["Toronto","Ottawa"]}',
    'Q: what if rates go up 200 bps and unemployment up 2 points -> {"intent":"stress","scenario":{"rate_bps":200,"unemp_pp":2,"hpi_pct":0}}',
  ].join('\n');
}

/** Returns a validated query object or null. */
export async function interpret(question, model, ctx) {
  if (!session) return null;
  const text = await complete([
    { role: 'system', content: interpretPrompt(model, ctx) },
    { role: 'user', content: `Q: ${String(question).slice(0, 300)}` },
  ], 260, 14000);
  return validateQuery(extractJson(text), model);
}

/** Two-sentence executive summary written from the facts. Returns text or null (also null if any number is not in the facts). */
export async function narrate(question, result) {
  if (!session || !result.facts || !result.facts.length) return null;
  const facts = result.facts.join('\n');
  const text = await complete([
    {
      role: 'system',
      content: [
        'You are a senior credit risk analyst writing for executives.',
        'Write exactly two short sentences answering the question, using ONLY the facts below.',
        'Quote numbers exactly as written in the facts and attach each number to the exact metric name it has there (a risk score is not an LTV). Mention at most four numbers. Never compute or invent numbers. No bullet points, no markdown.',
        '',
        'FACTS:',
        facts,
      ].join('\n'),
    },
    { role: 'user', content: String(question).slice(0, 300) },
  ], 160, 14000);
  if (!text) return null;
  const clean = text.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim();
  return numbersGrounded(clean, facts) && clean.length < 600 ? clean : null;
}
