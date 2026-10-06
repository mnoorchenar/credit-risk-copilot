import { buildFacts, templateMemo, computeGovernance } from './governance.js';
import { writeMemo, answer, llmLive, llmModel } from './llm.js';
import { $, esc, destroyCharts, badge } from './common.js';

let view = 'challenger';
let memo = null; // { text, source }

const SUGGEST = ['Should we deploy this model?', 'Which features drive defaults?', 'How does it compare to the other model?', 'What are the biggest risks?', 'How much better is it than the current rules?'];

export function render(S) {
  const el = $('#tab-copilot');
  destroyCharts(el);
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!have.length) { el.innerHTML = '<div class="card"><h2>Train a model first</h2><p class="lead">Go to the Train tab.</p></div>'; return; }
  if (!S.models[view]) view = have[0];
  const facts = buildFacts(S, view);
  const gv = computeGovernance(S, view); const ctx = { warn: gv.summary.warn, fail: gv.summary.fail };
  if (!memo || memo.view !== view || memo.stamp !== S.models[view].trainedAt) memo = { view, stamp: S.models[view].trainedAt, text: templateMemo(S, view), source: 'template' };
  el.innerHTML = `
    <div class="hero"><h2>From model output to committee-ready message</h2>
    <p>Senior stakeholders need a decision, not a notebook. The copilot writes the memo and answers questions <b>only from numbers this app computed</b>. Any figure the AI states that is not in the fact sheet is rejected, and a template is used instead.</p></div>
    <div class="card"><div class="card-head"><div><h2>Executive model risk memo</h2><p class="lead" id="memo-src"></p></div><div style="display:flex;gap:8px;flex-wrap:wrap"><div class="seg" id="cp-seg">${have.map((k) => `<button data-v="${k}" class="${k === view ? 'on' : ''}">${k === 'champion' ? 'Champion' : 'Challenger'}</button>`).join('')}</div><button class="btn primary" id="gen">${llmLive() ? 'Write with AI' : 'Refresh template'}</button><button class="btn" id="copy">Copy</button></div></div>
      <div class="memo" id="memo"></div></div>
    <div class="grid g21">
      <div class="card"><div class="card-head"><div><h2>Ask about the model</h2><p class="lead">${llmLive() ? `Answered by ${esc(llmModel())} from the fact sheet.` : 'AI is not connected, so answers are pulled directly from the fact sheet. Connect AI (top right) for written answers.'}</p></div></div>
        <div class="chips" id="sg">${SUGGEST.map((s) => `<button class="chip">${esc(s)}</button>`).join('')}</div>
        <div class="msgs" id="qa" style="margin-top:10px"></div>
        <form class="ask" id="ask"><input type="text" id="q" placeholder="e.g. Why is the challenger better?" maxlength="200" aria-label="Question about the model"><button class="btn primary" type="submit">Ask</button></form></div>
      <div class="card"><div class="card-head"><div><h2>Fact sheet</h2><p class="lead">The only material the AI may quote.</p></div></div><div class="scroll" style="max-height:420px"><ul class="list" style="padding:8px 10px">${facts.map((f) => `<li>${esc(f)}</li>`).join('')}</ul></div></div>
    </div>`;
  paintMemo();
  $('#cp-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; view = b.dataset.v; render(S); });
  $('#gen').addEventListener('click', async () => {
    const btn = $('#gen'); btn.disabled = true;
    if (!llmLive()) { memo = { view, stamp: S.models[view].trainedAt, text: templateMemo(S, view), source: 'template' }; paintMemo(); btn.disabled = false; return; }
    btn.textContent = 'Writing…'; $('#memo').textContent = 'The model is drafting the memo from the fact sheet…';
    const t = await writeMemo(facts, ctx);
    memo = t ? { view, stamp: S.models[view].trainedAt, text: t, source: 'ai' } : { view, stamp: S.models[view].trainedAt, text: templateMemo(S, view), source: 'fallback' };
    paintMemo(); btn.disabled = false; btn.textContent = 'Write with AI';
  });
  $('#copy').addEventListener('click', () => { try { navigator.clipboard.writeText(memo.text); $('#copy').textContent = 'Copied'; setTimeout(() => { $('#copy').textContent = 'Copy'; }, 1500); } catch (e) { /* ignore */ } });
  $('#sg').addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (b) ask(S, facts, b.textContent, ctx); });
  $('#ask').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#q').value; $('#q').value = ''; ask(S, facts, v, ctx); });
}

function paintMemo() {
  $('#memo').textContent = memo.text;
  $('#memo-src').innerHTML = memo.source === 'ai' ? `${badge('pass', 'AI-written')} every number verified against the fact sheet` : memo.source === 'fallback' ? `${badge('warn', 'Template')} the AI draft was rejected (it quoted a figure not in the fact sheet or overstated the validation results), so the template was used` : `${badge('info', 'Template')} generated from the computed results${llmLive() ? '; press "Write with AI" for a written version' : ''}`;
}

const STOP = new Set(['the', 'a', 'an', 'is', 'are', 'of', 'to', 'we', 'it', 'this', 'that', 'how', 'what', 'which', 'does', 'do', 'and', 'for', 'in', 'on', 'than', 'with', 'should', 'much', 'be']);
function retrieve(question, facts) {
  const words = question.toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((w) => w.length > 2 && !STOP.has(w));
  const syn = { deploy: ['governance', 'recommend', 'checks'], drive: ['shap', 'drivers'], features: ['shap', 'drivers'], compare: ['versus', 'difference'], risk: ['governance', 'psi', 'drift'], risks: ['governance', 'drift', 'psi'], rules: ['rule', 'policy'], better: ['versus', 'rule'], accurate: ['auc', 'gini'], performance: ['auc', 'ks'], stable: ['psi', 'drift'], calibrated: ['calibration'] };
  const terms = new Set(words.flatMap((w) => [w, ...(syn[w] || [])]));
  const scored = facts.map((f, i) => ({ f, i, s: [...terms].reduce((a, t) => a + (f.toLowerCase().includes(t) ? 1 : 0), 0) })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s || a.i - b.i).slice(0, 3);
  return scored.length ? scored.map((x) => x.f) : facts.slice(0, 2);
}

async function ask(S, facts, q, ctx) {
  const question = String(q || '').trim(); if (!question) return;
  const box = $('#qa');
  const u = document.createElement('div'); u.className = 'm u'; u.textContent = question; box.appendChild(u);
  const b = document.createElement('div'); b.className = 'm b'; b.textContent = 'Thinking…'; box.appendChild(b); box.scrollTop = box.scrollHeight;
  let text = null; let src = '';
  if (llmLive()) { text = await answer(question, facts, ctx); src = text ? `AI · ${llmModel()} · numbers verified` : ''; }
  if (!text) { text = `From the model report: ${retrieve(question, facts).join(' ')}`; src = llmLive() ? 'AI answer rejected (unverified numbers); showing source facts' : 'Pulled from the fact sheet (AI not connected)'; }
  b.innerHTML = `${esc(text)}<small>${esc(src)}</small>`;
  box.scrollTop = box.scrollHeight;
}
