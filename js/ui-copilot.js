import { buildFacts, templateMemo, computeGovernance } from './governance.js';
import { shapSample } from './pipeline.js';
import { writeMemo, llmLive, llmModel, setModel, MODEL_OPTIONS } from './llm.js';
import { ask as askChat, resetChat, hasHistory } from './chat.js';
import { renderMessages } from './widget.js';
import { $, esc, destroyCharts, badge } from './common.js';

let view = 'challenger';
let memo = null; // { text, source }

const SUGGEST = ['Should we deploy this model?', 'Which features drive defaults?', 'How does it compare to the other model?', 'What are the biggest risks?', 'How much better is it than the current rules?'];

let renderToken = 0;
export async function render(S) {
  const el = $('#tab-copilot');
  destroyCharts(el);
  const my = ++renderToken;
  if (!S.shap[view] && S.models[view]) {
    el.innerHTML = '<div class="card"><h2>Preparing the fact sheet…</h2><p class="lead">Computing SHAP drivers so the memo and answers can cite them.</p></div>';
    await shapSample(S, view);
    if (my !== renderToken) return;
  }
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!have.length) { el.innerHTML = '<div class="card"><h2>Train a model first</h2><p class="lead">Go to the Train tab.</p></div>'; return; }
  if (!S.models[view]) view = have[0];
  const facts = buildFacts(S, view);
  const gv = computeGovernance(S, view); const ctx = { warn: gv.summary.warn, fail: gv.summary.fail };
  if (!memo || memo.view !== view || memo.stamp !== S.models[view].trainedAt) memo = { view, stamp: S.models[view].trainedAt, text: templateMemo(S, view), source: 'template' };
  el.innerHTML = `
    <div class="hero"><h2>From model output to committee-ready message</h2>
    <p>Senior stakeholders need a decision, not a notebook. The copilot writes the memo and answers questions <b>only from numbers this app computed</b>. Any figure the AI states that is not in the fact sheet is rejected, and a template is used instead.</p></div>
    <div class="card"><div class="card-head"><div><h2>Executive model risk memo</h2><p class="lead" id="memo-src"></p></div><div style="display:flex;gap:8px;flex-wrap:wrap"><div class="seg" id="cp-seg">${have.map((k) => `<button data-v="${k}" class="${k === view ? 'on' : ''}">${k === 'champion' ? 'Champion' : 'Challenger'}</button>`).join('')}</div>${modelSelect()}<button class="btn primary" id="gen">${llmLive() ? 'Write with AI' : 'Refresh template'}</button><button class="btn" id="copy">Copy</button></div></div>
      <div class="memo" id="memo"></div></div>
    <div class="grid g21">
      <div class="card"><div class="card-head"><div><h2>Ask about the model</h2><p class="lead">${llmLive() ? `Answered by ${esc(llmModel())} from the fact sheet.` : 'AI is not connected, so answers are pulled directly from the fact sheet. Connect AI (top right) for written answers.'}</p></div></div>
        <div class="chips" id="sg">${SUGGEST.map((s) => `<button class="chip">${esc(s)}</button>`).join('')}</div>
        <div class="msgs" id="qa" style="margin-top:10px"></div>
        <form class="ask" id="ask"><input type="text" id="q" placeholder="e.g. Why is the challenger better?" maxlength="200" aria-label="Question about the model"><button class="btn primary" type="submit">Ask</button><button class="btn" type="button" id="qa-reset" title="Clear this conversation">↻ Reset</button></form></div>
      <div class="card"><div class="card-head"><div><h2>Fact sheet</h2><p class="lead">The only material the AI may quote.</p></div></div><div class="scroll" style="max-height:420px"><ul class="list" style="padding:8px 10px">${facts.map((f) => `<li>${esc(f)}</li>`).join('')}</ul></div></div>
    </div>`;
  paintMemo();
  renderMessages($('#qa'), 'copilot');
  $('#qa-reset').disabled = !hasHistory('copilot');
  $('#cp-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; view = b.dataset.v; render(S); });
  const cpm = $('#cp-model'); if (cpm) cpm.addEventListener('change', () => { setModel(cpm.value); document.dispatchEvent(new CustomEvent('ai-changed')); });
  $('#gen').addEventListener('click', async () => {
    const btn = $('#gen'); btn.disabled = true;
    if (!llmLive()) { memo = { view, stamp: S.models[view].trainedAt, text: templateMemo(S, view), source: 'template' }; paintMemo(); btn.disabled = false; return; }
    btn.textContent = 'Writing…'; $('#memo').textContent = 'The model is drafting the memo from the fact sheet…';
    const t = await writeMemo(facts, ctx);
    memo = t ? { view, stamp: S.models[view].trainedAt, text: t, source: 'ai' } : { view, stamp: S.models[view].trainedAt, text: templateMemo(S, view), source: 'fallback' };
    paintMemo(); btn.disabled = false; btn.textContent = 'Write with AI';
  });
  $('#copy').addEventListener('click', () => { try { navigator.clipboard.writeText(memo.text); $('#copy').textContent = 'Copied'; setTimeout(() => { $('#copy').textContent = 'Copy'; }, 1500); } catch (e) { /* ignore */ } });
  $('#sg').addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (b) ask(S, b.textContent); });
  $('#ask').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#q').value; $('#q').value = ''; ask(S, v); });
  $('#qa-reset').addEventListener('click', () => { resetChat('copilot'); renderMessages($('#qa'), 'copilot'); $('#qa-reset').disabled = true; });
}

function paintMemo() {
  $('#memo').textContent = memo.text;
  $('#memo-src').innerHTML = memo.source === 'ai' ? `${badge('pass', 'AI-written')} every number verified against the fact sheet` : memo.source === 'fallback' ? `${badge('warn', 'Template')} the AI draft was rejected (it quoted a figure not in the fact sheet or overstated the validation results), so the template was used` : `${badge('info', 'Template')} generated from the computed results${llmLive() ? '; press "Write with AI" for a written version' : ''}`;
}

function modelSelect() {
  if (!llmLive()) return '';
  const cur = llmModel();
  const extra = MODEL_OPTIONS.some((m) => m.id === cur) ? '' : `<option value="${esc(cur)}" selected>${esc(cur)}</option>`;
  return `<select id="cp-model" aria-label="Language model" title="Language model">${extra}${MODEL_OPTIONS.map((m) => `<option value="${m.id}" ${m.id === cur ? 'selected' : ''}>${esc(m.label)}</option>`).join('')}</select>`;
}

async function ask(S, q) {
  const question = String(q || '').trim(); if (!question) return;
  const box = $('#qa'); const btn = $('#qa-reset');
  btn.disabled = true;
  box.insertAdjacentHTML('beforeend', `<div class="m u">${esc(question)}</div><div class="m b"><span class="think"><i></i><i></i><i></i></span></div>`);
  box.scrollTop = box.scrollHeight;
  await askChat(S, 'copilot', question, () => window.__getStates());
  renderMessages(box, 'copilot');
  btn.disabled = false;
}

export const state = () => ({ view });
