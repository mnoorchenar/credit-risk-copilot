import { $, $$, busy } from './common.js';
import { createWorkspace, trainModel, DEFAULTS } from './pipeline.js';
import { connect, disconnect, savedToken, savedModel, setModel, llmLive, llmModel, MODEL_OPTIONS } from './llm.js';
import * as data from './ui-data.js';
import * as train from './ui-train.js';
import * as evaluate from './ui-eval.js';
import * as explain from './ui-explain.js';
import * as decide from './ui-decide.js';
import * as govern from './ui-govern.js';
import * as copilot from './ui-copilot.js';

const TABS = { data, train, eval: evaluate, explain, decide, govern, copilot };
let S = null;
let active = 'data';
const dirty = new Set(Object.keys(TABS));

function show(tab) {
  active = tab;
  $$('.tab').forEach((t) => { const on = t.dataset.tab === tab; t.classList.toggle('on', on); t.setAttribute('aria-selected', String(on)); });
  $$('.panel').forEach((p) => { p.hidden = p.id !== `tab-${tab}`; });
  if (S && dirty.has(tab)) { dirty.delete(tab); TABS[tab].render(S); }
  try { history.replaceState(null, '', `#${tab}`); } catch (e) { /* ignore */ }
  window.scrollTo({ top: 0 });
}

function invalidate(except) { for (const k of Object.keys(TABS)) if (k !== except) dirty.add(k); }

function refreshPill() {
  const live = llmLive();
  $('#ai-pill').classList.toggle('live', live);
  $('#ai-text').textContent = live ? `AI live · ${llmModel().split('/').pop()}` : 'Connect AI';
}
function paintAi() {
  refreshPill();
  document.dispatchEvent(new CustomEvent('ai-changed'));
}

const CUSTOM = '__custom__';
function modelId() {
  const sel = $('#ai-model').value;
  return sel === CUSTOM ? $('#ai-custom').value.trim() : sel;
}
function fillModels() {
  const sel = $('#ai-model');
  const want = savedModel();
  const known = MODEL_OPTIONS.some((m) => m.id === want);
  sel.innerHTML = MODEL_OPTIONS.map((m) => `<option value="${m.id}">${m.label} · ${m.note}</option>`).join('') + `<option value="${CUSTOM}">Custom model id…</option>`;
  sel.value = known ? want : want ? CUSTOM : MODEL_OPTIONS[0].id;
  $('#ai-custom').hidden = sel.value !== CUSTOM;
  if (sel.value === CUSTOM) $('#ai-custom').value = want;
}

function wireAi() {
  const pop = $('#ai-pop'); const pill = $('#ai-pill');
  const toggle = (open) => { pop.hidden = !open; pill.setAttribute('aria-expanded', String(open)); if (open) $('#ai-token').focus(); };
  fillModels();
  const switched = () => { if (llmLive() && modelId()) { setModel(modelId()); $('#ai-msg').textContent = `Switched to ${modelId().split('/').pop()}.`; paintAi(); } };
  $('#ai-model').addEventListener('change', () => { $('#ai-custom').hidden = $('#ai-model').value !== CUSTOM; if ($('#ai-model').value !== CUSTOM) switched(); });
  $('#ai-custom').addEventListener('change', switched);
  pill.addEventListener('click', () => toggle(pop.hidden));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') toggle(false); });
  document.addEventListener('click', (e) => { if (!pop.hidden && !e.target.closest('.ai-wrap')) toggle(false); });
  $('#ai-connect').addEventListener('click', async () => {
    const btn = $('#ai-connect'); btn.disabled = true; $('#ai-msg').textContent = 'Checking token…';
    const r = await connect($('#ai-token').value, $('#ai-remember').checked, modelId());
    btn.disabled = false;
    $('#ai-msg').textContent = r.ok ? (r.usedFallback ? `Connected, but your chosen model did not answer, so ${r.model.split('/').pop()} is being used.` : 'Connected. The memo and Q&A now use AI.') : r.reason;
    if (r.ok) { $('#ai-token').value = ''; setTimeout(() => toggle(false), 800); dirty.add('copilot'); if (active === 'copilot') show('copilot'); }
    paintAi();
  });
  $('#ai-off').addEventListener('click', () => { disconnect(); $('#ai-msg').textContent = 'Disconnected. Using built-in templates.'; paintAi(); dirty.add('copilot'); });
}

function syncTheme() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  $('#theme-btn').textContent = dark ? 'Light' : 'Dark';
  $('#theme-btn').setAttribute('aria-pressed', String(dark));
}

function wire() {
  $('#tabs').addEventListener('click', (e) => { const b = e.target.closest('.tab'); if (b) show(b.dataset.tab); });
  $('#theme-btn').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('crc-theme', next); } catch (e) { /* ignore */ }
    syncTheme();
    if (train.isTraining()) return;
    invalidate(); dirty.delete(active); TABS[active].render(S);
  });
  document.addEventListener('models-changed', () => invalidate('train'));
  document.addEventListener('ai-changed', () => { refreshPill(); dirty.add('copilot'); if (active === 'copilot') { dirty.delete('copilot'); copilot.render(S); } });
  document.addEventListener('cutoffs-changed', () => { for (const k of ['explain', 'govern', 'copilot']) dirty.add(k); });
  document.addEventListener('training-state', (e) => {
    const pill = $('#ws-pill');
    pill.classList.toggle('work', e.detail.active);
    $('#ws-text').textContent = e.detail.active ? `Training ${e.detail.which}…` : 'Workspace ready';
  });
  wireAi();
}

async function boot() {
  syncTheme();
  wire();
  busy('Preparing workspace…', 0.02);
  $('#ws-pill').classList.add('work');
  try {
    S = await createWorkspace((m, f) => { $('#busy-msg').textContent = m; $('#busy-bar').style.width = `${Math.round(f * 40)}%`; });
    busy('Training the champion scorecard…', 0.45);
    await trainModel(S, 'champion', DEFAULTS.champion, (pt) => busy('Training the champion scorecard…', 0.4 + 0.15 * (pt.iter / pt.total)));
    busy('Training the gradient-boosting challenger…', 0.6);
    await trainModel(S, 'challenger', DEFAULTS.challenger, (pt) => busy('Training the gradient-boosting challenger…', 0.55 + 0.4 * (pt.iter / pt.total)));
  } catch (err) {
    busy(null);
    $('#tab-data').innerHTML = `<div class="card"><h2>Something went wrong</h2><p class="lead">${String(err && err.message || err)}</p></div>`;
    console.error(err);
    return;
  }
  busy(null);
  $('#ws-pill').classList.remove('work'); $('#ws-pill').classList.add('live'); $('#ws-text').textContent = 'Workspace ready';
  const saved = savedToken();
  const start = (location.hash || '#data').slice(1);
  show(TABS[start] ? start : 'data');
  if (saved) { await connect(saved, true, savedModel()); paintAi(); dirty.add('copilot'); }
  window.__studio = { S, show };
}
boot();
