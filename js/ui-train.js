import { $, $$, esc, chart, axes, colors, kpi, badge, destroyCharts, num, pct } from './common.js';
import { DEFAULTS, PRESETS, trainModel } from './pipeline.js';

const SLIDERS = {
  champion: [
    { k: 'epochs', l: 'Training epochs', min: 10, max: 600, step: 10, f: (v) => v },
    { k: 'lr', l: 'Learning rate', min: 0.005, max: 0.2, step: 0.005, f: (v) => v.toFixed(3) },
    { k: 'l2', l: 'L2 regularisation', min: 0, max: 0.5, step: 0.005, f: (v) => v.toFixed(3) },
    { k: 'posWeight', l: 'Default class weight', min: 1, max: 20, step: 1, f: (v) => `${v}x` },
  ],
  challenger: [
    { k: 'trees', l: 'Number of trees', min: 10, max: 600, step: 10, f: (v) => v },
    { k: 'lr', l: 'Learning rate', min: 0.01, max: 0.4, step: 0.01, f: (v) => v.toFixed(2) },
    { k: 'maxDepth', l: 'Tree depth', min: 1, max: 8, step: 1, f: (v) => v },
    { k: 'minChildWeight', l: 'Min. samples per leaf (weight)', min: 1, max: 100, step: 1, f: (v) => v },
    { k: 'lambda', l: 'L2 on leaf values', min: 0, max: 50, step: 1, f: (v) => v },
    { k: 'subsample', l: 'Row subsample', min: 0.4, max: 1, step: 0.05, f: (v) => v.toFixed(2) },
    { k: 'posWeight', l: 'Default class weight', min: 1, max: 20, step: 1, f: (v) => `${v}x` },
    { k: 'patience', l: 'Early stopping patience (0 = off)', min: 0, max: 60, step: 5, f: (v) => (v === 0 ? 'off' : v) },
  ],
};
const params = { champion: { ...DEFAULTS.champion }, challenger: { ...DEFAULTS.challenger } };
let view = 'challenger';
let training = false;
let curves = {};

export const isTraining = () => training;

function controls(which) {
  const p = params[which];
  return SLIDERS[which].map((s) => `<label class="field"><span>${s.l}<b id="v-${which}-${s.k}">${s.f(p[s.k])}</b></span><input type="range" id="s-${which}-${s.k}" min="${s.min}" max="${s.max}" step="${s.step}" value="${p[s.k]}"></label>`).join('');
}

function modelCard(S, which) {
  const e = S.models[which];
  const col = which === 'champion' ? 'var(--c-champ)' : 'var(--c-chal)';
  const info = which === 'champion'
    ? 'Regularised logistic regression fitted with Adam. Interpretable by construction: one coefficient per feature, the format regulators know best.'
    : 'Gradient-boosted decision trees built from scratch with histogram splits and Newton steps. Captures non-linearity and interactions, so it needs explainability tooling (SHAP).';
  return `<div class="card"><div class="card-head"><div><span class="model-tag"><i style="background:${col}"></i>${which === 'champion' ? 'Champion · logistic scorecard' : 'Challenger · gradient boosting'}</span><p class="lead">${info}</p></div></div>
    <div class="preset"><span class="note">Presets</span><div class="seg" id="pre-${which}">${Object.entries(PRESETS[which]).map(([k, v]) => `<button data-p="${k}" title="${esc(v.note)}">${esc(v.label)}</button>`).join('')}</div><span class="note" id="pn-${which}"></span></div>
    <div class="ctl">${controls(which)}</div>
    <div style="display:flex;gap:10px;align-items:center;margin:14px 0 8px"><button class="btn primary" id="go-${which}">${e ? 'Retrain' : 'Train'} model</button><div class="prog" style="flex:1"><span id="pg-${which}" style="width:${e ? 100 : 0}%"></span></div></div>
    <div class="log" id="lg-${which}">${e ? logLine(e) : 'Not trained yet.'}</div></div>`;
}

function logLine(e) {
  const last = e.history[e.history.length - 1];
  const stop = e.history.stopped ? ` Early stopping at iteration ${e.history.stopped}, kept best ${e.bestIter}.` : '';
  return `Done in ${(e.trainMs / 1000).toFixed(1)}s. Validation AUC ${num(e.metrics.val.auc)} · test AUC ${num(e.metrics.test.auc)} · log-loss ${num(last.val.loss, 4)}.${stop}`;
}

export function render(S) {
  const el = $('#tab-train');
  destroyCharts(el);
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!S.models[view] && have.length) view = have[0];
  el.innerHTML = `
    <div class="hero"><h2>Train a champion and a challenger</h2>
    <p>Change the settings and press <b>Train</b>: the learning curves update live. Training loss always keeps falling; what matters is the <b>validation</b> curve. Try the <b>Overfit</b> preset to see a model memorise its training data, then <b>Balanced</b> to see early stopping fix it.</p></div>
    <div class="grid g2">${modelCard(S, 'champion')}${modelCard(S, 'challenger')}</div>
    <div class="card"><div class="card-head"><div><h2>Learning curves</h2><p class="lead" id="curve-lead"></p></div>
      <div class="seg" id="view-seg"><button data-v="champion" class="${view === 'champion' ? 'on' : ''}">Champion</button><button data-v="challenger" class="${view === 'challenger' ? 'on' : ''}">Challenger</button></div></div>
      <div class="grid g2"><div><div class="chart" id="cv-loss"><canvas></canvas></div></div><div><div class="chart" id="cv-auc"><canvas></canvas></div></div></div>
      <div id="diag" style="margin-top:12px"></div></div>
    <div class="card" id="cmp"></div>`;
  for (const which of ['champion', 'challenger']) wire(S, which);
  $('#view-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b || training) return; view = b.dataset.v; $$('#view-seg button').forEach((x) => x.classList.toggle('on', x === b)); drawCurves(S, view, S.models[view]?.history || []); });
  drawCurves(S, view, S.models[view]?.history || []);
  paintCompare(S);
}

function wire(S, which) {
  for (const s of SLIDERS[which]) {
    $(`#s-${which}-${s.k}`).addEventListener('input', (e) => { params[which][s.k] = Number(e.target.value); $(`#v-${which}-${s.k}`).textContent = s.f(params[which][s.k]); });
  }
  $(`#pre-${which}`).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b || training) return;
    const pr = PRESETS[which][b.dataset.p];
    Object.assign(params[which], pr.p);
    for (const s of SLIDERS[which]) { $(`#s-${which}-${s.k}`).value = params[which][s.k]; $(`#v-${which}-${s.k}`).textContent = s.f(params[which][s.k]); }
    $$(`#pre-${which} button`).forEach((x) => x.classList.toggle('on', x === b));
    $(`#pn-${which}`).textContent = pr.note;
  });
  $(`#go-${which}`).addEventListener('click', () => run(S, which));
}

function drawCurves(S, which, history) {
  const c = colors();
  const col = which === 'champion' ? c.champion : c.challenger;
  $('#curve-lead').textContent = which === 'champion' ? 'Logistic regression: one point per group of epochs.' : 'Gradient boosting: one point per group of trees. The marker shows the iteration with the lowest validation loss.';
  const pts = (key, m) => history.map((h) => ({ x: h.iter, y: h[key][m] }));
  const best = history.length ? history.reduce((a, h) => (h.val.loss < a.val.loss ? h : a), history[0]) : null;
  const mk = (id, m, title, fmt) => chart($(`#${id} canvas`), {
    type: 'line',
    data: { datasets: [
      { label: 'Training', data: pts('train', m), borderColor: c.muted, backgroundColor: c.muted, borderDash: [5, 4], borderWidth: 2, pointRadius: 0, tension: 0.2 },
      { label: 'Validation', data: pts('val', m), borderColor: col, backgroundColor: col, borderWidth: 2.5, pointRadius: 0, tension: 0.2 },
      ...(best && m === 'loss' ? [{ label: 'Best validation loss', data: [{ x: best.iter, y: best.val.loss }], borderColor: c.accent, backgroundColor: c.accent, pointRadius: 6, showLine: false }] : []),
    ] },
    options: { animation: false, parsing: false, scales: axes({ xTitle: which === 'champion' ? 'Epoch' : 'Trees', yTitle: title, x: { type: 'linear' } }), plugins: { tooltip: { mode: 'index', intersect: false, callbacks: { label: (x) => `${x.dataset.label}: ${fmt(x.parsed.y)}` } } }, interaction: { mode: 'index', intersect: false } },
  });
  mk('cv-loss', 'loss', 'Log-loss (lower is better)', (v) => v.toFixed(4));
  mk('cv-auc', 'auc', 'AUC (higher is better)', (v) => v.toFixed(3));
  paintDiag(S, which, history);
}

function paintDiag(S, which, history) {
  const el = $('#diag');
  if (!history.length) { el.innerHTML = ''; return; }
  const last = history[history.length - 1];
  const minVal = Math.min(...history.map((h) => h.val.loss));
  const gap = last.train.auc - last.val.auc;
  const lossRise = (last.val.loss - minVal) / minVal;
  let status = 'pass'; let title = 'Healthy fit'; let text = `Training and validation AUC are close (gap ${gap.toFixed(3)}) and validation loss is at its best level.`;
  if (gap > 0.08 || lossRise > 0.08) { status = 'fail'; title = 'Overfitting'; text = `The model scores ${num(last.train.auc)} AUC on data it trained on but only ${num(last.val.auc)} on unseen validation data (gap ${gap.toFixed(3)}), and validation loss has risen ${pct(lossRise, 0)} above its best. It is memorising noise. Add regularisation, shallower trees or early stopping.`; }
  else if (last.val.auc < 0.76) { status = 'warn'; title = 'Underfitting'; text = `Validation AUC of ${num(last.val.auc)} is low and training AUC is similar: the model is too simple or has not trained long enough.`; }
  el.innerHTML = `<div class="callout ${status === 'pass' ? 'ok' : ''}">${badge(status, title)} &nbsp;${esc(text)}</div>`;
}

async function run(S, which) {
  if (training) return;
  training = true; view = which;
  $$('#tab-train .btn, #tab-train .seg button').forEach((b) => { b.disabled = true; });
  $$('#view-seg button').forEach((x) => x.classList.toggle('on', x.dataset.v === which));
  const log = $(`#lg-${which}`); const pg = $(`#pg-${which}`);
  const hist = [];
  drawCurves(S, which, hist);
  document.dispatchEvent(new CustomEvent('training-state', { detail: { which, active: true } }));
  try {
    await trainModel(S, which, params[which], (pt) => {
      hist.push(pt);
      pg.style.width = `${Math.round((pt.iter / pt.total) * 100)}%`;
      log.textContent = `${which === 'champion' ? 'Epoch' : 'Tree'} ${pt.iter}/${pt.total}   train loss ${pt.train.loss.toFixed(4)}   validation loss ${pt.val.loss.toFixed(4)}   validation AUC ${pt.val.auc.toFixed(3)}`;
      if (hist.length % 3 === 0) drawCurvesLive(which, hist);
    });
  } finally { training = false; }
  document.dispatchEvent(new CustomEvent('models-changed', { detail: { which } }));
  document.dispatchEvent(new CustomEvent('training-state', { detail: { which, active: false } }));
  render(S);
}

function drawCurvesLive(which, history) {
  const c = colors();
  const col = which === 'champion' ? c.champion : c.challenger;
  const set = (id, m) => {
    const canvas = $(`#${id} canvas`); const ch = Chart.getChart(canvas); if (!ch) return;
    ch.data.datasets[0].data = history.map((h) => ({ x: h.iter, y: h.train[m] }));
    ch.data.datasets[1].data = history.map((h) => ({ x: h.iter, y: h.val[m] }));
    ch.data.datasets[1].borderColor = col;
    ch.update('none');
  };
  set('cv-loss', 'loss'); set('cv-auc', 'auc');
}

function paintCompare(S) {
  const el = $('#cmp');
  const es = ['champion', 'challenger'].filter((k) => S.models[k]).map((k) => S.models[k]);
  if (es.length < 2) { el.innerHTML = '<div class="card-head"><h2>Champion vs challenger</h2></div><p class="lead">Train both models to compare them.</p>'; return; }
  const [a, b] = es;
  const row = (label, f, better = 'high') => { const x = f(a); const y = f(b); const win = better === 'high' ? (x > y ? 0 : 1) : (x < y ? 0 : 1); return `<tr><td class="l">${label}</td><td ${win === 0 ? 'style="font-weight:800"' : ''}>${x}</td><td ${win === 1 ? 'style="font-weight:800"' : ''}>${y}</td></tr>`; };
  const dAuc = b.metrics.test.auc - a.metrics.test.auc;
  const verdict = Math.abs(dAuc) < 0.005 ? 'The models are statistically indistinguishable on ranking power; the simpler scorecard is the safer choice.'
    : dAuc > 0 ? `The challenger ranks defaults ${(dAuc * 100).toFixed(1)} AUC points better on the test set. Whether that justifies a less transparent model is a governance decision: the Explain and Govern tabs give the evidence.`
      : `The scorecard outperforms the challenger by ${(-dAuc * 100).toFixed(1)} AUC points and is simpler to explain. Keep it as champion.`;
  el.innerHTML = `<div class="card-head"><div><h2>Champion vs challenger</h2><p class="lead">Bold = better. All metrics use the same accounts.</p></div></div>
    <div class="scroll"><table class="t"><thead><tr><th class="l">Metric</th><th><span class="sw" style="background:var(--c-champ)"></span>Champion</th><th><span class="sw" style="background:var(--c-chal)"></span>Challenger</th></tr></thead><tbody>
    ${row('Train AUC', (e) => num(e.metrics.train.auc))}${row('Validation AUC', (e) => num(e.metrics.val.auc))}${row('Test AUC', (e) => num(e.metrics.test.auc))}
    ${row('Test Gini', (e) => num(e.metrics.test.gini))}${row('Test KS', (e) => num(e.metrics.test.ks))}${row('Test log-loss', (e) => num(e.metrics.test.logloss, 4), 'low')}${row('Test Brier score', (e) => num(e.metrics.test.brier, 4), 'low')}
    ${row('Train – validation AUC gap', (e) => num(e.metrics.train.auc - e.metrics.val.auc), 'low')}<tr><td class="l">Training time</td><td>${(a.trainMs / 1000).toFixed(1)}s</td><td>${(b.trainMs / 1000).toFixed(1)}s</td></tr>
    </tbody></table></div><div class="callout info" style="margin-top:12px">${esc(verdict)}</div>`;
}

export const state = () => ({ view, training, params: { champion: { ...params.champion }, challenger: { ...params.challenger } } });
