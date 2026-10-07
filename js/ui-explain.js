import { FEATURES, F, FIDX, fmtFeature, column } from './data.js';
import { shapRow, partialDependence, permutationImportance, predictProba, sigmoid, logit } from './ml.js';
import { reasons, recourse } from './explain.js';
import { shapSample } from './pipeline.js';
import { $, $$, esc, chart, axes, colors, diverge, kpi, badge, pct, destroyCharts, busy } from './common.js';

let view = 'challenger';
let feat = 'credit_score';
let imp = 'shap';
let acct = null;
let token = 0;
const permCache = {};

export function render(S) {
  const el = $('#tab-explain');
  destroyCharts(el);
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!have.length) { el.innerHTML = '<div class="card"><h2>Train a model first</h2><p class="lead">Go to the Train tab.</p></div>'; return; }
  if (!S.models[view]) view = have[0];
  const my = ++token;
  el.innerHTML = `
    <div class="hero"><h2>Why does the model say that?</h2>
    <p>A boosted tree model can beat a scorecard, but only if you can explain it. <b>SHAP</b> splits every prediction into one contribution per feature (exact for trees, verified against brute-force Shapley values). Contributions add up to the prediction, so every number below reconciles.</p></div>
    <div class="card"><div class="card-head"><div><h2>Model under review</h2></div><div class="seg" id="ex-seg">${have.map((k) => `<button data-v="${k}" class="${k === view ? 'on' : ''}">${k === 'champion' ? 'Champion · logistic' : 'Challenger · boosting'}</button>`).join('')}</div></div>
      <div class="callout info">${view === 'champion' ? 'For a logistic scorecard, SHAP reduces to <b>coefficient × standardised value</b>: the model is transparent by construction.' : 'Tree ensembles are not readable directly. SHAP gives each prediction an exact, additive explanation.'}</div></div>
    <div class="grid g21">
      <div class="card"><div class="card-head"><div><h2>Global importance</h2><p class="lead" id="imp-lead"></p></div><div class="seg" id="imp-seg"><button data-v="shap" class="${imp === 'shap' ? 'on' : ''}">Mean |SHAP|</button><button data-v="perm" class="${imp === 'perm' ? 'on' : ''}">Permutation (AUC drop)</button></div></div><div id="imp-bars"><p class="note">Computing…</p></div></div>
      <div class="card"><div class="card-head"><div><h2>How to read the plots</h2></div></div><ul class="list"><li><b>Mean |SHAP|</b>: average size of a feature's push on the prediction.</li><li><b>Permutation</b>: how much AUC drops if the feature is shuffled. Model-agnostic cross-check.</li><li><b>Beeswarm</b>: each dot is an account. Right = raises risk, left = lowers it. Red = high feature value, blue = low.</li></ul></div>
    </div>
    <div class="card"><div class="card-head"><div><h2>SHAP summary (beeswarm)</h2><p class="lead">320 test accounts. Look for the shape: a clean left-to-right colour gradient means a simple monotonic effect; mixed colours mean interactions.</p></div></div><div class="chart" style="height:520px"><canvas id="swarm"></canvas></div></div>
    <div class="card"><div class="card-head"><div><h2>Feature deep-dive</h2><p class="lead" id="dd-lead"></p></div></div><div class="chips" id="dd-chips">${FEATURES.map((f) => `<button class="chip ${f.key === feat ? 'on' : ''}" data-k="${f.key}">${esc(f.short)}</button>`).join('')}</div>
      <div class="grid g2" style="margin-top:12px"><div><div class="chart"><canvas id="pdp"></canvas></div></div><div><div class="chart"><canvas id="dep"></canvas></div></div></div><div id="dd-note" style="margin-top:10px"></div></div>
    <div class="card"><div class="card-head"><div><h2>Explain one decision</h2><p class="lead">Pick an account to see the exact reasons behind its PD, in the form used for adverse-action notices, and what would change the outcome.</p></div></div>
      <div class="chips" id="pick"><button class="chip" data-pick="high">Highest-risk account</button><button class="chip" data-pick="edge">Near the decline cutoff</button><button class="chip" data-pick="rev">Manual-review case</button><button class="chip" data-pick="ok">Typical approval</button><button class="chip" data-pick="rand">Random account</button><label style="display:inline-flex;gap:6px;align-items:center;font-size:12.5px;color:var(--ink2)">Account # <input type="number" id="acct-n" min="0" max="${S.te.n - 1}" style="width:90px"></label></div>
      <div id="local" style="margin-top:14px"></div></div>`;
  $('#ex-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; view = b.dataset.v; acct = null; render(S); });
  $('#imp-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; imp = b.dataset.v; $$('#imp-seg button').forEach((x) => x.classList.toggle('on', x === b)); paintImportance(S, my); });
  $('#dd-chips').addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (!b) return; feat = b.dataset.k; $$('#dd-chips .chip').forEach((x) => x.classList.toggle('on', x === b)); paintDeepDive(S, my); });
  $('#pick').addEventListener('click', (e) => { const b = e.target.closest('[data-pick]'); if (b) choose(S, b.dataset.pick); });
  $('#acct-n').addEventListener('change', (e) => { const v = Number(e.target.value); if (Number.isInteger(v) && v >= 0 && v < S.te.n) { acct = v; paintLocal(S); } });
  prepare(S, my);
}

async function prepare(S, my) {
  const entry = S.models[view];
  if (!S.shap[view]) { if (view === 'challenger') busy('Computing exact TreeSHAP values…', 0.1); await shapSample(S, view, (f) => { if (view === 'challenger') busy('Computing exact TreeSHAP values…', f); }); busy(null); }
  if (my !== token) return;
  paintImportance(S, my);
  paintSwarm(S);
  paintDeepDive(S, my);
  if (acct === null) choose(S, 'edge'); else paintLocal(S);
  void entry;
}

function importance(S) {
  const sh = S.shap[view]; const n = sh.n;
  const out = FEATURES.map((f, j) => { let s = 0; for (let i = 0; i < n; i++) s += Math.abs(sh.values[i * F + j]); return { j, label: f.label, v: s / n }; });
  return out.sort((a, b) => b.v - a.v);
}

async function paintImportance(S, my) {
  const el = $('#imp-bars');
  if (imp === 'shap') {
    const rows = importance(S); const mx = rows[0].v;
    $('#imp-lead').textContent = 'Average absolute contribution to the log-odds of default, over 320 test accounts.';
    el.innerHTML = rows.map((r) => `<div class="bar"><span>${esc(r.label)}</span><span class="track"><span class="fill" style="width:${(r.v / mx) * 100}%"></span></span><span class="v">${r.v.toFixed(3)}</span></div>`).join('');
  } else {
    $('#imp-lead').textContent = 'Drop in test AUC when the feature is randomly shuffled (2 repeats, 1,500 accounts). A model-agnostic cross-check of SHAP.';
    if (!permCache[view] || permCache[view].tag !== S.models[view].trainedAt) {
      el.innerHTML = '<p class="note">Shuffling each feature and re-scoring…</p>';
      const n = 1500; const X = S.te.X.slice(0, n * F); const y = S.te.y.slice(0, n);
      const r = await permutationImportance(S.models[view].model, X, y, n, F, 2);
      permCache[view] = { r, tag: S.models[view].trainedAt };
    }
    if (my !== token) return;
    const r = permCache[view].r;
    const rows = FEATURES.map((f, j) => ({ label: f.label, v: Math.max(0, r.drops[j]) })).sort((a, b) => b.v - a.v);
    const mx = rows[0].v || 1;
    el.innerHTML = rows.map((x) => `<div class="bar"><span>${esc(x.label)}</span><span class="track"><span class="fill" style="width:${(x.v / mx) * 100}%"></span></span><span class="v">${x.v.toFixed(4)}</span></div>`).join('');
  }
}

function paintSwarm(S) {
  const sh = S.shap[view]; const c = colors();
  const order = importance(S).map((r) => r.j);
  const rankOf = new Map(order.map((j, k) => [j, order.length - 1 - k]));
  const pts = [];
  for (const j of order) {
    const col = new Float64Array(sh.n); for (let i = 0; i < sh.n; i++) col[i] = sh.X[i * F + j];
    const s = Float64Array.from(col).sort(); const lo = s[Math.floor(sh.n * 0.05)]; const hi = s[Math.floor(sh.n * 0.95)] || lo + 1;
    for (let i = 0; i < sh.n; i++) {
      const t = FEATURES[j].binary ? col[i] : Math.min(1, Math.max(0, (col[i] - lo) / (hi - lo || 1)));
      pts.push({ x: sh.values[i * F + j], y: rankOf.get(j) + (((i * 7919) % 100) / 100 - 0.5) * 0.7, j, v: col[i], t });
    }
  }
  chart($('#swarm'), {
    type: 'scatter',
    data: { datasets: [{ data: pts, pointRadius: 2.6, pointBackgroundColor: pts.map((p) => diverge(p.t)), pointBorderWidth: 0 }] },
    options: {
      parsing: false, animation: false,
      scales: {
        x: { ...axes({ xTitle: 'SHAP value (impact on log-odds of default)' }).x, grid: { color: c.grid } },
        y: { min: -0.6, max: order.length - 0.4, grid: { color: c.grid }, border: { display: false }, ticks: { color: c.ink2, stepSize: 1, font: { size: 11.5 }, callback: (v) => (Number.isInteger(v) ? FEATURES[order[order.length - 1 - v]]?.label ?? '' : '') } },
      },
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: () => '', label: (x) => { const p = x.raw; return `${FEATURES[p.j].label} = ${fmtFeature(FEATURES[p.j].key, p.v)}  →  SHAP ${p.x >= 0 ? '+' : ''}${p.x.toFixed(3)}`; } } } },
    },
  });
}

function paintDeepDive(S, my) {
  if (my !== token) return;
  const j = FIDX[feat]; const f = FEATURES[j]; const c = colors();
  const entry = S.models[view]; const sh = S.shap[view];
  const col = column(S.te, j); const s = Float64Array.from(col).sort();
  const lo = f.binary ? 0 : s[Math.floor(S.te.n * 0.02)]; const hi = f.binary ? 1 : s[Math.floor(S.te.n * 0.98)];
  const grid = f.binary ? [0, 1] : Array.from({ length: 22 }, (_, k) => lo + ((hi - lo) * k) / 21);
  const n = 400; const idx = Array.from({ length: n }, (_, i) => Math.floor((i * S.te.n) / n));
  const Xs = new Float64Array(n * F); idx.forEach((i, k) => Xs.set(S.te.X.slice(i * F, i * F + F), k * F));
  const pdp = partialDependence(entry.model, Xs, n, F, j, grid, 25);
  const iceCol = c.muted;
  chart($('#pdp'), {
    type: 'line',
    data: { datasets: [
      ...pdp.ice.map((line, k) => ({ label: k === 0 ? 'Individual accounts (ICE)' : '', data: line.map((v, g) => ({ x: grid[g], y: v * 100 })), borderColor: iceCol + '55', borderWidth: 1, pointRadius: 0, tension: 0.25 })),
      { label: 'Average (partial dependence)', data: pdp.mean.map((v, g) => ({ x: grid[g], y: v * 100 })), borderColor: view === 'champion' ? c.champion : c.challenger, backgroundColor: c.challenger, borderWidth: 3.2, pointRadius: f.binary ? 5 : 0, tension: 0.25 },
    ] },
    options: { parsing: false, animation: false, scales: axes({ xTitle: f.label, yTitle: 'Predicted PD %', x: { type: 'linear' } }), plugins: { legend: { labels: { filter: (i) => i.text !== '' } }, tooltip: { filter: (t) => t.dataset.label !== '' } } },
  });
  const sp = []; for (let i = 0; i < sh.n; i++) sp.push({ x: sh.X[i * F + j], y: sh.values[i * F + j] });
  chart($('#dep'), {
    type: 'scatter',
    data: { datasets: [{ label: 'SHAP value', data: sp, pointRadius: 2.8, backgroundColor: (view === 'champion' ? c.champion : c.challenger) + 'aa' }] },
    options: { parsing: false, animation: false, scales: axes({ xTitle: f.label, yTitle: 'SHAP value (log-odds)', x: { type: 'linear' } }), plugins: { legend: { display: false } } },
  });
  $('#dd-lead').textContent = `${f.label}: ${f.desc}. Left: what-if curve (set this feature to each value for every account). Right: the actual contribution it made, account by account; vertical spread at one value means other features interact with it.`;
  const exp = f.sign;
  let viol = 0;
  for (let g = 1; g < pdp.mean.length; g++) if (exp * (pdp.mean[g] - pdp.mean[g - 1]) < -0.03 * (Math.max(...pdp.mean) - Math.min(...pdp.mean))) viol++;
  const dirTxt = exp > 0 ? 'risk should rise as it increases' : exp < 0 ? 'risk should fall as it increases' : 'no fixed expected direction';
  $('#dd-note').innerHTML = exp === 0 ? `<div class="callout info">${esc(f.label)}: ${dirTxt}. Review the shape with the business.</div>` : `<div class="callout ${viol ? '' : 'ok'}">Business-logic check: ${dirTxt}. ${viol ? badge('warn', `${viol} non-monotonic step(s)`) : badge('pass', 'Monotonic')} ${viol ? 'The curve bends against expectation; minor wiggles are usually noise, large ones need review.' : 'The model behaves as an underwriter would expect.'}</div>`;
}

function choose(S, kind) {
  const e = S.models[view]; const p = e.preds.test; const { c1, c2 } = S.cutoffs; const n = p.length;
  let best = 0;
  const pickBy = (fn, initial) => { let b = -1; let bv = initial; for (let i = 0; i < n; i++) { const v = fn(i); if (v < bv) { bv = v; b = i; } } return b; };
  if (kind === 'high') best = pickBy((i) => -p[i], Infinity);
  else if (kind === 'edge') best = pickBy((i) => Math.abs(p[i] - c2) + (i % 97) * 1e-7, Infinity);
  else if (kind === 'rev') best = pickBy((i) => Math.abs(p[i] - (c1 + c2) / 2) + (i % 89) * 1e-7, Infinity);
  else if (kind === 'ok') best = pickBy((i) => Math.abs(p[i] - c1 / 2) + (i % 83) * 1e-7, Infinity);
  else best = Math.floor(Math.random() * n);
  acct = best;
  $('#acct-n').value = best;
  paintLocal(S);
}

function paintLocal(S) {
  const el = $('#local'); if (!el) return;
  const e = S.models[view]; const c = colors();
  const o = acct * F; const X = S.te.X;
  const { phi, base } = shapRow(e.model, X, o, F);
  const margin = base + phi.reduce((a, b) => a + b, 0);
  const pd = sigmoid(margin); const avg = sigmoid(base);
  const { c1, c2 } = S.cutoffs;
  const band = pd < c1 ? ['pass', 'Auto-approve'] : pd < c2 ? ['warn', 'Manual review'] : ['fail', 'Decline'];
  const other = view === 'champion' ? 'challenger' : 'champion';
  const otherPd = S.models[other] ? S.models[other].preds.test[acct] : null;
  const rs = reasons(phi, X, o);
  const rc = band[1] === 'Auto-approve' ? { options: [] } : recourse(e.model, X, o, band[1] === 'Decline' ? c2 : c1);
  const target = band[1] === 'Decline' ? 'Manual review' : 'Auto-approve';
  el.innerHTML = `<div class="grid g21"><div>
      <div class="kpis">${kpi('Predicted PD', pct(pd, 1), `portfolio average ${pct(avg, 1)}`)}${kpi('Decision', badge(band[0], band[1]), `cutoffs ${pct(c1, 1)} / ${pct(c2, 1)}`)}${kpi('Actual outcome', S.te.y[acct] ? 'Defaulted' : 'Did not default', 'known only in backtest')}${otherPd === null ? '' : kpi(`${other === 'champion' ? 'Champion' : 'Challenger'} PD`, pct(otherPd, 1), 'other model')}</div>
      <div class="chart tall" style="height:380px;margin-top:12px"><canvas id="wf"></canvas></div>
      <p class="note">Bars show how each feature pushes this account away from the portfolio-average risk. Axis ticks are PDs; steps are in log-odds so they add up exactly.</p></div>
    <div><h3 style="font-size:14px;margin-bottom:6px">Account</h3><div class="acct">${FEATURES.map((f, j) => `<div><span>${esc(f.short)}</span><b>${esc(fmtFeature(f.key, X[o + j]))}</b></div>`).join('')}</div></div></div>
    <div class="grid g3" style="margin-top:14px">
      <div><h3 style="font-size:14px;margin-bottom:6px">Reasons that raise risk</h3>${rs.up.length ? `<ul class="list up">${rs.up.map((r) => `<li>${esc(r.text)} <small class="note">(+${r.phi.toFixed(2)})</small></li>`).join('')}</ul>` : '<p class="note">No feature materially raises this account\'s risk.</p>'}</div>
      <div><h3 style="font-size:14px;margin-bottom:6px">Reasons that lower risk</h3>${rs.down.length ? `<ul class="list down">${rs.down.map((r) => `<li>${esc(r.text)} <small class="note">(${r.phi.toFixed(2)})</small></li>`).join('')}</ul>` : '<p class="note">None material.</p>'}</div>
      <div><h3 style="font-size:14px;margin-bottom:6px">What would change the decision</h3>${band[1] === 'Auto-approve' ? '<p class="note">Already auto-approved.</p>' : rc.options.length ? `<ul class="list">${rc.options.map((r) => `<li>${esc(r.verb)}: ${typeof r.from === 'number' ? esc(fmtFeature(r.key, r.from)) : esc(r.from)} → ${typeof r.to === 'number' ? esc(fmtFeature(r.key, r.to)) : esc(r.to)} brings PD to ${pct(r.pd, 1)} (${target})</li>`).join('')}</ul><p class="note">Single-change what-ifs holding everything else fixed; illustrative, not advice.</p>` : '<p class="note">No single realistic change reaches the next band.</p>'}</div></div>`;
  // waterfall
  const order = [...phi.keys()].sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
  const top = order.slice(0, 8); const rest = order.slice(8);
  const items = top.map((j) => ({ label: `${FEATURES[j].short} = ${fmtFeature(FEATURES[j].key, X[o + j])}`, v: phi[j] }));
  const restSum = rest.reduce((a, j) => a + phi[j], 0);
  if (rest.length) items.push({ label: `${rest.length} other features`, v: restSum });
  const labels = ['Average account', ...items.map((i) => i.label), 'This account'];
  let cum = base; const data = [[base - 0.03, base + 0.03]]; const colorsArr = [c.muted];
  for (const it of items) { data.push([Math.min(cum, cum + it.v), Math.max(cum, cum + it.v)]); colorsArr.push(it.v >= 0 ? c.bad : c.champion); cum += it.v; }
  data.push([margin - 0.03, margin + 0.03]); colorsArr.push(c.ink);
  const vals = [null, ...items.map((i) => i.v), null];
  chart($('#wf'), {
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: colorsArr, borderRadius: 3, barPercentage: 0.75 }] },
    options: {
      indexAxis: 'y', animation: false,
      scales: {
        x: { ...axes({ xTitle: 'Predicted PD (log-odds scale)' }).x, grid: { color: c.grid }, ticks: { color: c.ink2, callback: (v) => `${(sigmoid(v) * 100).toFixed(1)}%`, maxTicksLimit: 7 } },
        y: { grid: { display: false }, ticks: { color: c.ink2, font: { size: 11.5 } }, border: { display: false } },
      },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => { const i = x.dataIndex; if (vals[i] === null) return i === 0 ? `Average account: PD ${(avg * 100).toFixed(1)}%` : `This account: PD ${(pd * 100).toFixed(1)}%`; return `${vals[i] >= 0 ? '+' : ''}${vals[i].toFixed(3)} log-odds (odds × ${Math.exp(vals[i]).toFixed(2)})`; } } } },
    },
  });
  void logit; void predictProba;
}

export const state = () => ({ view, feat, acct, perm: permCache[view] ? permCache[view].r : null });
