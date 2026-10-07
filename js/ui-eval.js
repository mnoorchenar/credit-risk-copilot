import { $, $$, esc, chart, axes, colors, kpi, pct, num, destroyCharts } from './common.js';

let view = 'challenger';
let cutoff = 0.06;

export function render(S) {
  const el = $('#tab-eval');
  destroyCharts(el);
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!have.length) { el.innerHTML = '<div class="card"><h2>Train a model first</h2><p class="lead">Go to the Train tab.</p></div>'; return; }
  if (!S.models[view]) view = have[0];
  const e = S.models[view];
  const m = e.metrics.test;
  el.innerHTML = `
    <div class="hero"><h2>Is the model any good?</h2>
    <p>All numbers are on the <b>test set</b> (${S.te.n.toLocaleString('en-CA')} accounts the model never saw during training or tuning). Ranking power (AUC, KS) tells you if riskier accounts get higher scores; calibration tells you whether a "5%" prediction really means 5 in 100 default.</p></div>
    <div class="card"><div class="card-head"><div><h2>Headline performance</h2></div>
      <div class="seg" id="ev-seg">${have.map((k) => `<button data-v="${k}" class="${k === view ? 'on' : ''}">${k === 'champion' ? 'Champion · logistic' : 'Challenger · boosting'}</button>`).join('')}</div></div>
      <div class="kpis">${kpi('AUC', num(m.auc), 'ranking power')}${kpi('Gini', num(m.gini), '2 × AUC − 1')}${kpi('KS', num(m.ks), 'max separation')}${kpi('Brier score', num(m.brier, 4), 'lower is better')}${kpi('Calibration slope', num(e.slope, 2), 'ideal 1.00', Math.abs(e.slope - 1) < 0.2 ? 'good' : 'bad')}${kpi('Calibration error', pct(e.calib.ece, 2), 'expected, 10 bins')}</div></div>
    <div class="grid g2">
      <div class="card"><div class="card-head"><div><h2>ROC curve</h2><p class="lead">Further toward the top-left corner is better.</p></div></div><div class="chart"><canvas id="roc"></canvas></div></div>
      <div class="card"><div class="card-head"><div><h2>Calibration</h2><p class="lead">Predicted default rate vs what actually happened, by decile of predicted risk.</p></div></div><div class="chart"><canvas id="cal"></canvas></div></div>
    </div>
    <div class="grid g2">
      <div class="card"><div class="card-head"><div><h2>Lift by risk decile</h2><p class="lead">Decile 1 holds the 10% of accounts the model considers riskiest.</p></div></div><div class="chart"><canvas id="lift"></canvas></div>
        <div class="scroll" style="margin-top:10px;max-height:240px"><table class="t"><thead><tr><th>Decile</th><th>Accounts</th><th>Defaults</th><th>Bad rate</th><th>Lift</th><th>Cum. captured</th></tr></thead><tbody>${e.deciles.map((d) => `<tr><td>${d.decile}</td><td>${d.n}</td><td>${d.bad}</td><td>${pct(d.rate)}</td><td>${d.lift.toFixed(1)}x</td><td>${pct(d.cumCapture, 0)}</td></tr>`).join('')}</tbody></table></div></div>
      <div class="card"><div class="card-head"><div><h2>Score distribution</h2><p class="lead">Share of each outcome group by predicted PD. Good separation = defaults sit to the right.</p></div></div><div class="chart"><canvas id="dist"></canvas></div></div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Operating point</h2><p class="lead">Decline everyone whose predicted PD is at or above the cutoff. Move the slider to see the trade-off between catching defaults and turning away good customers.</p></div></div>
      <div class="grid g2"><div><label class="field"><span>PD cutoff<b id="cut-v"></b></span><input type="range" id="cut" min="0.01" max="0.4" step="0.005" value="${cutoff}"></label><div id="op-kpis" class="kpis" style="margin-top:12px"></div></div><div id="matrix"></div></div></div>`;
  $('#ev-seg').addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (!b) return; view = b.dataset.v; render(S); });
  $('#cut').addEventListener('input', (ev) => { cutoff = Number(ev.target.value); paintOp(S); });
  draw(S);
  paintOp(S);
}

function draw(S) {
  const c = colors(); const e = S.models[view];
  const both = ['champion', 'challenger'].filter((k) => S.models[k]);
  const col = (k) => (k === 'champion' ? c.champion : c.challenger);
  const name = (k) => (k === 'champion' ? 'Champion · logistic' : 'Challenger · boosting');
  chart($('#roc'), {
    type: 'line',
    data: { datasets: [
      ...both.map((k) => ({ label: `${name(k)} (AUC ${S.models[k].metrics.test.auc.toFixed(3)})`, data: S.models[k].roc, borderColor: col(k), backgroundColor: col(k), borderWidth: 2.2, pointRadius: 0, tension: 0 })),
      { label: 'Random', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: c.muted, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
    ] },
    options: { parsing: false, scales: axes({ xTitle: 'False positive rate (good accounts declined)', yTitle: 'True positive rate (defaults caught)', x: { type: 'linear', min: 0, max: 1 }, y: { min: 0, max: 1 } }), interaction: { mode: 'nearest', intersect: false } },
  });
  const mx = Math.max(...both.flatMap((k) => S.models[k].calib.bins.map((b) => Math.max(b.pred, b.obs)))) * 1.1;
  chart($('#cal'), {
    type: 'line',
    data: { datasets: [
      ...both.map((k) => ({ label: name(k), data: S.models[k].calib.bins.map((b) => ({ x: b.pred * 100, y: b.obs * 100 })), borderColor: col(k), backgroundColor: col(k), borderWidth: 2, pointRadius: 4, tension: 0 })),
      { label: 'Perfect calibration', data: [{ x: 0, y: 0 }, { x: mx * 100, y: mx * 100 }], borderColor: c.muted, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
    ] },
    options: { parsing: false, scales: axes({ xTitle: 'Predicted default rate %', yTitle: 'Observed default rate %', x: { type: 'linear', min: 0 }, y: { min: 0 } }), plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: predicted ${x.parsed.x.toFixed(1)}%, observed ${x.parsed.y.toFixed(1)}%` } } } },
  });
  chart($('#lift'), {
    type: 'bar',
    data: { labels: e.deciles.map((d) => `D${d.decile}`), datasets: [
      { type: 'bar', label: 'Observed bad rate', data: e.deciles.map((d) => d.rate * 100), backgroundColor: col(view), borderRadius: 3 },
      { type: 'line', label: 'Average predicted PD', data: e.deciles.map((d) => d.avgPd * 100), borderColor: c.accent, backgroundColor: c.accent, borderWidth: 2, pointRadius: 3 },
    ] },
    options: { scales: axes({ yTitle: 'Default rate %', xTitle: 'Risk decile (D1 = riskiest)' }) },
  });
  // distribution of predicted PD by outcome
  const p = e.preds.test; const y = S.te.y; const edges = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 1.01];
  const labels = edges.map((hi, i) => (i === 0 ? '<1%' : i === edges.length - 1 ? `${edges[i - 1] * 100}%+` : `${edges[i - 1] * 100}–${hi * 100}%`));
  const g = new Array(edges.length).fill(0); const b = new Array(edges.length).fill(0);
  for (let i = 0; i < p.length; i++) { let k = 0; while (p[i] >= edges[k]) k++; (y[i] ? b : g)[k]++; }
  const sg = g.reduce((a, v) => a + v, 0); const sb = b.reduce((a, v) => a + v, 0);
  chart($('#dist'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: 'No default', data: g.map((v) => (v / sg) * 100), backgroundColor: c.good, borderRadius: 3 },
      { label: 'Default', data: b.map((v) => (v / sb) * 100), backgroundColor: c.bad, borderRadius: 3 },
    ] },
    options: { scales: axes({ xTitle: 'Predicted PD', yTitle: '% of outcome group' }) },
  });
}

function paintOp(S) {
  const e = S.models[view]; const p = e.preds.test; const y = S.te.y;
  $('#cut-v').textContent = pct(cutoff);
  let tp = 0; let fp = 0; let fn = 0; let tn = 0;
  for (let i = 0; i < p.length; i++) { const dec = p[i] >= cutoff; if (dec && y[i]) tp++; else if (dec) fp++; else if (y[i]) fn++; else tn++; }
  const n = p.length; const bads = tp + fn;
  $('#op-kpis').innerHTML = `${kpi('Approval rate', pct((tn + fn) / n, 1))}${kpi('Bad rate among approved', pct(fn / Math.max(1, tn + fn), 2), `portfolio ${pct(bads / n, 2)}`)}${kpi('Defaults caught', pct(tp / Math.max(1, bads), 0), 'recall')}${kpi('Good accounts declined', pct(fp / Math.max(1, fp + tn), 1), 'false positive rate')}`;
  $('#matrix').innerHTML = `<div class="matrix"><div class="h"></div><div class="h">Did not default</div><div class="h">Defaulted</div>
    <div class="h">Approved</div><div class="good"><b>${tn.toLocaleString('en-CA')}</b>correctly approved</div><div class="badc"><b>${fn.toLocaleString('en-CA')}</b>missed defaults</div>
    <div class="h">Declined</div><div class="badc"><b>${fp.toLocaleString('en-CA')}</b>good customers declined</div><div class="good"><b>${tp.toLocaleString('en-CA')}</b>defaults avoided</div></div>`;
}

export const state = () => ({ view, cutoff });
