import { FEATURES } from './data.js';
import { stressTest, ICON } from './explain.js';
import { computeGovernance, modelCard } from './governance.js';
import { $, esc, chart, axes, colors, kpi, badge, pct, money, num, destroyCharts } from './common.js';

let view = 'challenger';
const shock = { dUnemp: 0, dHpi: 0, dRate: 0 };

export function render(S) {
  const el = $('#tab-govern');
  destroyCharts(el);
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!have.length) { el.innerHTML = '<div class="card"><h2>Train a model first</h2><p class="lead">Go to the Train tab.</p></div>'; return; }
  if (!S.models[view]) view = have[0];
  const e = S.models[view]; const g = computeGovernance(S, view);
  const verdict = g.summary.fail ? ['fail', 'Not ready for deployment'] : g.summary.warn ? ['warn', 'Approve with conditions'] : ['pass', 'Ready for deployment'];
  el.innerHTML = `
    <div class="hero"><h2>Would this model survive validation?</h2>
    <p>The same checks a model risk team runs before sign-off, executed automatically against the model you trained. Change the hyperparameters on the Train tab and watch the scorecard change.</p></div>
    <div class="card"><div class="card-head"><div><h2>Validation scorecard</h2></div><div class="seg" id="gv-seg">${have.map((k) => `<button data-v="${k}" class="${k === view ? 'on' : ''}">${k === 'champion' ? 'Champion · logistic' : 'Challenger · boosting'}</button>`).join('')}</div></div>
      <div class="kpis">${kpi('Passed', g.summary.pass, `of ${g.checks.length} checks`, 'good')}${kpi('Warnings', g.summary.warn)}${kpi('Failed', g.summary.fail, '', g.summary.fail ? 'bad' : '')}${kpi('Verdict', badge(verdict[0], verdict[1]))}</div>
      <div style="margin-top:10px">${g.checks.map((c) => `<div class="gov"><span class="st ${c.status}" style="justify-content:center">${ICON[c.status]}</span><div><b>${esc(c.title)}</b><small>${esc(c.detail)}</small></div><div>${esc(c.value)}<small>threshold: ${esc(c.threshold)}</small></div><span>${badge(c.status, c.status === 'pass' ? 'Pass' : c.status === 'warn' ? 'Warning' : 'Fail')}</span></div>`).join('')}</div>
      <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap"><button class="btn" id="dl">Download model card (.md)</button><span class="note" style="align-self:center">A one-page summary of data, method, performance and limitations.</span></div></div>
    <div class="grid g2">
      <div class="card"><div class="card-head"><div><h2>Population drift</h2><p class="lead">PSI of each feature between the training population and the recent vintage. Above 0.10 = noticeable, above 0.25 = significant.</p></div></div><div class="chart tall" style="height:380px"><canvas id="drift"></canvas></div></div>
      <div class="card"><div class="card-head"><div><h2>Out-of-time performance</h2><p class="lead">The model was trained on older accounts. How does it do on the newest originations, where rates are higher and prices softer?</p></div></div><div class="kpis">${kpi('Test AUC', num(e.metrics.test.auc))}${kpi('Recent-vintage AUC', num(e.metrics.recent.auc), `${(e.metrics.recent.auc - e.metrics.test.auc >= 0 ? '+' : '')}${(e.metrics.recent.auc - e.metrics.test.auc).toFixed(3)}`)}${kpi('Mean predicted PD', pct(e.metrics.test.meanPd, 2), 'test')}${kpi('Mean predicted PD', pct(e.metrics.recent.meanPd, 2), 'recent')}${kpi('Observed default rate', pct(e.metrics.recent.rate, 2), 'recent')}${kpi('Score PSI', num(e.scorePsi), 'train vs recent')}</div>
        <div class="callout info" style="margin-top:12px">Recent accounts are riskier on average (higher LTV, higher unemployment, softer prices). A well-behaved model <b>raises its predicted PD</b> accordingly (${pct(e.metrics.test.meanPd, 2)} → ${pct(e.metrics.recent.meanPd, 2)}) to track the observed ${pct(e.metrics.recent.rate, 2)}.</div></div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Segment consistency</h2><p class="lead">Approval rate at the current decline cutoff and ranking accuracy by business segment.</p></div></div>
      <div class="grid g3">${Object.entries(g.segs).map(([k, rows]) => `<div><h3 style="font-size:14px;margin-bottom:4px">${k}</h3><table class="t"><thead><tr><th>Segment</th><th>n</th><th>Bad rate</th><th>Approval</th><th>AUC</th></tr></thead><tbody>${rows.map((s) => `<tr><td class="l">${esc(s.name)}</td><td>${s.n}</td><td>${pct(s.badRate)}</td><td>${pct(s.approval, 0)} ${s.ratio < 0.8 ? badge('warn', `${s.ratio.toFixed(2)}`) : ''}</td><td>${Number.isNaN(s.auc) ? '–' : s.auc.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>`).join('')}</div>
      <p class="note" style="margin-top:8px">Flag shown when a segment's approval rate is below 80% of the best segment. Differences can be legitimate risk differences; the point is to review them, not to assume.</p></div>
    <div class="card"><div class="card-head"><div><h2>Stress and sensitivity</h2><p class="lead">Re-score the test portfolio under a macro shock. The model reads unemployment, house prices and renewal payment shock directly, so no separate overlay is needed.</p></div></div>
      <div class="grid g3"><label class="field"><span>Unemployment<b id="sv-u">+${shock.dUnemp} pts</b></span><input type="range" id="sh-u" min="0" max="5" step="0.5" value="${shock.dUnemp}"></label>
      <label class="field"><span>House prices<b id="sv-h">${shock.dHpi}%</b></span><input type="range" id="sh-h" min="-30" max="0" step="1" value="${shock.dHpi}"></label>
      <label class="field"><span>Interest rates at renewal<b id="sv-r">+${shock.dRate} bps</b></span><input type="range" id="sh-r" min="0" max="300" step="25" value="${shock.dRate}"></label></div>
      <div class="grid g2" style="margin-top:12px"><div id="stress-k"></div><div><div class="chart short"><canvas id="stress"></canvas></div></div></div></div>`;
  $('#gv-seg').addEventListener('click', (ev) => { const b = ev.target.closest('button'); if (!b) return; view = b.dataset.v; render(S); });
  $('#dl').addEventListener('click', () => { const blob = new Blob([modelCard(S, view)], { type: 'text/markdown' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `model-card-${view}.md`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); });
  for (const [id, k, f] of [['sh-u', 'dUnemp', (v) => `+${v} pts`], ['sh-h', 'dHpi', (v) => `${v}%`], ['sh-r', 'dRate', (v) => `+${v} bps`]]) {
    $(`#${id}`).addEventListener('input', (ev) => { shock[k] = Number(ev.target.value); $(`#sv-${id.slice(-1)}`).textContent = f(shock[k]); paintStress(S); });
  }
  paintDrift(S, g); paintStress(S);
}

function paintDrift(S, g) {
  const c = colors();
  const d = g.drift;
  chart($('#drift'), {
    type: 'bar',
    data: { labels: d.map((x) => x.label), datasets: [{ label: 'PSI', data: d.map((x) => x.psi), backgroundColor: d.map((x) => (x.psi > 0.25 ? c.bad : x.psi > 0.1 ? c.accent + 'bb' : c.good)), borderRadius: 3 }] },
    options: { indexAxis: 'y', scales: { ...axes({ xTitle: 'Population Stability Index' }), y: { grid: { display: false }, ticks: { color: c.ink2, font: { size: 11 } }, border: { display: false } } }, plugins: { legend: { display: false }, tooltip: { callbacks: { label: (x) => `PSI ${x.parsed.x.toFixed(3)}` } } } },
  });
}

function paintStress(S) {
  const e = S.models[view]; const c = colors();
  const base = stressTest(e.model, S.te.X, S.te.n, S.ead, S.lgd, { dUnemp: 0, dHpi: 0, dRate: 0 }, S.te.seg.region);
  const st = stressTest(e.model, S.te.X, S.te.n, S.ead, S.lgd, shock, S.te.seg.region);
  const up = st.meanPd / base.meanPd - 1;
  $('#stress-k').innerHTML = `<div class="kpis">${kpi('Mean PD', pct(st.meanPd, 2), `baseline ${pct(base.meanPd, 2)}`, up > 0.001 ? 'bad' : '')}${kpi('Change in PD', `${up >= 0 ? '+' : ''}${(up * 100).toFixed(0)}%`, 'relative')}${kpi('Expected loss', money(st.expLoss), `baseline ${money(base.expLoss)}`, up > 0.001 ? 'bad' : '')}${kpi('Extra loss', money(st.expLoss - base.expLoss), 'on the test book')}</div><p class="note" style="margin-top:8px">Rates act through payment shock (about +7% payment per 100 bps for accounts renewing within 12 months) and debt service. Prices act through LTV and the local price trend.</p>`;
  chart($('#stress'), {
    type: 'bar',
    data: { labels: st.byRegion.map((r) => r.name), datasets: [
      { label: 'Baseline', data: base.byRegion.map((r) => r.pd * 100), backgroundColor: c.good, borderRadius: 3 },
      { label: 'Stressed', data: st.byRegion.map((r) => r.pd * 100), backgroundColor: c.bad, borderRadius: 3 },
    ] },
    options: { scales: axes({ yTitle: 'Mean PD %' }), plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${x.parsed.y.toFixed(2)}%` } } } },
  });
  void FEATURES;
}

export const state = () => ({ view, shock: { ...shock } });
