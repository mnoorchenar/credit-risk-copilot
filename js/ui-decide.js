import { F, FIDX } from './data.js';
import { bandTable, strategyCurve, swapSet, policyApproves, POLICY } from './explain.js';
import { $, esc, chart, axes, colors, kpi, pct, money, destroyCharts, badge } from './common.js';

let view = 'challenger';

export function render(S) {
  const el = $('#tab-decide');
  destroyCharts(el);
  const have = ['champion', 'challenger'].filter((k) => S.models[k]);
  if (!have.length) { el.innerHTML = '<div class="card"><h2>Train a model first</h2><p class="lead">Go to the Train tab.</p></div>'; return; }
  if (!S.models[view]) view = have[0];
  const { c1, c2 } = S.cutoffs;
  el.innerHTML = `
    <div class="hero"><h2>Turn the score into a decision framework</h2>
    <p>A PD is only useful once it drives action. Set the cutoffs for <b>auto-approve</b>, <b>manual review</b> and <b>decline</b>, then see volumes, performance and expected loss. The comparison with today's rules shows who the model would swap in and out.</p></div>
    <div class="card"><div class="card-head"><div><h2>Decision cutoffs</h2></div><div class="seg" id="dc-seg">${have.map((k) => `<button data-v="${k}" class="${k === view ? 'on' : ''}">${k === 'champion' ? 'Champion · logistic' : 'Challenger · boosting'}</button>`).join('')}</div></div>
      <div class="grid g3"><label class="field"><span>Auto-approve below PD<b id="c1v">${pct(c1)}</b></span><input type="range" id="c1" min="0.005" max="0.15" step="0.0025" value="${c1}"></label>
      <label class="field"><span>Decline at or above PD<b id="c2v">${pct(c2)}</b></span><input type="range" id="c2" min="0.02" max="0.4" step="0.005" value="${c2}"></label>
      <label class="field"><span>Loss given default (LGD)<b id="lgdv">${pct(S.lgd, 0)}</b></span><input type="range" id="lgd" min="0.1" max="0.6" step="0.01" value="${S.lgd}"></label></div></div>
    <div class="card"><div class="card-head"><div><h2>Outcome by decision band</h2><p class="lead">Observed = what actually happened on the test set. Expected = what the model predicted. Close agreement means the PDs can be trusted for provisioning.</p></div></div><div id="bands"></div></div>
    <div class="grid g2"><div class="card"><div class="card-head"><div><h2>Expected vs realised loss</h2><p class="lead">PD × LGD × exposure, by band.</p></div></div><div class="chart"><canvas id="loss"></canvas></div></div>
      <div class="card"><div class="card-head"><div><h2>Approval vs bad rate</h2><p class="lead">Approve the best accounts first. Lower curve = better selection.</p></div></div><div class="chart"><canvas id="strat"></canvas></div></div></div>
    <div class="card"><div class="card-head"><div><h2>Model vs current rules: swap-set analysis</h2><p class="lead">Rules approve an account if score ≥ ${POLICY.minScore}, LTV ≤ ${POLICY.maxLtv}% and debt service ≤ ${POLICY.maxTds}%. The model is held to the <b>same approval rate</b>, so the comparison is fair.</p></div></div><div id="swap"></div></div>`;
  $('#dc-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; view = b.dataset.v; render(S); });
  const upd = () => {
    S.cutoffs.c1 = Number($('#c1').value); S.cutoffs.c2 = Math.max(Number($('#c2').value), S.cutoffs.c1 + 0.005); S.lgd = Number($('#lgd').value);
    $('#c1v').textContent = pct(S.cutoffs.c1); $('#c2v').textContent = pct(S.cutoffs.c2); $('#lgdv').textContent = pct(S.lgd, 0);
    paint(S);
    document.dispatchEvent(new CustomEvent('cutoffs-changed'));
  };
  ['c1', 'c2', 'lgd'].forEach((id) => $(`#${id}`).addEventListener('input', upd));
  paint(S);
}

function paint(S) {
  const e = S.models[view]; const c = colors(); const { c1, c2 } = S.cutoffs;
  const pd = e.preds.test; const y = S.te.y;
  const rows = bandTable(pd, y, S.ead, S.lgd, c1, c2);
  const tot = rows.reduce((a, r) => ({ ead: a.ead + r.ead, exp: a.exp + r.expLoss, real: a.real + r.realLoss }), { ead: 0, exp: 0, real: 0 });
  const st = { 'Auto-approve': 'pass', 'Manual review': 'warn', Decline: 'fail' };
  $('#bands').innerHTML = `<div class="scroll" style="max-height:none"><table class="t"><thead><tr><th class="l">Band</th><th>Accounts</th><th>Share</th><th>Avg predicted PD</th><th>Observed bad rate</th><th>Exposure</th><th>Expected loss</th><th>Realised loss</th></tr></thead><tbody>
    ${rows.map((r) => `<tr><td class="l">${badge(st[r.name], r.name)}</td><td>${r.n.toLocaleString('en-CA')}</td><td>${pct(r.share)}</td><td>${pct(r.avgPd, 2)}</td><td>${pct(r.badRate, 2)}</td><td>${money(r.ead)}</td><td>${money(r.expLoss)}</td><td>${money(r.realLoss)}</td></tr>`).join('')}
    <tr style="font-weight:800"><td class="l">Total</td><td>${pd.length.toLocaleString('en-CA')}</td><td>100%</td><td>${pct(pd.reduce((a, v) => a + v, 0) / pd.length, 2)}</td><td>${pct(y.reduce((a, v) => a + v, 0) / y.length, 2)}</td><td>${money(tot.ead)}</td><td>${money(tot.exp)}</td><td>${money(tot.real)}</td></tr></tbody></table></div>
    <div class="callout info" style="margin-top:12px">Auto-approve holds <b>${pct(rows[0].share, 0)}</b> of accounts at a bad rate of <b>${pct(rows[0].badRate, 2)}</b>; the decline band captures <b>${pct(rows[2].bad / Math.max(1, y.reduce((a, v) => a + v, 0)), 0)}</b> of all defaults from <b>${pct(rows[2].share, 0)}</b> of accounts. Total expected loss ${money(tot.exp)} against ${money(tot.real)} realised.</div>`;
  chart($('#loss'), {
    type: 'bar',
    data: { labels: rows.map((r) => r.name), datasets: [
      { label: 'Expected loss', data: rows.map((r) => r.expLoss / 1e6), backgroundColor: view === 'champion' ? c.champion : c.challenger, borderRadius: 3 },
      { label: 'Realised loss', data: rows.map((r) => r.realLoss / 1e6), backgroundColor: c.bad, borderRadius: 3 },
    ] },
    options: { scales: axes({ yTitle: 'CAD millions' }), plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: $${x.parsed.y.toFixed(2)}M` } } } },
  });
  const scoreRisk = new Float64Array(S.te.n); for (let i = 0; i < S.te.n; i++) scoreRisk[i] = -S.te.X[i * F + FIDX.credit_score];
  const mCurve = strategyCurve(y, pd); const sCurve = strategyCurve(y, scoreRisk);
  const pol = new Uint8Array(S.te.n); let pa = 0; let pb = 0; for (let i = 0; i < S.te.n; i++) { pol[i] = policyApproves(S.te.X, i * F) ? 1 : 0; if (pol[i]) { pa++; pb += y[i]; } }
  const pts = (a) => a.map((p) => ({ x: p.x * 100, y: p.y * 100 }));
  let ap = 0; let ab = 0; for (let i = 0; i < pd.length; i++) if (pd[i] < c2) { ap++; ab += y[i]; }
  chart($('#strat'), {
    type: 'line',
    data: { datasets: [
      { label: view === 'champion' ? 'Scorecard ranking' : 'Boosting ranking', data: pts(mCurve), borderColor: view === 'champion' ? c.champion : c.challenger, backgroundColor: c.challenger, borderWidth: 2.4, pointRadius: 0 },
      { label: 'Credit score only', data: pts(sCurve), borderColor: c.muted, borderDash: [5, 4], borderWidth: 2, pointRadius: 0 },
      { type: 'scatter', label: 'Current rules', data: [{ x: (pa / S.te.n) * 100, y: (pb / Math.max(1, pa)) * 100 }], backgroundColor: c.bad, pointRadius: 7, pointStyle: 'rectRot' },
      { type: 'scatter', label: 'Model at current cutoffs', data: [{ x: (ap / S.te.n) * 100, y: (ab / Math.max(1, ap)) * 100 }], backgroundColor: c.ink, pointRadius: 6 },
    ] },
    options: { parsing: false, scales: axes({ xTitle: 'Approval rate %', yTitle: 'Bad rate among approved %', x: { type: 'linear', min: 0, max: 100 }, y: { min: 0 } }) },
  });
  const sw = swapSet(y, pol, pd);
  const cell = (k, label, cls) => `<div class="${cls}"><b>${sw.cells[k][0].toLocaleString('en-CA')}</b>${label}<br><small>bad rate ${pct(sw.rates[k], 2)}</small></div>`;
  $('#swap').innerHTML = `<div class="grid g2"><div><div class="matrix"><div class="h"></div><div class="h">Model approves</div><div class="h">Model declines</div>
    <div class="h">Rules approve</div>${cell('both', 'approved by both', 'good')}${cell('policyOnly', 'swap-out: rules approve, model declines', 'badc')}
    <div class="h">Rules decline</div>${cell('modelOnly', 'swap-in: rules decline, model approves', 'good')}${cell('neither', 'declined by both', 'badc')}</div></div>
    <div><div class="kpis">${kpi('Approval rate (both)', pct(sw.approvals / S.te.n, 1))}${kpi('Rules bad rate', pct(sw.policyBadRate, 2))}${kpi('Model bad rate', pct(sw.modelBadRate, 2), 'same approval rate', sw.modelBadRate < sw.policyBadRate ? 'good' : 'bad')}${kpi('Defaults avoided', sw.badsAvoided, 'vs. current rules', sw.badsAvoided > 0 ? 'good' : 'bad')}</div>
    <div class="callout ${sw.badsAvoided > 0 ? 'ok' : ''}" style="margin-top:12px">${sw.badsAvoided > 0 ? `Swapping in ${sw.cells.modelOnly[0]} accounts the rules decline (bad rate ${pct(sw.rates.modelOnly, 2)}) and swapping out ${sw.cells.policyOnly[0]} the rules approve (bad rate ${pct(sw.rates.policyOnly, 2)}) lowers the approved bad rate from ${pct(sw.policyBadRate, 2)} to ${pct(sw.modelBadRate, 2)} with <b>no change in volume</b>.` : 'At this approval rate the model does not improve on the current rules.'}</div></div></div>`;
  void esc;
}

export const state = () => ({ view });
