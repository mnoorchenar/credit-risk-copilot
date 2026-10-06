import { FEATURES, F, FIDX, featureProfile, fmtFeature, column } from './data.js';
import { $, $$, esc, chart, axes, colors, kpi, badge, pct, destroyCharts } from './common.js';

let current = 'credit_score';

export function render(S) {
  const el = $('#tab-data');
  destroyCharts(el);
  const rate = S.clean.y.reduce((a, b) => a + b, 0) / S.clean.n;
  const issues = S.checks.filter((c) => c.count > 0);
  const sign = (s) => (s > 0 ? 'higher = riskier' : s < 0 ? 'higher = safer' : 'no fixed direction');
  el.innerHTML = `
    <div class="hero"><h2>Know the data before you model it</h2>
    <p>${S.world.raw.n.toLocaleString('en-CA')} synthetic mortgage and HELOC accounts. The target is <b>90+ days past due within 12 months</b>. The raw extract is deliberately messy, like a real one: automated controls find the problems, fix them reproducibly and log every change.</p></div>
    <div class="kpis">
      ${kpi('Accounts in raw extract', S.world.raw.n.toLocaleString('en-CA'))}
      ${kpi('After cleaning', S.clean.n.toLocaleString('en-CA'), `${S.clean.fixes.droppedDuplicates} duplicates removed`)}
      ${kpi('Default rate', pct(rate, 2), 'target event')}
      ${kpi('Features', F, 'borrower, loan, local economy')}
      ${kpi('Train / val / test', `${S.tr.n.toLocaleString('en-CA')} / ${S.va.n.toLocaleString('en-CA')} / ${S.te.n.toLocaleString('en-CA')}`, '60 / 20 / 20, fixed seed')}
      ${kpi('Recent vintage (out-of-time)', S.recent.n.toLocaleString('en-CA'), 'held out for drift checks')}
    </div>
    <div class="card"><div class="card-head"><div><h2>Automated data integrity controls</h2><p class="lead">${issues.length} of ${S.checks.length} controls raised findings. Findings are fixed before any model sees the data.</p></div></div>
      <div class="scroll"><table class="t"><thead><tr><th>Control</th><th class="l">Check</th><th>Records</th><th>Rate</th><th>Status</th><th class="l">Remediation</th></tr></thead><tbody>
      ${S.checks.map((c) => `<tr><td class="l">${esc(c.category)}</td><td class="l">${esc(c.name)}</td><td>${c.count.toLocaleString('en-CA')}</td><td>${pct(c.rate, 2)}</td><td>${c.count === 0 ? badge('pass', 'Pass') : /Flag|Review/.test(c.action) ? badge('warn', 'Flagged') : badge('pass', 'Fixed')}</td><td class="l">${esc(c.action)}</td></tr>`).join('')}
      </tbody></table></div>
      <div class="callout info" style="margin-top:12px"><b>Applied fixes:</b> ${S.clean.fixes.invalid.toLocaleString('en-CA')} invalid values set to missing, ${S.clean.fixes.imputed.toLocaleString('en-CA')} cells imputed with training-set medians (no leakage from validation or test), ${S.clean.fixes.droppedDuplicates.toLocaleString('en-CA')} duplicate accounts dropped.</div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Explore a feature</h2><p class="lead">Left: share of accounts by value for each outcome. Right: default rate by value (bins with at least 30 accounts).</p></div></div>
      <div class="chips" id="feat-chips">${FEATURES.map((f) => `<button class="chip ${f.key === current ? 'on' : ''}" data-k="${f.key}">${esc(f.short)}</button>`).join('')}</div>
      <div class="grid g2" style="margin-top:12px"><div><div class="chart" id="eda-a"><canvas></canvas></div></div><div><div class="chart" id="eda-b"><canvas></canvas></div></div></div>
    </div>
    <div class="card"><div class="card-head"><div><h2>Feature dictionary</h2></div></div>
      <div class="scroll"><table class="t"><thead><tr><th class="l">Feature</th><th class="l">Description</th><th class="l">Expected effect</th><th>Mean</th><th>Default rate: low / high</th></tr></thead><tbody>
      ${FEATURES.map((f, j) => { const c = column(S.tr, j); const m = c.reduce((a, b) => a + b, 0) / c.length; const med = [...c].sort((a, b) => a - b)[c.length >> 1]; let lo = [0, 0]; let hi = [0, 0]; for (let i = 0; i < S.tr.n; i++) { const t = c[i] <= med ? lo : hi; t[0]++; t[1] += S.tr.y[i]; } return `<tr><td class="l"><b>${esc(f.label)}</b></td><td class="l">${esc(f.desc)}</td><td class="l">${sign(f.sign)}</td><td>${esc(fmtFeature(f.key, m))}</td><td>${pct(lo[1] / Math.max(1, lo[0]))} / ${pct(hi[1] / Math.max(1, hi[0]))}</td></tr>`; }).join('')}
      </tbody></table></div></div>`;
  $('#feat-chips').addEventListener('click', (e) => { const b = e.target.closest('.chip'); if (!b) return; current = b.dataset.k; $$('.chip', el).forEach((c) => c.classList.toggle('on', c === b)); drawEda(S); });
  drawEda(S);
}

function drawEda(S) {
  const j = FIDX[current]; const f = FEATURES[j]; const c = colors();
  const p = featureProfile(S.tr, j, f.binary ? 2 : 20);
  const totG = p.good.reduce((a, b) => a + b, 0); const totB = p.bad.reduce((a, b) => a + b, 0);
  const labels = p.centers.map((v) => (f.binary ? (v ? 'Yes' : 'No') : fmtFeature(current, v)));
  chart($('#eda-a canvas'), {
    type: 'bar',
    data: { labels, datasets: [
      { label: 'No default', data: p.good.map((v) => (v / totG) * 100), backgroundColor: c.good, borderRadius: 3 },
      { label: 'Default', data: p.bad.map((v) => (v / totB) * 100), backgroundColor: c.bad, borderRadius: 3 },
    ] },
    options: { scales: axes({ yTitle: '% of outcome group', xTitle: f.label }), plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${x.parsed.y.toFixed(1)}%` } } } },
  });
  const base = (totB / (totG + totB)) * 100;
  chart($('#eda-b canvas'), {
    type: 'line',
    data: { labels, datasets: [
      { label: 'Default rate', data: p.rate, borderColor: c.bad, backgroundColor: c.bad, borderWidth: 2, pointRadius: 3, spanGaps: true, tension: 0.25 },
      { label: 'Portfolio average', data: labels.map(() => base), borderColor: c.muted, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
    ] },
    options: { scales: axes({ yTitle: 'Default rate %', xTitle: f.label, y: { beginAtZero: true } }), plugins: { tooltip: { callbacks: { label: (x) => `${x.dataset.label}: ${x.parsed.y?.toFixed(2)}%` } } } },
  });
}
