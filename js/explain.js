// Explainability, decision strategy and governance helpers built on top of ml.js. Pure functions, no DOM.
import { F, FEATURES, FIDX, REGIONS, EMPLOYMENT, PROPERTY, fmtFeature, column } from './data.js';
import { auc, marginOf, predictProba, partialDependence, psi, sigmoid } from './ml.js';

// ---- reason codes (adverse-action style) ---------------------------------------------------------------------------------
const WORDS = {
  credit_score: (v) => `Credit score of ${fmtFeature('credit_score', v)} is weak relative to the portfolio`,
  ltv: (v) => `Loan-to-value of ${fmtFeature('ltv', v)} leaves a thin equity cushion`,
  tds: (v) => `Debt service ratio of ${fmtFeature('tds', v)} is stretched`,
  income: (v) => `Income of ${fmtFeature('income', v)} is low for this exposure`,
  loan_amount: (v) => `Loan amount of ${fmtFeature('loan_amount', v)} is large`,
  mortgage_rate: (v) => `Contract rate of ${fmtFeature('mortgage_rate', v)} is a risk factor in this vintage`,
  months_on_book: (v) => `Account age of ${fmtFeature('months_on_book', v)} months (early-life risk)`,
  renewal_shock: (v) => `Payment increase of ${fmtFeature('renewal_shock', v)} at renewal`,
  heloc_util: (v) => `HELOC utilization of ${fmtFeature('heloc_util', v)} is high`,
  prior_delinq: (v) => `${fmtFeature('prior_delinq', v)} delinquency event(s) in the last 12 months`,
  unemp: (v) => `Local unemployment of ${fmtFeature('unemp', v)} is elevated`,
  hpi_12m: (v) => `Local house prices moved ${fmtFeature('hpi_12m', v)} over 12 months`,
  insured: () => 'Mortgage is not default-insured',
  self_employed: () => 'Self-employed income is harder to verify',
  condo: () => 'Condominium property type',
};
const PROTECT = {
  credit_score: (v) => `Credit score of ${fmtFeature('credit_score', v)} is strong`,
  ltv: (v) => `Low loan-to-value of ${fmtFeature('ltv', v)}`,
  tds: (v) => `Comfortable debt service ratio of ${fmtFeature('tds', v)}`,
  income: (v) => `Income of ${fmtFeature('income', v)} supports the exposure`,
  prior_delinq: () => 'No recent delinquencies',
  insured: () => 'Mortgage is default-insured',
  heloc_util: () => 'Low or no HELOC utilization',
  renewal_shock: () => 'No renewal payment shock in the next 12 months',
  unemp: (v) => `Local unemployment of ${fmtFeature('unemp', v)} is low`,
  hpi_12m: (v) => `Local house prices are firm (${fmtFeature('hpi_12m', v)})`,
};

export function reasons(phi, X, off, topN = 4) {
  const items = Array.from({ length: F }, (_, j) => ({ j, key: FEATURES[j].key, phi: phi[j], value: X[off + j] }));
  const up = items.filter((i) => i.phi > 0.01).sort((a, b) => b.phi - a.phi).slice(0, topN).map((i) => ({ ...i, text: (WORDS[i.key] || (() => FEATURES[i.j].label))(i.value) }));
  const down = items.filter((i) => i.phi < -0.01).sort((a, b) => a.phi - b.phi).slice(0, 3).map((i) => ({ ...i, text: (PROTECT[i.key] || (() => `${FEATURES[i.j].label} lowers risk`))(i.value) }));
  return { up, down };
}

// ---- counterfactual recourse -------------------------------------------------------------------------------------------------
const CHANGES = [
  { key: 'tds', dir: -1, max: 22, verb: 'Reduce debt service ratio' },
  { key: 'ltv', dir: -1, max: 35, verb: 'Reduce loan-to-value' },
  { key: 'credit_score', dir: 1, max: 140, verb: 'Raise credit score' },
  { key: 'heloc_util', dir: -1, max: 100, verb: 'Pay down the HELOC' },
  { key: 'income', dir: 1, max: 80, verb: 'Increase verified income' },
];

function pdOf(model, row) { return sigmoid(marginOf(model, row, 1, F)[0]); }

/** Smallest single-feature change (and one two-feature combination) that brings PD under the cutoff. Illustrative only. */
export function recourse(model, X, off, cutoff) {
  const base = Float64Array.from(X.slice(off, off + F));
  const pd0 = pdOf(model, base);
  if (pd0 < cutoff) return { pd: pd0, options: [] };
  const options = [];
  for (const c of CHANGES) {
    const j = FIDX[c.key]; const row = Float64Array.from(base);
    const f = FEATURES[j];
    for (let s = 1; s <= 40; s++) {
      const delta = (c.max * s) / 40;
      let v = base[j] + c.dir * delta;
      v = Math.min(f.max, Math.max(f.min, v));
      row[j] = v;
      const pd = pdOf(model, row);
      if (pd < cutoff) { options.push({ key: c.key, verb: c.verb, from: base[j], to: v, pd, size: delta / c.max }); break; }
    }
  }
  const row = Float64Array.from(base); const jl = FIDX.ltv; const jt = FIDX.tds; let best = null;
  for (let a = 1; a <= 10; a++) for (let b = 1; b <= 10; b++) {
    row[jl] = Math.max(FEATURES[jl].min, base[jl] - (a * 3.5)); row[jt] = Math.max(FEATURES[jt].min, base[jt] - (b * 2.2));
    const pd = pdOf(model, row);
    if (pd < cutoff && (!best || a + b < best.cost)) best = { cost: a + b, ltv: row[jl], tds: row[jt], pd };
  }
  if (best) options.push({ key: 'combo', verb: 'Reduce LTV and debt service together', from: `${base[jl].toFixed(0)}% / ${base[jt].toFixed(0)}%`, to: `${best.ltv.toFixed(0)}% / ${best.tds.toFixed(0)}%`, pd: best.pd, size: best.cost / 20 });
  options.sort((a, b) => a.size - b.size);
  return { pd: pd0, options: options.slice(0, 3) };
}

// ---- decision strategy ----------------------------------------------------------------------------------------------------
export const POLICY = { minScore: 680, maxLtv: 80, maxTds: 44 };
export const policyApproves = (X, off) => X[off + FIDX.credit_score] >= POLICY.minScore && X[off + FIDX.ltv] <= POLICY.maxLtv && X[off + FIDX.tds] <= POLICY.maxTds;

export function bandTable(pd, y, ead, lgd, c1, c2) {
  const names = ['Auto-approve', 'Manual review', 'Decline'];
  const rows = names.map((name) => ({ name, n: 0, avgPd: 0, bad: 0, ead: 0, expLoss: 0, realLoss: 0 }));
  for (let i = 0; i < pd.length; i++) {
    const b = pd[i] < c1 ? 0 : pd[i] < c2 ? 1 : 2; const r = rows[b];
    r.n++; r.avgPd += pd[i]; r.bad += y[i]; r.ead += ead[i]; r.expLoss += pd[i] * lgd * ead[i]; r.realLoss += y[i] * lgd * ead[i];
  }
  for (const r of rows) { r.share = r.n / pd.length; r.avgPd = r.n ? r.avgPd / r.n : 0; r.badRate = r.n ? r.bad / r.n : 0; }
  return rows;
}

/** Approve the lowest-risk accounts first: approval rate (x) versus bad rate among approved (y). */
export function strategyCurve(y, risk, points = 40) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => risk[a] - risk[b]);
  const out = []; let bad = 0;
  const marks = new Set(Array.from({ length: points }, (_, k) => Math.max(1, Math.round(((k + 1) / points) * n))));
  for (let k = 0; k < n; k++) { bad += y[idx[k]]; if (marks.has(k + 1)) out.push({ x: (k + 1) / n, y: bad / (k + 1) }); }
  return out;
}

/** Compare the rule-based policy to the model at the same approval rate: who swaps in and out, and how they perform. */
export function swapSet(y, policyOk, pd) {
  const n = y.length; let approvals = 0; for (let i = 0; i < n; i++) if (policyOk[i]) approvals++;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => pd[a] - pd[b]);
  const modelOk = new Uint8Array(n); for (let k = 0; k < approvals; k++) modelOk[order[k]] = 1;
  const cells = { both: [0, 0], policyOnly: [0, 0], modelOnly: [0, 0], neither: [0, 0] };
  for (let i = 0; i < n; i++) {
    const c = policyOk[i] && modelOk[i] ? cells.both : policyOk[i] ? cells.policyOnly : modelOk[i] ? cells.modelOnly : cells.neither;
    c[0]++; c[1] += y[i];
  }
  const rate = (c) => (c[0] ? c[1] / c[0] : 0);
  const polBad = (cells.both[1] + cells.policyOnly[1]); const modBad = (cells.both[1] + cells.modelOnly[1]);
  return { approvals, cells, rates: { both: rate(cells.both), policyOnly: rate(cells.policyOnly), modelOnly: rate(cells.modelOnly), neither: rate(cells.neither) }, policyBadRate: polBad / approvals, modelBadRate: modBad / approvals, badsAvoided: polBad - modBad };
}

// ---- governance -------------------------------------------------------------------------------------------------------------
export function monotonicity(model, X, n, j, sign, sample = 400) {
  const col = column({ X, n }, j); const s = Float64Array.from(col).sort();
  const lo = s[Math.floor(n * 0.03)]; const hi = s[Math.floor(n * 0.97)];
  const grid = Array.from({ length: 16 }, (_, k) => lo + ((hi - lo) * k) / 15);
  const step = Math.max(1, Math.floor(n / sample)); const idx = []; for (let i = 0; i < n; i += step) idx.push(i);
  const Xs = new Float64Array(idx.length * F); idx.forEach((i, k) => Xs.set(X.slice(i * F, i * F + F), k * F));
  const pdp = partialDependence(model, Xs, idx.length, F, j, grid, 0);
  const range = Math.max(...pdp.mean) - Math.min(...pdp.mean) || 1e-9;
  let violations = 0;
  for (let k = 1; k < pdp.mean.length; k++) if (sign * (pdp.mean[k] - pdp.mean[k - 1]) < -0.04 * range) violations++;
  return { violations, ok: violations === 0, grid, mean: pdp.mean };
}

export function segmentTable(pd, y, seg, labels, cutoff) {
  const out = [];
  labels.forEach((name, k) => {
    const ids = []; for (let i = 0; i < y.length; i++) if (seg[i] === k) ids.push(i);
    if (ids.length < 80) return;
    const yy = ids.map((i) => y[i]); const pp = ids.map((i) => pd[i]);
    out.push({ name, n: ids.length, badRate: yy.reduce((a, b) => a + b, 0) / ids.length, avgPd: pp.reduce((a, b) => a + b, 0) / ids.length, approval: pp.filter((p) => p < cutoff).length / ids.length, auc: yy.some((v) => v) && yy.some((v) => !v) ? auc(yy, pp) : NaN });
  });
  const best = Math.max(...out.map((s) => s.approval), 1e-9);
  for (const s of out) s.ratio = s.approval / best;
  return out;
}

/** Re-score the test set under a macro shock. Rates act through payment shock and debt service; prices through LTV. */
export function stressTest(model, X, n, ead, lgd, { dUnemp = 0, dHpi = 0, dRate = 0 }, regionSeg) {
  const Xs = Float64Array.from(X);
  for (let i = 0; i < n; i++) {
    const o = i * F;
    Xs[o + FIDX.unemp] += dUnemp;
    Xs[o + FIDX.hpi_12m] += dHpi;
    Xs[o + FIDX.ltv] = Math.min(100, Xs[o + FIDX.ltv] / Math.max(0.5, 1 + dHpi / 100));
    if (Xs[o + FIDX.renewal_shock] > 0) { Xs[o + FIDX.renewal_shock] += 7 * (dRate / 100); Xs[o + FIDX.tds] += Xs[o + FIDX.tds] * 0.07 * (dRate / 100); }
  }
  const pd = predictProba(model, Xs, n, F);
  let el = 0; let mean = 0; const reg = REGIONS.map(() => ({ n: 0, pd: 0 }));
  for (let i = 0; i < n; i++) { mean += pd[i]; el += pd[i] * lgd * ead[i]; if (regionSeg) { reg[regionSeg[i]].n++; reg[regionSeg[i]].pd += pd[i]; } }
  return { meanPd: mean / n, expLoss: el, byRegion: reg.map((r, k) => ({ name: REGIONS[k], pd: r.n ? r.pd / r.n : 0 })) };
}

export function featureDrift(trainData, recentData) {
  return FEATURES.map((f, j) => ({ key: f.key, label: f.label, psi: f.binary ? Math.abs(mean(column(trainData, j)) - mean(column(recentData, j))) * 0.5 : psi(column(trainData, j), column(recentData, j)) }));
}
const mean = (a) => { let s = 0; for (const v of a) s += v; return s / a.length; };

export function status(v, okMax, warnMax) { return v <= okMax ? 'pass' : v <= warnMax ? 'warn' : 'fail'; }
export const ICON = { pass: '✔', warn: '▲', fail: '✖' };

export { EMPLOYMENT, PROPERTY, REGIONS };
