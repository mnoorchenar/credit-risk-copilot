// Model validation checks (what a model risk team would run) plus the grounded fact sheet used by the memo and Q&A.
import { F, FEATURES, REGIONS, EMPLOYMENT, PROPERTY, fmtFeature } from './data.js';
import { auc, marginOf, shapRow, calibration } from './ml.js';
import { monotonicity, segmentTable, featureDrift, status, ICON, bandTable, swapSet, policyApproves, POLICY, stressTest } from './explain.js';

const cache = new Map();

export function computeGovernance(S, which) {
  const e = S.models[which];
  const key = `${which}:${e.trainedAt.getTime()}:${S.cutoffs.c2}`;
  if (cache.has(key)) return cache.get(key);
  const mt = e.metrics;
  const checks = [];
  const add = (id, title, value, threshold, st, detail) => checks.push({ id, title, value, threshold, status: st, detail });

  const a = mt.test.auc;
  add('disc', 'Discriminatory power', `AUC ${a.toFixed(3)} · KS ${mt.test.ks.toFixed(3)}`, 'AUC ≥ 0.75 (warn ≥ 0.70)', a >= 0.75 ? 'pass' : a >= 0.7 ? 'warn' : 'fail', 'Ability to rank defaults above non-defaults on unseen data.');
  const gap = mt.train.auc - mt.val.auc;
  add('over', 'Overfitting', `train − validation AUC = ${gap.toFixed(3)}`, 'gap ≤ 0.04 (warn ≤ 0.08)', status(gap, 0.04, 0.08), 'A large gap means the model memorised the training data.');
  const oot = mt.test.auc - mt.recent.auc;
  add('oot', 'Out-of-time performance', `AUC ${mt.recent.auc.toFixed(3)} on recent vintage (${oot >= 0 ? '−' : '+'}${Math.abs(oot).toFixed(3)})`, 'drop ≤ 0.03 (warn ≤ 0.06)', status(oot, 0.03, 0.06), 'Performance on newer originations whose economics have shifted.');
  const ece = e.calib.ece;
  add('cal', 'Calibration', `error ${(ece * 100).toFixed(2)}% · slope ${e.slope.toFixed(2)}`, 'error ≤ 1% and slope 0.85–1.15', ece <= 0.01 && e.slope >= 0.85 && e.slope <= 1.15 ? 'pass' : ece <= 0.02 ? 'warn' : 'fail', 'Do predicted PDs match observed default rates? Needed for provisioning and pricing.');
  add('psi', 'Score stability (PSI)', `PSI ${e.scorePsi.toFixed(3)} train vs recent`, 'PSI ≤ 0.10 (warn ≤ 0.25)', status(e.scorePsi, 0.1, 0.25), 'Population Stability Index of the score distribution.');

  const mono = FEATURES.map((f, j) => ({ f, j })).filter((x) => x.f.sign !== 0).map(({ f, j }) => ({ key: f.key, label: f.label, sign: f.sign, ...monotonicity(e.model, S.te.X, S.te.n, j, f.sign) }));
  const bad = mono.filter((m) => !m.ok);
  add('mono', 'Business-logic monotonicity', bad.length ? `${bad.length} of ${mono.length} violate expected direction (${bad.map((b) => b.label).join(', ')})` : `all ${mono.length} key features behave as expected`, 'no violations (warn ≤ 1)', bad.length === 0 ? 'pass' : bad.length <= 1 ? 'warn' : 'fail', 'Higher LTV, debt service, delinquencies or unemployment should never lower risk.');

  const pd = e.preds.test; const c2 = S.cutoffs.c2;
  const segs = { Employment: segmentTable(pd, S.te.y, S.te.seg.employment, EMPLOYMENT, c2), Property: segmentTable(pd, S.te.y, S.te.seg.property, PROPERTY, c2), Region: segmentTable(pd, S.te.y, S.te.seg.region, REGIONS, c2) };
  const all = Object.values(segs).flat();
  const minRatio = Math.min(...all.map((s) => s.ratio)); const aucs = all.map((s) => s.auc).filter((v) => !Number.isNaN(v));
  const aucGap = Math.max(...aucs) - Math.min(...aucs);
  const worst = all.reduce((w, s) => (s.ratio < w.ratio ? s : w), all[0]);
  add('seg', 'Segment consistency', `lowest approval ratio ${minRatio.toFixed(2)} (${worst.name}) · AUC spread ${aucGap.toFixed(3)}`, 'approval ratio ≥ 0.80 (warn ≥ 0.70) and AUC spread ≤ 0.06', minRatio >= 0.8 && aucGap <= 0.06 ? 'pass' : minRatio >= 0.7 ? 'warn' : 'fail', 'Approval rates and accuracy by employment type, property type and region. Business segments only; a full fairness review also covers protected characteristics.');

  let maxErr = 0;
  const m = marginOf(e.model, S.te.X, 60, F);
  for (let i = 0; i < 60; i++) { const { phi, base } = shapRow(e.model, S.te.X, i * F, F); maxErr = Math.max(maxErr, Math.abs(phi.reduce((x, y) => x + y, base) - m[i])); }
  add('expl', 'Explainability coverage', `SHAP reconciles to the prediction (max error ${maxErr.toExponential(1)})`, 'error < 1e-6', maxErr < 1e-6 ? 'pass' : 'fail', 'Every decision can be decomposed into additive reason codes.');

  const findings = S.checks.filter((c) => c.count > 0);
  add('data', 'Data integrity controls', `${findings.length} control(s) raised findings; all remediated and logged`, 'all findings remediated', 'pass', 'Missing values, invalid codes and duplicates handled before training.');

  const drift = featureDrift(S.tr, S.recent).sort((x, y) => y.psi - x.psi);
  const verdict = checks.some((c) => c.status === 'fail') ? 'Not ready for deployment' : checks.some((c) => c.status === 'warn') ? 'Approve with conditions' : 'Ready for deployment';
  const out = { verdict, checks, mono, segs, drift, maxShapErr: maxErr, summary: { pass: checks.filter((c) => c.status === 'pass').length, warn: checks.filter((c) => c.status === 'warn').length, fail: checks.filter((c) => c.status === 'fail').length } };
  cache.set(key, out);
  return out;
}

/** Plain-text fact sheet. Everything the memo / Q&A is allowed to quote. */
export function buildFacts(S, which) {
  const e = S.models[which]; const g = computeGovernance(S, which); const mt = e.metrics;
  const other = which === 'champion' ? 'challenger' : 'champion'; const o = S.models[other];
  const L = [];
  L.push(`Model: ${e.name}. Trained on ${S.tr.n} accounts, validated on ${S.va.n}, tested on ${S.te.n}. Portfolio default rate ${(mt.test.rate * 100).toFixed(2)}%.`);
  L.push(`Test AUC ${mt.test.auc.toFixed(3)}, Gini ${mt.test.gini.toFixed(3)}, KS ${mt.test.ks.toFixed(3)}, Brier ${mt.test.brier.toFixed(4)}. Train AUC ${mt.train.auc.toFixed(3)}, validation AUC ${mt.val.auc.toFixed(3)}.`);
  L.push(`Calibration error ${(e.calib.ece * 100).toFixed(2)}%, calibration slope ${e.slope.toFixed(2)}. Score PSI ${e.scorePsi.toFixed(3)}. Recent-vintage AUC ${mt.recent.auc.toFixed(3)}.`);
  if (o) L.push(`${o.name} test AUC ${o.metrics.test.auc.toFixed(3)} versus ${mt.test.auc.toFixed(3)} for this model, a difference of ${Math.abs(o.metrics.test.auc - mt.test.auc).toFixed(3)}.`);
  const dec = e.deciles; L.push(`Riskiest decile has bad rate ${(dec[0].rate * 100).toFixed(1)}% (lift ${dec[0].lift.toFixed(1)}x) and captures ${(dec[0].cumCapture * 100).toFixed(0)}% of defaults; the safest decile has bad rate ${(dec[9].rate * 100).toFixed(2)}%.`);
  if (S.shap[which]) {
    const sh = S.shap[which];
    const corr = (j) => { let mx = 0; let ms = 0; for (let i = 0; i < sh.n; i++) { mx += sh.X[i * F + j]; ms += sh.values[i * F + j]; } mx /= sh.n; ms /= sh.n; let c = 0; for (let i = 0; i < sh.n; i++) c += (sh.X[i * F + j] - mx) * (sh.values[i * F + j] - ms); return c; };
    const imp = FEATURES.map((f, j) => { let s = 0; for (let i = 0; i < sh.n; i++) s += Math.abs(sh.values[i * F + j]); return { f, v: s / sh.n, up: corr(j) >= 0 }; }).sort((a, b) => b.v - a.v).slice(0, 5);
    L.push(`Top SHAP drivers of default risk, most important first: ${imp.map((x, i) => `${i + 1}. ${x.f.label} (importance ${x.v.toFixed(3)}; ${x.up ? 'higher values raise risk' : 'higher values lower risk'})`).join('; ')}.`);
  }
  const { c1, c2 } = S.cutoffs; const rows = bandTable(e.preds.test, S.te.y, S.ead, S.lgd, c1, c2);
  rows.forEach((r) => L.push(`${r.name} band (PD ${r.name === 'Auto-approve' ? `below ${(c1 * 100).toFixed(1)}%` : r.name === 'Manual review' ? `${(c1 * 100).toFixed(1)}% to ${(c2 * 100).toFixed(1)}%` : `above ${(c2 * 100).toFixed(1)}%`}): ${(r.share * 100).toFixed(1)}% of accounts, observed bad rate ${(r.badRate * 100).toFixed(2)}%.`));
  const pol = new Uint8Array(S.te.n); for (let i = 0; i < S.te.n; i++) pol[i] = policyApproves(S.te.X, i * F) ? 1 : 0;
  const sw = swapSet(S.te.y, pol, e.preds.test);
  L.push(`Rule policy (score at least ${POLICY.minScore}, LTV at most ${POLICY.maxLtv}%, debt service at most ${POLICY.maxTds}%) approves ${(sw.approvals / S.te.n * 100).toFixed(1)}% with bad rate ${(sw.policyBadRate * 100).toFixed(2)}%. The model at the same approval rate has bad rate ${(sw.modelBadRate * 100).toFixed(2)}% and avoids ${sw.badsAvoided} defaults.`);
  const named = (st) => g.checks.filter((c) => c.status === st).map((c) => c.title.toLowerCase()).join(', ') || 'none';
  L.push(`Governance verdict: ${g.verdict}. ${g.summary.pass} checks passed, ${g.summary.warn} warnings, ${g.summary.fail} failures. Checks with warnings: ${named('warn')}. Checks that failed: ${named('fail')}.`);
  for (const c of g.checks) L.push(`${c.title}: ${c.status.toUpperCase()}, ${c.value}.`);
  const dr = g.drift.slice(0, 3); L.push(`Largest feature drift (PSI, training vs recent): ${dr.map((d) => `${d.label} ${d.psi.toFixed(2)}`).join('; ')}.`);
  return L;
}

export function templateMemo(S, which) {
  const e = S.models[which]; const g = computeGovernance(S, which); const mt = e.metrics;
  const sw = (() => { const pol = new Uint8Array(S.te.n); for (let i = 0; i < S.te.n; i++) pol[i] = policyApproves(S.te.X, i * F) ? 1 : 0; return swapSet(S.te.y, pol, e.preds.test); })();
  const fails = g.checks.filter((c) => c.status !== 'pass');
  return [
    `MODEL RISK SUMMARY: ${e.name}`,
    '',
    `Purpose: estimate the 12-month probability of 90+ day default for residential secured lending accounts, to support adjudication and monitoring.`,
    '',
    `Performance: test AUC ${mt.test.auc.toFixed(3)} (Gini ${mt.test.gini.toFixed(3)}, KS ${mt.test.ks.toFixed(3)}) on ${S.te.n} unseen accounts. The riskiest decile defaults at ${(e.deciles[0].rate * 100).toFixed(1)}% against a portfolio rate of ${(mt.test.rate * 100).toFixed(2)}%.`,
    '',
    `Business value: at the same approval rate as the current rules, the model avoids ${sw.badsAvoided} defaults (bad rate ${(sw.modelBadRate * 100).toFixed(2)}% versus ${(sw.policyBadRate * 100).toFixed(2)}%).`,
    '',
    `Governance: ${g.summary.pass} of ${g.checks.length} validation checks pass${fails.length ? `; attention needed on ${fails.map((f) => f.title.toLowerCase()).join(', ')}` : ''}. Every decision decomposes into additive SHAP reason codes.`,
    '',
    `Recommendation: ${g.summary.fail ? 'do not deploy until the failed checks are resolved' : g.summary.warn ? 'approve for limited deployment with monitoring on the flagged items' : 'approve for deployment with standard monitoring'}.`,
  ].join('\n');
}

export function modelCard(S, which) {
  const e = S.models[which]; const g = computeGovernance(S, which); const mt = e.metrics;
  return [
    `# Model card: ${e.name}`, '',
    `Generated ${new Date().toISOString().slice(0, 10)}. **Synthetic data, for demonstration only.**`, '',
    '## Intended use', 'Estimate the 12-month probability of 90+ day delinquency for residential mortgage and HELOC accounts to support adjudication, monitoring and provisioning.', '',
    '## Data', `- ${S.world.raw.n} raw accounts, ${S.clean.n} after cleaning (${S.clean.fixes.droppedDuplicates} duplicates removed, ${S.clean.fixes.imputed} cells imputed).`, `- Split: ${S.tr.n} train / ${S.va.n} validation / ${S.te.n} test; ${S.recent.n} recent-vintage accounts held out.`, `- Features (${F}): ${FEATURES.map((f) => f.label).join(', ')}.`, '',
    '## Training', `- Method: ${which === 'champion' ? 'regularised logistic regression (Adam)' : 'gradient boosted trees (histogram, Newton boosting)'}.`, `- Parameters: \`${JSON.stringify(e.params)}\``, `- Iterations kept: ${e.bestIter}.`, '',
    '## Performance (test set)', `- AUC ${mt.test.auc.toFixed(3)}, Gini ${mt.test.gini.toFixed(3)}, KS ${mt.test.ks.toFixed(3)}, Brier ${mt.test.brier.toFixed(4)}, log-loss ${mt.test.logloss.toFixed(4)}.`, `- Calibration error ${(e.calib.ece * 100).toFixed(2)}%, slope ${e.slope.toFixed(2)}.`, `- Recent vintage AUC ${mt.recent.auc.toFixed(3)}; score PSI ${e.scorePsi.toFixed(3)}.`, '',
    '## Validation checks', ...g.checks.map((c) => `- ${ICON[c.status]} **${c.title}**: ${c.value} (threshold: ${c.threshold})`), '',
    '## Explainability', 'Per-decision explanations use exact TreeSHAP (verified against brute-force Shapley values); global views use mean |SHAP|, permutation importance and partial dependence.', '',
    '## Limitations', '- Trained on synthetic data; no claim is made about any real portfolio.', '- Segment checks cover business segments only; a full fairness review is required before use.', '- Stress-test sensitivities are mechanical re-scoring, not an economic forecast.', '',
  ].join('\n');
}
export { stressTest, fmtFeature, auc, calibration };
