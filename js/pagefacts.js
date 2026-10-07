// Page-aware fact sheets. Each tab contributes what is on screen right now (selected feature, account, model, cutoffs, sliders)
// plus a short glossary, so the page chatbot can answer from the same numbers the user is looking at. No DOM access.
import { F, FEATURES, FIDX, fmtFeature, column, featureProfile } from './data.js';
import { partialDependence, shapRow } from './ml.js';
import { reasons, recourse, bandTable, swapSet, policyApproves, POLICY, stressTest } from './explain.js';
import { computeGovernance, buildFacts } from './governance.js';
import { PRESETS } from './pipeline.js';

const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const n3 = (v) => Number(v).toFixed(3);
const label = (k) => (k === 'champion' ? 'Champion (logistic scorecard)' : 'Challenger (gradient boosting)');
const trained = (S) => ['champion', 'challenger'].filter((k) => S.models[k]);

export const PAGES = {
  data: 'Data', train: 'Train', eval: 'Evaluate', explain: 'Explain', decide: 'Decide', govern: 'Govern', copilot: 'Copilot',
};

export const SUGGESTIONS = {
  data: ['Which integrity problems were found?', 'How were missing values handled?', 'What does the selected feature tell us?', 'Why is the split stratified?'],
  train: ['Is the challenger overfitting?', 'Compare the champion and challenger', 'What does the learning rate do?', 'What does the Overfit preset do?'],
  eval: ['How good is this model?', 'How well calibrated is it?', 'What happens at the current cutoff?', 'Which decile is riskiest?'],
  explain: ['What drives defaults overall?', 'Why was this account scored this way?', 'What would change this decision?', 'Is the selected feature behaving sensibly?'],
  decide: ['How do the cutoffs split the book?', 'How does the model compare with the rules?', 'Which band holds the most defaults?', 'How big is the expected loss?'],
  govern: ['Which checks have warnings and why?', 'Is the model stable over time?', 'What happens under the current stress?', 'Should we deploy this model?'],
  copilot: ['Should we deploy this model?', 'Which features drive defaults?', 'How does it compare to the other model?', 'What are the biggest risks?'],
};

const GLOSSARY = {
  data: [
    'Definition: a stratified split keeps the same default rate in the train, validation and test sets, so differences in performance are not caused by an unlucky split.',
    'Definition: median imputation replaces missing or invalid values with the median of the training set only, so no information leaks from validation or test data.',
    'Definition: the recent vintage is a block of newer accounts held out of training to test how the model copes with drift in the population.',
  ],
  train: [
    'Definition: log-loss measures how far predicted probabilities are from the outcomes; lower is better. AUC measures how well accounts are ranked by risk; higher is better.',
    'Definition: overfitting means the model memorises its training data, so training performance keeps improving while validation performance gets worse. Underfitting means it is too simple to learn the pattern.',
    'Definition: learning rate controls how big each boosting step is; small steps need more trees but generalise better. Tree depth controls how many interactions a tree can capture. Min samples per leaf and L2 on leaf values regularise the trees.',
    'Definition: early stopping halts training when validation loss stops improving for a number of rounds (the patience) and keeps the best iteration.',
    'Definition: a default class weight above 1 makes mistakes on defaults count more, which raises recall of defaults but distorts the predicted probabilities.',
  ],
  eval: [
    'Definition: AUC is the probability that a random defaulter is scored riskier than a random non-defaulter (0.5 is random, 1.0 is perfect). Gini is 2 times AUC minus 1.',
    'Definition: KS is the maximum gap between the cumulative share of defaulters and non-defaulters ordered by score; higher means better separation.',
    'Definition: calibration compares predicted PD with the observed default rate. A calibration slope near 1 means well calibrated; below 1 means over-confident; above 1 means under-confident.',
    'Definition: lift is a decile bad rate divided by the portfolio bad rate. The Brier score is the mean squared error of the predicted probabilities (lower is better).',
  ],
  explain: [
    'Definition: SHAP splits each prediction into one additive contribution per feature, measured in log-odds. Positive values raise predicted default risk, negative values lower it, and they sum to the model output.',
    'Definition: in the beeswarm each dot is an account; position is the SHAP value, colour is the feature value (red = high, blue = low).',
    'Definition: partial dependence shows the average predicted PD when one feature is set to each value for every account; ICE lines show individual accounts.',
    'Definition: permutation importance is the drop in AUC when a feature is shuffled; it is a model-agnostic cross-check of SHAP.',
    'Definition: reason codes are the top features raising an account\'s risk, in plain language, as used in adverse-action notices. Recourse shows a change in the borrower\'s inputs that would move the decision to the next band, holding everything else fixed.',
  ],
  decide: [
    'Definition: auto-approve accounts have PD below the first cutoff, decline accounts have PD at or above the second cutoff, and everything between goes to manual review.',
    'Definition: expected loss is PD times loss given default (LGD) times exposure. Realised loss uses what actually happened in the test set.',
    `Definition: the current rules approve an account if credit score is at least ${POLICY.minScore}, LTV is at most ${POLICY.maxLtv}% and debt service is at most ${POLICY.maxTds}%. The swap-set analysis holds the model to the same approval rate; swap-in accounts are declined by the rules but approved by the model, swap-out accounts are the reverse.`,
  ],
  govern: [
    'Definition: PSI (population stability index) compares two distributions; below 0.10 is stable, 0.10 to 0.25 is a noticeable shift, above 0.25 is significant.',
    'Definition: the monotonicity check verifies that the model behaves as an underwriter expects, for example higher LTV, debt service, delinquencies or unemployment should not lower risk.',
    'Definition: the out-of-time check scores the newest originations, whose economics have shifted, and compares AUC with the test set.',
    'Definition: segment consistency compares approval rates and AUC across employment type, property type and region; a ratio below 0.80 against the best segment is flagged for review. It covers business segments only and is not a full fairness review.',
  ],
  copilot: [],
};

function overview(S) {
  const L = [];
  for (const k of trained(S)) { const e = S.models[k]; L.push(`Overview: ${label(k)} has test AUC ${n3(e.metrics.test.auc)}, Gini ${n3(e.metrics.test.gini)} and KS ${n3(e.metrics.test.ks)} on ${S.te.n} unseen accounts.`); }
  return L;
}

// ---- Data ------------------------------------------------------------------------------------------------------------------
function dataFacts(S, st) {
  const rate = S.clean.y.reduce((a, b) => a + b, 0) / S.clean.n;
  const L = [
    `Page: Data. ${S.world.raw.n} raw accounts, ${S.clean.n} after cleaning (${S.clean.fixes.droppedDuplicates} duplicates removed). Default rate ${pc(rate, 2)}; the target event is 90+ days past due within 12 months.`,
    `Split: ${S.tr.n} train, ${S.va.n} validation and ${S.te.n} test accounts, stratified so each has the same default rate; ${S.recent.n} recent-vintage accounts are held out for drift checks. Features: ${F}.`,
  ];
  for (const c of S.checks) L.push(`Integrity control "${c.name}" (${c.category}): ${c.count} records (${pc(c.rate, 2)}); status ${c.count === 0 ? 'pass' : /Flag|Review/.test(c.action) ? 'flagged for review' : 'fixed'}; action: ${c.action}.`);
  L.push(`Applied fixes: ${S.clean.fixes.invalid} invalid values set to missing, ${S.clean.fixes.imputed} cells imputed with training-set medians, ${S.clean.fixes.droppedDuplicates} duplicate accounts dropped.`);
  const j = FIDX[st.feature]; const f = FEATURES[j]; const p = featureProfile(S.tr, j, f.binary ? 2 : 20);
  const idx = p.rate.map((v, i) => (v === null ? -1 : i)).filter((i) => i >= 0);
  if (idx.length) {
    const a = idx[0]; const b = idx[idx.length - 1];
    const at = (i) => (f.binary ? (p.centers[i] ? 'Yes' : 'No') : fmtFeature(f.key, p.centers[i]));
    const tot = p.good.reduce((x, y) => x + y, 0) + p.bad.reduce((x, y) => x + y, 0);
    L.push(`Feature on screen: ${f.label} (${f.desc}). Default rate is ${p.rate[a].toFixed(2)}% at ${at(a)} versus ${p.rate[b].toFixed(2)}% at ${at(b)}; portfolio average ${(p.bad.reduce((x, y) => x + y, 0) / tot * 100).toFixed(2)}%.`);
  }
  for (let k = 0; k < F; k++) {
    const ft = FEATURES[k]; const c = column(S.tr, k); const med = [...c].sort((x, y) => x - y)[c.length >> 1]; let lo = [0, 0]; let hi = [0, 0];
    for (let i = 0; i < S.tr.n; i++) { const t = c[i] <= med ? lo : hi; t[0]++; t[1] += S.tr.y[i]; }
    const mean = c.reduce((x, y) => x + y, 0) / c.length;
    L.push(`Feature ${ft.label}: ${ft.desc}; expected effect ${ft.sign > 0 ? 'higher is riskier' : ft.sign < 0 ? 'higher is safer' : 'no fixed direction'}; mean ${fmtFeature(ft.key, mean)}; default rate ${pc(lo[1] / Math.max(1, lo[0]))} at or below the median versus ${pc(hi[1] / Math.max(1, hi[0]))} above it.`);
  }
  return L;
}

// ---- Train -----------------------------------------------------------------------------------------------------------------
function trainFacts(S, st) {
  const L = [`Page: Train. Learning curves on screen show the ${label(st.view)}. ${st.training ? 'Training is running right now.' : ''}`];
  for (const k of trained(S)) {
    const e = S.models[k]; const h = e.history; const last = h[h.length - 1]; const minV = h.reduce((a, x) => (x.val.loss < a.val.loss ? x : a), h[0]);
    const gap = last.train.auc - last.val.auc;
    L.push(`${label(k)}: trained with ${JSON.stringify(e.params)} for ${last.iter} iterations${h.stopped ? ` (early stopping triggered at ${h.stopped}, best ${e.bestIter} kept)` : ''} in ${(e.trainMs / 1000).toFixed(1)}s.`);
    L.push(`${label(k)}: AUC train ${n3(e.metrics.train.auc)}, validation ${n3(e.metrics.val.auc)}, test ${n3(e.metrics.test.auc)}; train minus validation AUC gap ${n3(gap)}; final log-loss train ${last.train.loss.toFixed(4)} and validation ${last.val.loss.toFixed(4)}; lowest validation log-loss ${minV.val.loss.toFixed(4)} at iteration ${minV.iter}.`);
    const lossRise = (last.val.loss - minV.val.loss) / minV.val.loss;
    L.push(`${label(k)} diagnosis: ${gap > 0.08 || lossRise > 0.08 ? 'overfitting (training metrics far better than validation, and validation loss has risen after its best point)' : last.val.auc < 0.76 ? 'underfitting (validation AUC is low and close to training)' : 'healthy fit (training and validation close, validation loss near its best)'}.`);
  }
  if (trained(S).length === 2) { const d = S.models.challenger.metrics.test.auc - S.models.champion.metrics.test.auc; L.push(`Champion versus challenger: test AUC ${n3(S.models.champion.metrics.test.auc)} versus ${n3(S.models.challenger.metrics.test.auc)}, a difference of ${n3(Math.abs(d))} in favour of the ${d >= 0 ? 'challenger' : 'champion'}.`); }
  for (const k of ['champion', 'challenger']) L.push(`Slider settings for the ${label(k)} right now (may not be trained yet): ${JSON.stringify(st.params[k])}.`);
  for (const k of ['champion', 'challenger']) for (const [name, pr] of Object.entries(PRESETS[k])) L.push(`Preset "${pr.label}" for the ${label(k)}: ${pr.note}.`);
  return L;
}

// ---- Evaluate --------------------------------------------------------------------------------------------------------------
function evalFacts(S, st) {
  const L = [`Page: Evaluate. All metrics are on the test set of ${S.te.n} accounts the model never saw. The model shown in detail is the ${label(st.view)}.`];
  for (const k of trained(S)) {
    const e = S.models[k]; const m = e.metrics.test;
    L.push(`${label(k)} test performance: AUC ${n3(m.auc)}, Gini ${n3(m.gini)}, KS ${n3(m.ks)}, Brier ${m.brier.toFixed(4)}, log-loss ${m.logloss.toFixed(4)}; calibration slope ${e.slope.toFixed(2)} and calibration error ${pc(e.calib.ece, 2)}; mean predicted PD ${pc(m.meanPd, 2)} versus observed default rate ${pc(m.rate, 2)}.`);
  }
  const e = S.models[st.view];
  for (const d of e.deciles) L.push(`Risk decile ${d.decile}${d.decile === 1 ? ' (riskiest)' : d.decile === 10 ? ' (safest)' : ''}: ${d.n} accounts, ${d.bad} defaults, bad rate ${pc(d.rate)}, average predicted PD ${pc(d.avgPd)}, lift ${d.lift.toFixed(1)}x, cumulative share of defaults captured ${pc(d.cumCapture, 0)}.`);
  const p = e.preds.test; const y = S.te.y; let tp = 0; let fp = 0; let fn = 0; let tn = 0;
  for (let i = 0; i < p.length; i++) { const dec = p[i] >= st.cutoff; if (dec && y[i]) tp++; else if (dec) fp++; else if (y[i]) fn++; else tn++; }
  L.push(`Operating point on screen: decline when predicted PD is at or above ${pc(st.cutoff)}. Approved ${tn + fn} accounts (${pc((tn + fn) / p.length)}) with bad rate ${pc(fn / Math.max(1, tn + fn), 2)}; defaults caught ${tp} of ${tp + fn} (${pc(tp / Math.max(1, tp + fn), 0)}); good accounts declined ${fp} (${pc(fp / Math.max(1, fp + tn))}).`);
  return L;
}

// ---- Explain ---------------------------------------------------------------------------------------------------------------
function explainFacts(S, st) {
  const L = [`Page: Explain. The model under review is the ${label(st.view)}.`];
  const e = S.models[st.view]; const sh = S.shap[st.view];
  if (sh) {
    const corr = (j) => { let mx = 0; let ms = 0; for (let i = 0; i < sh.n; i++) { mx += sh.X[i * F + j]; ms += sh.values[i * F + j]; } mx /= sh.n; ms /= sh.n; let c = 0; for (let i = 0; i < sh.n; i++) c += (sh.X[i * F + j] - mx) * (sh.values[i * F + j] - ms); return c; };
    const imp = FEATURES.map((f, j) => { let s = 0; for (let i = 0; i < sh.n; i++) s += Math.abs(sh.values[i * F + j]); return { f, v: s / sh.n, up: corr(j) >= 0 }; }).sort((a, b) => b.v - a.v);
    L.push(`Global SHAP importance (mean absolute SHAP in log-odds over ${sh.n} test accounts), most important first: ${imp.slice(0, 8).map((x, i) => `${i + 1}. ${x.f.label} ${x.v.toFixed(3)} (${x.up ? 'higher values raise risk' : 'higher values lower risk'})`).join('; ')}.`);
    L.push(`Least influential features: ${imp.slice(-3).map((x) => `${x.f.label} ${x.v.toFixed(3)}`).join('; ')}.`);
  }
  if (st.perm) { const rows = FEATURES.map((f, j) => ({ f, v: st.perm.drops[j] })).sort((a, b) => b.v - a.v).slice(0, 5); L.push(`Permutation importance (AUC drop when shuffled): ${rows.map((x) => `${x.f.label} ${x.v.toFixed(4)}`).join('; ')}.`); }
  const j = FIDX[st.feat]; const f = FEATURES[j];
  const col = column(S.te, j); const s = Float64Array.from(col).sort();
  const lo = f.binary ? 0 : s[Math.floor(S.te.n * 0.05)]; const hi = f.binary ? 1 : s[Math.floor(S.te.n * 0.95)];
  const grid = f.binary ? [0, 1] : Array.from({ length: 5 }, (_, k) => lo + ((hi - lo) * k) / 4);
  const n = 300; const Xs = new Float64Array(n * F); for (let k = 0; k < n; k++) Xs.set(S.te.X.slice(Math.floor((k * S.te.n) / n) * F, Math.floor((k * S.te.n) / n) * F + F), k * F);
  const pdp = partialDependence(e.model, Xs, n, F, j, grid, 0);
  const rising = pdp.mean[pdp.mean.length - 1] > pdp.mean[0];
  L.push(`Feature deep-dive on screen: ${f.label}. Average predicted PD by value: ${grid.map((g, k) => `${fmtFeature(f.key, g)} gives ${pc(pdp.mean[k], 2)}`).join('; ')}. Overall, risk ${rising ? 'rises' : 'falls'} as the feature increases. Expected direction: ${f.sign > 0 ? 'risk should rise' : f.sign < 0 ? 'risk should fall' : 'no fixed expectation'}${f.sign !== 0 ? `; the model ${((f.sign > 0) === rising) ? 'agrees' : 'disagrees'} on the overall trend` : ''}.`);
  if (st.acct !== null && st.acct !== undefined) {
    const o = st.acct * F; const { phi, base } = shapRow(e.model, S.te.X, o, F);
    const pd = 1 / (1 + Math.exp(-(base + phi.reduce((a, b) => a + b, 0)))); const avg = 1 / (1 + Math.exp(-base));
    const { c1, c2 } = S.cutoffs; const band = pd < c1 ? 'Auto-approve' : pd < c2 ? 'Manual review' : 'Decline';
    L.push(`Account on screen: test account number ${st.acct}. Predicted PD ${pc(pd)} versus portfolio average ${pc(avg)}; decision ${band} (cutoffs ${pc(c1)} and ${pc(c2)}); actual outcome ${S.te.y[st.acct] ? 'defaulted' : 'did not default'}.`);
    L.push(`Account features: ${FEATURES.map((ft, k) => `${ft.label} ${fmtFeature(ft.key, S.te.X[o + k])}`).join('; ')}.`);
    const rs = reasons(phi, S.te.X, o);
    L.push(`Reasons raising this account's risk (SHAP in log-odds): ${rs.up.length ? rs.up.map((r) => `${r.text} (+${r.phi.toFixed(2)})`).join('; ') : 'none material'}.`);
    L.push(`Reasons lowering this account's risk: ${rs.down.length ? rs.down.map((r) => `${r.text} (${r.phi.toFixed(2)})`).join('; ') : 'none material'}.`);
    if (band !== 'Auto-approve') { const rc = recourse(e.model, S.te.X, o, band === 'Decline' ? c2 : c1); L.push(`What would change the decision to ${band === 'Decline' ? 'Manual review' : 'Auto-approve'} (single changes, everything else fixed): ${rc.options.length ? rc.options.map((r) => `${r.verb} from ${typeof r.from === 'number' ? fmtFeature(r.key, r.from) : r.from} to ${typeof r.to === 'number' ? fmtFeature(r.key, r.to) : r.to} gives PD ${pc(r.pd)}`).join('; ') : 'no single realistic change reaches the next band'}.`); }
    const other = st.view === 'champion' ? 'challenger' : 'champion'; if (S.models[other]) L.push(`The ${label(other)} gives this account PD ${pc(S.models[other].preds.test[st.acct])}.`);
  }
  return L;
}

// ---- Decide ----------------------------------------------------------------------------------------------------------------
function decideFacts(S, st) {
  const e = S.models[st.view]; const { c1, c2 } = S.cutoffs; const rows = bandTable(e.preds.test, S.te.y, S.ead, S.lgd, c1, c2);
  const L = [`Page: Decide. Model in use: ${label(st.view)}. Auto-approve below PD ${pc(c1)}, manual review from ${pc(c1)} to ${pc(c2)}, decline at or above ${pc(c2)}; loss given default ${pc(S.lgd, 0)}.`];
  const totBad = S.te.y.reduce((a, b) => a + b, 0); let eTot = 0; let rTot = 0;
  for (const r of rows) { eTot += r.expLoss; rTot += r.realLoss; L.push(`${r.name} band: ${r.n} accounts (${pc(r.share)}), average predicted PD ${pc(r.avgPd, 2)}, observed bad rate ${pc(r.badRate, 2)}, ${r.bad} defaults (${pc(r.bad / totBad, 0)} of all defaults), exposure $${(r.ead / 1e6).toFixed(0)}M, expected loss $${(r.expLoss / 1e6).toFixed(1)}M, realised loss $${(r.realLoss / 1e6).toFixed(1)}M.`); }
  L.push(`Total expected loss $${(eTot / 1e6).toFixed(1)}M versus realised loss $${(rTot / 1e6).toFixed(1)}M on the test book.`);
  const pol = new Uint8Array(S.te.n); for (let i = 0; i < S.te.n; i++) pol[i] = policyApproves(S.te.X, i * F) ? 1 : 0;
  const sw = swapSet(S.te.y, pol, e.preds.test);
  L.push(`Swap-set analysis at the same approval rate (${pc(sw.approvals / S.te.n)}): current rules bad rate ${pc(sw.policyBadRate, 2)} versus model bad rate ${pc(sw.modelBadRate, 2)}; the model avoids ${sw.badsAvoided} defaults.`);
  L.push(`Swap cells: approved by both ${sw.cells.both[0]} accounts (bad rate ${pc(sw.rates.both, 2)}); swap-out, rules approve but model declines, ${sw.cells.policyOnly[0]} accounts (bad rate ${pc(sw.rates.policyOnly, 2)}); swap-in, rules decline but model approves, ${sw.cells.modelOnly[0]} accounts (bad rate ${pc(sw.rates.modelOnly, 2)}); declined by both ${sw.cells.neither[0]} accounts (bad rate ${pc(sw.rates.neither, 2)}).`);
  return L;
}

// ---- Govern ----------------------------------------------------------------------------------------------------------------
function governFacts(S, st) {
  const g = computeGovernance(S, st.view); const L = [];
  L.push(`Page: Govern. Validation scorecard for the ${label(st.view)}: verdict ${g.verdict}; ${g.summary.pass} checks passed, ${g.summary.warn} warnings, ${g.summary.fail} failures out of ${g.checks.length}.`);
  for (const c of g.checks) L.push(`Check "${c.title}": ${c.status.toUpperCase()}; ${c.value}; threshold ${c.threshold}. ${c.detail}`);
  L.push(`Largest population drift (PSI, training versus recent vintage): ${g.drift.slice(0, 5).map((d) => `${d.label} ${d.psi.toFixed(3)}`).join('; ')}.`);
  for (const [k, rows] of Object.entries(g.segs)) L.push(`Segments by ${k.toLowerCase()}: ${rows.map((s) => `${s.name} ${s.n} accounts, bad rate ${pc(s.badRate)}, approval ${pc(s.approval, 0)} (ratio ${s.ratio.toFixed(2)}), AUC ${Number.isNaN(s.auc) ? 'n/a' : s.auc.toFixed(3)}`).join('; ')}.`);
  const e = S.models[st.view];
  const base = stressTest(e.model, S.te.X, S.te.n, S.ead, S.lgd, {}, S.te.seg.region);
  const sv = stressTest(e.model, S.te.X, S.te.n, S.ead, S.lgd, st.shock, S.te.seg.region);
  L.push(`Stress sliders now: unemployment +${st.shock.dUnemp} points, house prices ${st.shock.dHpi}%, interest rates +${st.shock.dRate} bps at renewal. Mean predicted PD ${pc(base.meanPd, 2)} at baseline versus ${pc(sv.meanPd, 2)} stressed; expected loss $${(base.expLoss / 1e6).toFixed(1)}M versus $${(sv.expLoss / 1e6).toFixed(1)}M.`);
  const rec = stressTest(e.model, S.te.X, S.te.n, S.ead, S.lgd, { dUnemp: 3, dHpi: -20, dRate: 200 }, S.te.seg.region);
  L.push(`Reference recession scenario (unemployment +3 points, house prices -20%, interest rates +200 bps at renewal): mean predicted PD ${pc(rec.meanPd, 2)} versus ${pc(base.meanPd, 2)} at baseline, a relative increase of ${((rec.meanPd / base.meanPd - 1) * 100).toFixed(0)}%; expected loss $${(rec.expLoss / 1e6).toFixed(1)}M versus $${(base.expLoss / 1e6).toFixed(1)}M. Move the sliders to try other shocks.`);
  L.push(`Stressed mean PD by region: ${sv.byRegion.map((r, i) => `${r.name} ${pc(r.pd, 2)} (baseline ${pc(base.byRegion[i].pd, 2)})`).join('; ')}.`);
  L.push(`Out-of-time: AUC ${n3(e.metrics.recent.auc)} on recent vintage versus ${n3(e.metrics.test.auc)} on test; mean predicted PD ${pc(e.metrics.test.meanPd, 2)} on test versus ${pc(e.metrics.recent.meanPd, 2)} on recent versus observed ${pc(e.metrics.recent.rate, 2)}; score PSI ${e.scorePsi.toFixed(3)}.`);
  return L;
}

/** Fact sheet for a page given its current UI state. Returns an array of plain-text lines. */
export function pageFacts(S, tab, st) {
  if (!trained(S).length) return ['No model has been trained yet.'];
  const ok = (v) => (S.models[v] ? v : trained(S)[0]);
  for (const k of Object.keys(st)) if (st[k] && st[k].view) st[k] = { ...st[k], view: ok(st[k].view) };
  let L;
  if (tab === 'data') L = dataFacts(S, st.data);
  else if (tab === 'train') L = trainFacts(S, st.train);
  else if (tab === 'eval') L = evalFacts(S, st.eval);
  else if (tab === 'explain') L = explainFacts(S, st.explain);
  else if (tab === 'decide') L = decideFacts(S, st.decide);
  else if (tab === 'govern') L = governFacts(S, st.govern);
  else L = buildFacts(S, st.copilot.view);
  const extra = tab === 'copilot' ? [] : overview(S);
  return [...L, ...extra, ...(GLOSSARY[tab] || [])];
}
