// Workspace state: builds the data, trains models, caches predictions, metrics and SHAP values.
import { F, FEATURES, FIDX, generate, integrityChecks, cleanData, split, take, column } from './data.js';
import { trainLogistic, trainGBM, predictProba, summarize, rocPoints, calibration, calibrationSlope, deciles, psi, shapMatrix, tick } from './ml.js';

export const DEFAULTS = {
  champion: { epochs: 250, lr: 0.05, l2: 0.01, posWeight: 1 },
  challenger: { trees: 300, lr: 0.04, maxDepth: 4, minChildWeight: 15, lambda: 10, subsample: 0.8, posWeight: 1, patience: 30 },
};
export const PRESETS = {
  challenger: {
    underfit: { label: 'Underfit', note: '10 stumps: too simple to learn the pattern', p: { trees: 10, lr: 0.1, maxDepth: 1, minChildWeight: 20, lambda: 5, subsample: 1, posWeight: 1, patience: 0 } },
    balanced: { label: 'Balanced', note: 'Regularised, early stopping on validation loss', p: DEFAULTS.challenger },
    overfit: { label: 'Overfit', note: 'Deep trees, no regularisation: memorises the training set', p: { trees: 120, lr: 0.3, maxDepth: 6, minChildWeight: 1, lambda: 0, subsample: 1, posWeight: 1, patience: 0 } },
  },
  champion: {
    underfit: { label: 'Barely trained', note: '12 epochs: coefficients have not converged', p: { epochs: 12, lr: 0.02, l2: 0.01, posWeight: 1 } },
    balanced: { label: 'Converged', note: 'Standard regularised scorecard fit', p: DEFAULTS.champion },
    overfit: { label: 'Rebalanced', note: 'Class weight 8x on defaults: higher recall, worse calibration', p: { epochs: 250, lr: 0.05, l2: 0.01, posWeight: 8 } },
  },
};

export async function createWorkspace(onStatus = () => {}) {
  onStatus('Generating 30,000 synthetic accounts…', 0.05); await tick();
  const world = generate({ n: 30000, recent: 6000, seed: 2026 });
  onStatus('Running data integrity controls…', 0.2); await tick();
  const checks = integrityChecks(world.raw);
  const sp0 = split(world.raw.n, 7, world.raw.y);
  const clean = cleanData(world.raw, sp0.train);
  const sp = split(clean.n, 7, clean.y);
  const tr = take(clean, sp.train); const va = take(clean, sp.val); const te = take(clean, sp.test);
  const recent = take({ ...world.recent }, Array.from({ length: world.recent.n }, (_, i) => i));
  const ead = column(te, FIDX.loan_amount).map((v) => v * 1000);
  return { world, checks, clean, sp, tr, va, te, recent, ead, lgd: 0.25, models: { champion: null, challenger: null }, shap: {}, cutoffs: { c1: 0.03, c2: 0.1 } };
}

export function evaluateEntry(S, entry) {
  const sets = { train: S.tr, val: S.va, test: S.te, recent: S.recent };
  const preds = {}; const metrics = {};
  for (const [name, d] of Object.entries(sets)) { preds[name] = predictProba(entry.model, d.X, d.n, F); metrics[name] = summarize(d.y, preds[name]); }
  entry.preds = preds; entry.metrics = metrics;
  entry.roc = rocPoints(S.te.y, preds.test);
  entry.calib = calibration(S.te.y, preds.test, 10);
  entry.slope = calibrationSlope(S.te.y, preds.test);
  entry.deciles = deciles(S.te.y, preds.test);
  entry.scorePsi = psi(preds.train, preds.recent);
  return entry;
}

export async function trainModel(S, which, params, onProgress) {
  const t0 = performance.now();
  const val = { X: S.va.X, y: S.va.y, n: S.va.n };
  const r = which === 'champion'
    ? await trainLogistic(S.tr.X, S.tr.y, S.tr.n, F, params, val, onProgress)
    : await trainGBM(S.tr.X, S.tr.y, S.tr.n, F, params, val, onProgress);
  const entry = { which, name: which === 'champion' ? 'Champion · logistic scorecard' : 'Challenger · gradient boosting', params: { ...params }, model: r.model, history: r.history, bestIter: r.bestIter, trainMs: performance.now() - t0, trainedAt: new Date() };
  evaluateEntry(S, entry);
  S.models[which] = entry;
  delete S.shap[which];
  return entry;
}

/** SHAP on a fixed random sample of test accounts (cached per model). */
export async function shapSample(S, which, onProgress) {
  if (S.shap[which]) return S.shap[which];
  const entry = S.models[which];
  const n = Math.min(320, S.te.n);
  const idx = Array.from({ length: n }, (_, i) => Math.floor((i * S.te.n) / n));
  const X = new Float64Array(n * F);
  idx.forEach((i, k) => X.set(S.te.X.slice(i * F, i * F + F), k * F));
  const r = await shapMatrix(entry.model, X, n, F, onProgress);
  S.shap[which] = { idx, X, values: r.values, base: r.base, n };
  return S.shap[which];
}

export const modelEntries = (S) => ['champion', 'challenger'].filter((k) => S.models[k]).map((k) => S.models[k]);
export { FEATURES };
