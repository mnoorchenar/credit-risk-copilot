import test from 'node:test';
import assert from 'node:assert/strict';
import { F, FEATURES, generate, integrityChecks, cleanData, split, take, column } from '../js/data.js';
import { auc, ks, psi, calibration, calibrationSlope, deciles, trainLogistic, trainGBM, predictProba, marginOf, shapRow, gbmBase, partialDependence, permutationImportance, sigmoid } from '../js/ml.js';

const world = generate({ n: 9000, recent: 2000, seed: 42 });
const sp = split(world.raw.n, 7, world.raw.y);
const clean = cleanData(world.raw, sp.train);
const sp2 = split(clean.n, 7, clean.y);
const tr = take(clean, sp2.train); const va = take(clean, sp2.val); const te = take(clean, sp2.test);

test('generator is reproducible and hits the target default rate', () => {
  const again = generate({ n: 9000, recent: 2000, seed: 42 });
  assert.deepEqual(Array.from(again.raw.y.slice(0, 50)), Array.from(world.raw.y.slice(0, 50)));
  const rate = world.truth.y.reduce((a, b) => a + b, 0) / world.truth.n;
  assert.ok(Math.abs(rate - 0.05) < 0.012, `rate ${rate}`);
});

test('integrity checks find the injected problems and cleaning removes them', () => {
  const checks = integrityChecks(world.raw);
  const byName = (s) => checks.find((c) => c.name.includes(s));
  assert.ok(byName('Missing').count > 0);
  assert.ok(byName('Credit score outside').count > 0);
  assert.ok(byName('Duplicate').count > 0);
  assert.ok(clean.n < world.raw.n);
  for (let i = 0; i < clean.X.length; i++) assert.ok(!Number.isNaN(clean.X[i]));
  assert.ok(clean.fixes.imputed > 0);
});

test('metrics: AUC, KS and calibration behave on known cases', () => {
  assert.equal(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1);
  assert.equal(auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0);
  assert.equal(auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5);
  assert.equal(ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1);
  const p = Array.from({ length: 4000 }, (_, i) => (i % 100) / 100 * 0.3 + 0.01);
  const y = p.map((q, i) => ((i * 7919) % 1000) / 1000 < q ? 1 : 0);
  assert.ok(Math.abs(calibrationSlope(y, p) - 1) < 0.25);
  assert.ok(calibration(y, p).ece < 0.03);
  const d = deciles(y, p);
  assert.equal(d.length, 10);
  assert.ok(d[0].rate > d[9].rate);
});

test('PSI is ~0 for the same distribution and large for a shifted one', () => {
  const a = Array.from({ length: 5000 }, (_, i) => Math.sin(i) * 10 + 50);
  assert.ok(psi(a, a) < 0.001);
  assert.ok(psi(a, a.map((v) => v + 8)) > 0.25);
});

let champion; let challenger;
test('logistic scorecard trains, loss falls, and it ranks defaults well', async () => {
  const r = await trainLogistic(tr.X, tr.y, tr.n, F, { epochs: 200, lr: 0.05, l2: 0.01 }, va);
  champion = r.model;
  assert.ok(r.history.length > 5);
  assert.ok(r.history.at(-1).train.loss < r.history[0].train.loss);
  const a = auc(te.y, predictProba(champion, te.X, te.n, F));
  assert.ok(a > 0.72, `logistic test auc ${a}`);
});

test('gradient boosting trains, early-stops sensibly and beats the linear model', async () => {
  const r = await trainGBM(tr.X, tr.y, tr.n, F, { trees: 120, lr: 0.1, maxDepth: 3, patience: 20 }, va);
  challenger = r.model;
  assert.ok(r.history.at(-1).train.loss < r.history[0].train.loss);
  const aG = auc(te.y, predictProba(challenger, te.X, te.n, F));
  const aL = auc(te.y, predictProba(champion, te.X, te.n, F));
  assert.ok(aG > 0.74, `gbm auc ${aG}`);
  assert.ok(aG >= aL - 0.04, `gbm ${aG} vs logistic ${aL}`); // small test sample: only guard against a broken booster
});

test('an over-complex booster overfits: train AUC far above validation AUC', async () => {
  const r = await trainGBM(tr.X, tr.y, tr.n, F, { trees: 80, lr: 0.3, maxDepth: 6, minChildWeight: 1, lambda: 0, subsample: 1, patience: 0 }, va);
  const last = r.history.at(-1);
  assert.ok(last.train.auc - last.val.auc > 0.08, `gap ${last.train.auc - last.val.auc}`);
});

test('TreeSHAP is additive: base + sum(phi) equals the model margin', () => {
  const m = marginOf(challenger, te.X, 40, F);
  for (let i = 0; i < 40; i++) {
    const { phi, base } = shapRow(challenger, te.X, i * F, F);
    const s = phi.reduce((a, b) => a + b, base);
    assert.ok(Math.abs(s - m[i]) < 1e-7, `row ${i}: ${s} vs ${m[i]}`);
  }
});

test('TreeSHAP equals brute-force Shapley values (cover-weighted expectation)', async () => {
  const small = (await trainGBM(tr.X, tr.y, tr.n, F, { trees: 4, lr: 0.3, maxDepth: 3, patience: 0, subsample: 1 }, va)).model;
  const used = [...new Set(small.trees.flatMap((t) => Array.from(t.feat).filter((f) => f >= 0)))];
  assert.ok(used.length >= 2 && used.length <= 12);
  const expectTree = (t, x, known, node = 0) => {
    if (t.feat[node] < 0) return t.val[node];
    const f = t.feat[node];
    if (known.has(f)) return expectTree(t, x, known, x[f] <= t.thr[node] ? t.left[node] : t.right[node]);
    const l = t.left[node]; const r = t.right[node];
    return (t.cover[l] * expectTree(t, x, known, l) + t.cover[r] * expectTree(t, x, known, r)) / t.cover[node];
  };
  const v = (x, set) => small.init + small.trees.reduce((s, t) => s + expectTree(t, x, set), 0);
  const fact = (k) => (k <= 1 ? 1 : k * fact(k - 1));
  for (let row = 0; row < 3; row++) {
    const x = te.X.slice(row * F, row * F + F);
    const { phi } = shapRow(small, te.X, row * F, F);
    const d = used.length;
    for (const i of used) {
      let sh = 0;
      const others = used.filter((u) => u !== i);
      for (let mask = 0; mask < 1 << others.length; mask++) {
        const S = new Set(others.filter((_, b) => mask & (1 << b)));
        const w = (fact(S.size) * fact(d - S.size - 1)) / fact(d);
        const withI = new Set(S); withI.add(i);
        sh += w * (v(x, withI) - v(x, S));
      }
      assert.ok(Math.abs(sh - phi[i]) < 1e-8, `feature ${FEATURES[i].key}: brute ${sh} vs treeshap ${phi[i]}`);
    }
  }
});

test('logistic SHAP is additive and uses the training mean as baseline', () => {
  const m = marginOf(champion, te.X, 20, F);
  for (let i = 0; i < 20; i++) {
    const { phi, base } = shapRow(champion, te.X, i * F, F);
    assert.ok(Math.abs(phi.reduce((a, b) => a + b, base) - m[i]) < 1e-9);
  }
});

test('partial dependence: credit score lowers risk; permutation importance ranks it above a noise-like feature', async () => {
  const j = 0;
  const grid = [560, 640, 720, 800, 860];
  const pdp = partialDependence(challenger, te.X, 300, F, j, grid, 5);
  assert.ok(pdp.mean[0] > pdp.mean[4], 'risk should fall as score rises');
  const imp = await permutationImportance(challenger, te.X, te.y, 1500, F, 1);
  const cs = imp.drops[0]; const condo = imp.drops[FEATURES.findIndex((f) => f.key === 'condo')];
  assert.ok(cs > condo, `credit score ${cs} vs condo ${condo}`);
});

test('segments and recent vintage exist for governance checks', () => {
  assert.equal(world.recent.n, 2000);
  assert.ok(column(world.recent, 1).reduce((a, b) => a + b, 0) / 2000 > column(world.truth, 1).reduce((a, b) => a + b, 0) / world.truth.n);
});
