// Machine learning core, written from scratch so every number on screen can be traced: metrics, a regularised logistic
// "scorecard", histogram gradient boosting, exact TreeSHAP, partial dependence, permutation importance, PSI.
// Matrices are flat row-major Float64Arrays (n rows x F columns). No DOM access: runs in the browser and in Node tests.

const tick = () => new Promise((r) => setTimeout(r, 0));
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export const logit = (p) => Math.log(Math.max(1e-9, p) / Math.max(1e-9, 1 - p));

// ---- metrics ----------------------------------------------------------------------------------------------------------
export function auc(y, s) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[a] - s[b]);
  let rankSum = 0; let pos = 0;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && s[idx[j + 1]] === s[idx[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) if (y[idx[k]]) { rankSum += avg; pos++; }
    i = j + 1;
  }
  const neg = n - pos;
  return pos === 0 || neg === 0 ? 0.5 : (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

export function ks(y, s) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[b] - s[a]);
  let pos = 0; for (let i = 0; i < n; i++) pos += y[i];
  const neg = n - pos;
  let cp = 0; let cn = 0; let best = 0;
  for (let k = 0; k < n; k++) { if (y[idx[k]]) cp++; else cn++; best = Math.max(best, Math.abs(cp / pos - cn / neg)); }
  return best;
}

export function logloss(y, p) {
  let s = 0;
  for (let i = 0; i < y.length; i++) { const q = Math.min(1 - 1e-9, Math.max(1e-9, p[i])); s -= y[i] ? Math.log(q) : Math.log(1 - q); }
  return s / y.length;
}
export function brier(y, p) { let s = 0; for (let i = 0; i < y.length; i++) s += (p[i] - y[i]) ** 2; return s / y.length; }

export function rocPoints(y, s, maxPts = 150) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[b] - s[a]);
  let pos = 0; for (let i = 0; i < n; i++) pos += y[i];
  const neg = n - pos;
  const pts = [{ x: 0, y: 0 }];
  let tp = 0; let fp = 0;
  const step = Math.max(1, Math.floor(n / maxPts));
  for (let k = 0; k < n; k++) { if (y[idx[k]]) tp++; else fp++; if (k % step === 0 || k === n - 1) pts.push({ x: fp / neg, y: tp / pos }); }
  return pts;
}

/** Equal-count calibration bins. */
export function calibration(y, p, bins = 10) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
  const out = []; let ece = 0;
  for (let b = 0; b < bins; b++) {
    const from = Math.floor((b * n) / bins); const to = Math.floor(((b + 1) * n) / bins);
    let sp = 0; let sy = 0;
    for (let k = from; k < to; k++) { sp += p[idx[k]]; sy += y[idx[k]]; }
    const m = to - from;
    if (!m) continue;
    out.push({ pred: sp / m, obs: sy / m, n: m });
    ece += (m / n) * Math.abs(sp / m - sy / m);
  }
  return { bins: out, ece };
}

/** Slope of outcome on logit(p): ~1 means well calibrated, <1 means over-confident. */
export function calibrationSlope(y, p) {
  const n = y.length; let a = 0; let b = 1;
  const z = new Float64Array(n); for (let i = 0; i < n; i++) z[i] = logit(p[i]);
  for (let it = 0; it < 25; it++) {
    let g0 = 0; let g1 = 0; let h00 = 0; let h01 = 0; let h11 = 0;
    for (let i = 0; i < n; i++) {
      const q = sigmoid(a + b * z[i]); const w = q * (1 - q);
      g0 += y[i] - q; g1 += (y[i] - q) * z[i]; h00 += w; h01 += w * z[i]; h11 += w * z[i] * z[i];
    }
    const det = h00 * h11 - h01 * h01;
    if (Math.abs(det) < 1e-12) break;
    a += (h11 * g0 - h01 * g1) / det; b += (-h01 * g0 + h00 * g1) / det;
  }
  return b;
}

/** Highest-risk first. Returns deciles with count, observed bad rate and lift. */
export function deciles(y, p) {
  const n = y.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[b] - p[a]);
  let tot = 0; for (let i = 0; i < n; i++) tot += y[i];
  const base = tot / n; const out = []; let cum = 0;
  for (let d = 0; d < 10; d++) {
    const from = Math.floor((d * n) / 10); const to = Math.floor(((d + 1) * n) / 10);
    let bad = 0; let sp = 0;
    for (let k = from; k < to; k++) { bad += y[idx[k]]; sp += p[idx[k]]; }
    cum += bad;
    out.push({ decile: d + 1, n: to - from, bad, rate: bad / (to - from), avgPd: sp / (to - from), lift: bad / (to - from) / base, cumCapture: cum / tot });
  }
  return out;
}

/** Population stability index of `actual` against the quantile bins of `expected`. */
export function psi(expected, actual, bins = 10) {
  const e = Float64Array.from(expected).sort();
  const edges = [];
  for (let b = 1; b < bins; b++) edges.push(e[Math.floor((b * e.length) / bins)]);
  const bucket = (v) => { let k = 0; while (k < edges.length && v > edges[k]) k++; return k; };
  const ce = new Array(bins).fill(0); const ca = new Array(bins).fill(0);
  for (const v of expected) ce[bucket(v)]++;
  for (const v of actual) ca[bucket(v)]++;
  let s = 0;
  for (let b = 0; b < bins; b++) { const pe = Math.max(ce[b] / expected.length, 1e-4); const pa = Math.max(ca[b] / actual.length, 1e-4); s += (pa - pe) * Math.log(pa / pe); }
  return s;
}

export function summarize(y, p) {
  const a = auc(y, p);
  return { auc: a, gini: 2 * a - 1, ks: ks(y, p), logloss: logloss(y, p), brier: brier(y, p), rate: y.reduce((s, v) => s + v, 0) / y.length, meanPd: p.reduce((s, v) => s + v, 0) / p.length };
}

// ---- generic prediction -----------------------------------------------------------------------------------------------
export function marginOf(model, X, n, F) {
  const out = new Float64Array(n);
  if (model.type === 'logistic') {
    for (let i = 0; i < n; i++) { let z = model.b; for (let j = 0; j < F; j++) z += model.w[j] * ((X[i * F + j] - model.mean[j]) / model.std[j]); out[i] = z; }
  } else {
    for (let i = 0; i < n; i++) out[i] = gbmMargin(model, X, i * F);
  }
  return out;
}
export function predictProba(model, X, n, F) {
  const m = marginOf(model, X, n, F);
  for (let i = 0; i < n; i++) m[i] = sigmoid(m[i]);
  return m;
}

// ---- logistic regression (champion scorecard) -------------------------------------------------------------------------
export async function trainLogistic(Xtr, ytr, n, F, params, val, onProgress) {
  const { epochs = 250, lr = 0.05, l2 = 0.01, posWeight = 1 } = params;
  const mean = new Float64Array(F); const std = new Float64Array(F);
  for (let j = 0; j < F; j++) { let s = 0; for (let i = 0; i < n; i++) s += Xtr[i * F + j]; mean[j] = s / n; }
  for (let j = 0; j < F; j++) { let s = 0; for (let i = 0; i < n; i++) s += (Xtr[i * F + j] - mean[j]) ** 2; std[j] = Math.sqrt(s / n) || 1; }
  const Z = new Float64Array(n * F);
  for (let i = 0; i < n; i++) for (let j = 0; j < F; j++) Z[i * F + j] = (Xtr[i * F + j] - mean[j]) / std[j];
  const w = new Float64Array(F); let b = Math.log(Math.max(1e-6, ytr.reduce((s, v) => s + v, 0) / n) / (1 - ytr.reduce((s, v) => s + v, 0) / n));
  const mw = new Float64Array(F); const vw = new Float64Array(F); let mb = 0; let vb = 0;
  const model = { type: 'logistic', mean, std, w, b, F };
  const history = [];
  const cadence = Math.max(1, Math.ceil(epochs / 60));
  const p = new Float64Array(n);
  for (let ep = 1; ep <= epochs; ep++) {
    const gw = new Float64Array(F); let gb = 0;
    for (let i = 0; i < n; i++) {
      let z = b; for (let j = 0; j < F; j++) z += w[j] * Z[i * F + j];
      p[i] = sigmoid(z);
      const e = (p[i] - ytr[i]) * (ytr[i] ? posWeight : 1);
      gb += e; for (let j = 0; j < F; j++) gw[j] += e * Z[i * F + j];
    }
    for (let j = 0; j < F; j++) {
      const g = gw[j] / n + l2 * w[j];
      mw[j] = 0.9 * mw[j] + 0.1 * g; vw[j] = 0.999 * vw[j] + 0.001 * g * g;
      w[j] -= (lr * (mw[j] / (1 - 0.9 ** ep))) / (Math.sqrt(vw[j] / (1 - 0.999 ** ep)) + 1e-8);
    }
    const g0 = gb / n; mb = 0.9 * mb + 0.1 * g0; vb = 0.999 * vb + 0.001 * g0 * g0;
    b -= (lr * (mb / (1 - 0.9 ** ep))) / (Math.sqrt(vb / (1 - 0.999 ** ep)) + 1e-8);
    model.b = b;
    if (ep % cadence === 0 || ep === epochs) {
      const pt = predictProba(model, Xtr, n, F); const pv = predictProba(model, val.X, val.n, F);
      const point = { iter: ep, total: epochs, train: { loss: logloss(ytr, pt), auc: auc(ytr, pt) }, val: { loss: logloss(val.y, pv), auc: auc(val.y, pv) } };
      history.push(point);
      if (onProgress) { onProgress(point, history); await tick(); }
    }
  }
  model.b = b;
  return { model, history, bestIter: epochs };
}

// ---- gradient boosting (challenger) -----------------------------------------------------------------------------------
function makeBins(Xtr, n, F, maxBins) {
  const edges = []; const binned = [];
  for (let j = 0; j < F; j++) {
    const c = new Float64Array(n); for (let i = 0; i < n; i++) c[i] = Xtr[i * F + j];
    const s = Float64Array.from(c).sort();
    const e = [];
    for (let b = 1; b < maxBins; b++) { const v = s[Math.floor((b * n) / maxBins)]; if (!e.length || v > e[e.length - 1]) e.push(v); }
    // drop an edge equal to the max (it would create an empty right bin)
    while (e.length && e[e.length - 1] >= s[n - 1]) e.pop();
    edges.push(Float64Array.from(e));
    const bj = new Uint8Array(n);
    for (let i = 0; i < n; i++) { const v = c[i]; let lo = 0; let hi = e.length; while (lo < hi) { const m = (lo + hi) >> 1; if (v <= e[m]) hi = m; else lo = m + 1; } bj[i] = lo; }
    binned.push(bj);
  }
  return { edges, binned };
}

function rngSimple(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function growTree(binned, edges, g, h, rows, F, params, nb) {
  const { maxDepth, minChildWeight, lambda, gamma = 0, lr } = params;
  const feat = []; const thr = []; const bin = []; const left = []; const right = []; const val = [];
  const gsum = new Float64Array(nb); const hsum = new Float64Array(nb);
  function build(rs, depth) {
    const id = feat.length;
    feat.push(-1); thr.push(0); bin.push(0); left.push(-1); right.push(-1); val.push(0);
    let G = 0; let H = 0;
    for (let k = 0; k < rs.length; k++) { G += g[rs[k]]; H += h[rs[k]]; }
    val[id] = (-G / (H + lambda)) * lr;
    if (depth >= maxDepth || H < 2 * minChildWeight || rs.length < 4) return id;
    let bestGain = 1e-7; let bf = -1; let bb = -1;
    const parent = (G * G) / (H + lambda);
    for (let f = 0; f < F; f++) {
      const nbf = edges[f].length + 1;
      if (nbf < 2) continue;
      gsum.fill(0, 0, nbf); hsum.fill(0, 0, nbf);
      const col = binned[f];
      for (let k = 0; k < rs.length; k++) { const r = rs[k]; const b = col[r]; gsum[b] += g[r]; hsum[b] += h[r]; }
      let GL = 0; let HL = 0;
      for (let b = 0; b < nbf - 1; b++) {
        GL += gsum[b]; HL += hsum[b];
        const HR = H - HL;
        if (HL < minChildWeight || HR < minChildWeight) continue;
        const GR = G - GL;
        const gain = 0.5 * ((GL * GL) / (HL + lambda) + (GR * GR) / (HR + lambda) - parent) - gamma;
        if (gain > bestGain) { bestGain = gain; bf = f; bb = b; }
      }
    }
    if (bf < 0) return id;
    const col = binned[bf];
    const l = []; const r2 = [];
    for (let k = 0; k < rs.length; k++) (col[rs[k]] <= bb ? l : r2).push(rs[k]);
    feat[id] = bf; bin[id] = bb; thr[id] = edges[bf][bb];
    left[id] = build(l, depth + 1);
    right[id] = build(r2, depth + 1);
    return id;
  }
  build(rows, 0);
  return { feat: Int16Array.from(feat), thr: Float64Array.from(thr), bin: Int16Array.from(bin), left: Int32Array.from(left), right: Int32Array.from(right), val: Float64Array.from(val), cover: new Float64Array(feat.length) };
}

function gbmMargin(model, X, off) {
  let s = model.init;
  const trees = model.trees;
  for (let t = 0; t < trees.length; t++) {
    const tr = trees[t]; let node = 0;
    while (tr.feat[node] >= 0) node = X[off + tr.feat[node]] <= tr.thr[node] ? tr.left[node] : tr.right[node];
    s += tr.val[node];
  }
  return s;
}

export async function trainGBM(Xtr, ytr, n, F, params, val, onProgress) {
  const p = { trees: 150, lr: 0.08, maxDepth: 3, minChildWeight: 20, lambda: 5, subsample: 0.8, posWeight: 1, patience: 25, maxBins: 32, seed: 11, ...params };
  const { edges, binned } = makeBins(Xtr, n, F, p.maxBins);
  const nb = p.maxBins + 1;
  let pos = 0; for (let i = 0; i < n; i++) pos += ytr[i];
  const init = Math.log((pos / n) / (1 - pos / n));
  const model = { type: 'gbm', F, init, trees: [], lr: p.lr };
  const Ftr = new Float64Array(n).fill(init);
  const Fval = new Float64Array(val.n).fill(init);
  const g = new Float64Array(n); const h = new Float64Array(n);
  const rand = rngSimple(p.seed);
  const history = []; let bestLoss = Infinity; let bestIter = 0;
  const cadence = Math.max(1, Math.ceil(p.trees / 60));
  const pv = new Float64Array(val.n); const pt = new Float64Array(n);
  for (let t = 1; t <= p.trees; t++) {
    for (let i = 0; i < n; i++) {
      const q = sigmoid(Ftr[i]); const w = ytr[i] ? p.posWeight : 1;
      g[i] = (q - ytr[i]) * w; h[i] = Math.max(q * (1 - q), 1e-6) * w;
    }
    const rows = [];
    for (let i = 0; i < n; i++) if (p.subsample >= 1 || rand() < p.subsample) rows.push(i);
    const tree = growTree(binned, edges, g, h, Int32Array.from(rows), F, { maxDepth: p.maxDepth, minChildWeight: p.minChildWeight, lambda: p.lambda, lr: p.lr }, nb);
    // route every training row: update predictions and node cover (row counts) used by TreeSHAP
    for (let i = 0; i < n; i++) {
      let node = 0; tree.cover[0]++;
      while (tree.feat[node] >= 0) { const f = tree.feat[node]; node = binned[f][i] <= tree.bin[node] ? tree.left[node] : tree.right[node]; tree.cover[node]++; }
      Ftr[i] += tree.val[node];
    }
    model.trees.push(tree);
    for (let i = 0; i < val.n; i++) {
      let node = 0; const off = i * F;
      while (tree.feat[node] >= 0) node = val.X[off + tree.feat[node]] <= tree.thr[node] ? tree.left[node] : tree.right[node];
      Fval[i] += tree.val[node];
    }
    for (let i = 0; i < val.n; i++) pv[i] = sigmoid(Fval[i]);
    const vloss = logloss(val.y, pv);
    if (vloss < bestLoss - 1e-6) { bestLoss = vloss; bestIter = t; }
    if (t % cadence === 0 || t === p.trees) {
      for (let i = 0; i < n; i++) pt[i] = sigmoid(Ftr[i]);
      const point = { iter: t, total: p.trees, train: { loss: logloss(ytr, pt), auc: auc(ytr, pt) }, val: { loss: vloss, auc: auc(val.y, pv) } };
      history.push(point);
      if (onProgress) { onProgress(point, history); await tick(); }
    }
    if (p.patience > 0 && t - bestIter >= p.patience) { history.stopped = t; break; }
  }
  if (p.patience > 0 && bestIter > 0 && bestIter < model.trees.length) model.trees.length = bestIter;
  model.bestIter = model.trees.length;
  return { model, history, bestIter: model.trees.length };
}

// ---- TreeSHAP (exact, path-dependent; Lundberg et al. 2018) ------------------------------------------------------------
function treeExpectation(tree, node = 0) {
  if (tree.feat[node] < 0) return tree.val[node];
  const l = tree.left[node]; const r = tree.right[node];
  return (tree.cover[l] * treeExpectation(tree, l) + tree.cover[r] * treeExpectation(tree, r)) / tree.cover[node];
}
export function gbmBase(model) {
  if (model._base === undefined) model._base = model.init + model.trees.reduce((s, t) => s + treeExpectation(t), 0);
  return model._base;
}

function extendPath(d, z, o, w, ud, pz, po, pi) {
  d[ud] = pi; z[ud] = pz; o[ud] = po; w[ud] = ud === 0 ? 1 : 0;
  for (let i = ud - 1; i >= 0; i--) { w[i + 1] += (po * w[i] * (i + 1)) / (ud + 1); w[i] = (pz * w[i] * (ud - i)) / (ud + 1); }
}
function unwindPath(d, z, o, w, ud, pi) {
  const po = o[pi]; const pz = z[pi]; let nx = w[ud];
  for (let i = ud - 1; i >= 0; i--) {
    if (po !== 0) { const t = w[i]; w[i] = (nx * (ud + 1)) / ((i + 1) * po); nx = t - (w[i] * pz * (ud - i)) / (ud + 1); }
    else w[i] = (w[i] * (ud + 1)) / (pz * (ud - i));
  }
  for (let i = pi; i < ud; i++) { d[i] = d[i + 1]; z[i] = z[i + 1]; o[i] = o[i + 1]; }
}
function unwoundSum(d, z, o, w, ud, pi) {
  const po = o[pi]; const pz = z[pi]; let nx = w[ud]; let total = 0;
  for (let i = ud - 1; i >= 0; i--) {
    if (po !== 0) { const t = (nx * (ud + 1)) / ((i + 1) * po); total += t; nx = w[i] - (t * pz * (ud - i)) / (ud + 1); }
    else total += w[i] / pz / ((ud - i) / (ud + 1));
  }
  return total;
}
function shapTree(tree, X, off, phi) {
  function rec(node, ud, pd, pz, po, pw, fz, fo, fi) {
    const d = pd.slice(); const z = pz.slice(); const o = po.slice(); const w = pw.slice();
    extendPath(d, z, o, w, ud, fz, fo, fi);
    if (tree.feat[node] < 0) {
      for (let i = 1; i <= ud; i++) phi[d[i]] += unwoundSum(d, z, o, w, ud, i) * (o[i] - z[i]) * tree.val[node];
      return;
    }
    const f = tree.feat[node];
    const hot = X[off + f] <= tree.thr[node] ? tree.left[node] : tree.right[node];
    const cold = hot === tree.left[node] ? tree.right[node] : tree.left[node];
    let iz = 1; let io = 1; let k = 0; let nud = ud;
    for (; k <= ud; k++) if (d[k] === f) break;
    if (k <= ud) { iz = z[k]; io = o[k]; unwindPath(d, z, o, w, ud, k); nud = ud - 1; }
    rec(hot, nud + 1, d, z, o, w, (iz * tree.cover[hot]) / tree.cover[node], io, f);
    rec(cold, nud + 1, d, z, o, w, (iz * tree.cover[cold]) / tree.cover[node], 0, f);
  }
  rec(0, 0, [], [], [], [], 1, 1, -1);
}

/** SHAP values in log-odds for one row. base + sum(phi) === model margin. */
export function shapRow(model, X, off, F) {
  const phi = new Float64Array(F);
  if (model.type === 'logistic') {
    for (let j = 0; j < F; j++) phi[j] = model.w[j] * ((X[off + j] - model.mean[j]) / model.std[j]);
    return { phi, base: model.b };
  }
  for (const t of model.trees) shapTree(t, X, off, phi);
  return { phi, base: gbmBase(model) };
}

export async function shapMatrix(model, X, n, F, onProgress) {
  const out = new Float64Array(n * F); let base = 0;
  for (let i = 0; i < n; i++) {
    const r = shapRow(model, X, i * F, F); base = r.base;
    out.set(r.phi, i * F);
    if (onProgress && i % 25 === 0) { onProgress(i / n); await tick(); }
  }
  return { values: out, base };
}

// ---- model-agnostic explainers --------------------------------------------------------------------------------------------
export function partialDependence(model, X, n, F, j, grid, iceRows = 30) {
  const mean = []; const ice = Array.from({ length: Math.min(iceRows, n) }, () => []);
  const row = new Float64Array(F);
  for (const g of grid) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < F; k++) row[k] = X[i * F + k];
      row[j] = g;
      const p = sigmoid(marginOf(model, row, 1, F)[0]);
      s += p; if (i < ice.length) ice[i].push(p);
    }
    mean.push(s / n);
  }
  return { grid, mean, ice };
}

export async function permutationImportance(model, X, y, n, F, repeats = 3, seed = 5, onProgress) {
  const base = auc(y, predictProba(model, X, n, F));
  const rand = rngSimple(seed);
  const out = [];
  for (let j = 0; j < F; j++) {
    let drop = 0;
    for (let r = 0; r < repeats; r++) {
      const Xp = Float64Array.from(X);
      for (let i = n - 1; i > 0; i--) { const k = Math.floor(rand() * (i + 1)); const a = Xp[i * F + j]; Xp[i * F + j] = Xp[k * F + j]; Xp[k * F + j] = a; }
      drop += base - auc(y, predictProba(model, Xp, n, F));
    }
    out.push(drop / repeats);
    if (onProgress) { onProgress((j + 1) / F); await tick(); }
  }
  return { base, drops: out };
}

export { tick };
