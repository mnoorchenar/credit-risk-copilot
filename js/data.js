// Synthetic residential secured lending (mortgage / HELOC) accounts with a realistic default process, deliberately dirty
// raw data, automated integrity checks and cleaning. Pure functions only (runs in the browser and in Node tests).

export const FEATURES = [
  { key: 'credit_score', label: 'Credit score', short: 'Score', dec: 0, min: 300, max: 900, sign: -1, actionable: true, desc: 'Bureau credit score' },
  { key: 'ltv', label: 'Loan-to-value', short: 'LTV', dec: 1, unit: '%', min: 5, max: 100, sign: 1, actionable: true, desc: 'Current loan-to-value ratio' },
  { key: 'tds', label: 'Debt service ratio', short: 'TDS', dec: 1, unit: '%', min: 5, max: 90, sign: 1, actionable: true, desc: 'Total debt service ratio' },
  { key: 'income', label: 'Annual income', short: 'Income', dec: 0, unit: 'k', prefix: '$', min: 20, max: 700, sign: -1, actionable: true, desc: 'Household income, CAD thousands' },
  { key: 'loan_amount', label: 'Loan amount', short: 'Loan', dec: 0, unit: 'k', prefix: '$', min: 40, max: 2000, sign: 0, actionable: false, desc: 'Outstanding balance, CAD thousands' },
  { key: 'mortgage_rate', label: 'Contract rate', short: 'Rate', dec: 2, unit: '%', min: 1, max: 10, sign: 0, actionable: false, desc: 'Current contract interest rate' },
  { key: 'months_on_book', label: 'Months on book', short: 'Age', dec: 0, min: 0, max: 150, sign: 0, actionable: false, desc: 'Account age in months' },
  { key: 'renewal_shock', label: 'Renewal payment shock', short: 'Shock', dec: 1, unit: '%', min: 0, max: 80, sign: 1, actionable: false, desc: 'Payment increase at renewal within 12 months' },
  { key: 'heloc_util', label: 'HELOC utilization', short: 'HELOC', dec: 0, unit: '%', min: 0, max: 100, sign: 1, actionable: true, desc: 'Home equity line utilization (0 = no HELOC)' },
  { key: 'prior_delinq', label: 'Delinquencies (12m)', short: 'Delinq', dec: 0, min: 0, max: 6, sign: 1, actionable: false, desc: 'Prior delinquency events in last 12 months' },
  { key: 'unemp', label: 'Local unemployment', short: 'Unemp', dec: 1, unit: '%', min: 2, max: 15, sign: 1, actionable: false, desc: 'Local unemployment rate' },
  { key: 'hpi_12m', label: 'Local house prices (12m)', short: 'HPI', dec: 1, unit: '%', min: -30, max: 30, sign: -1, actionable: false, desc: 'Local house price change, 12 months' },
  { key: 'insured', label: 'Mortgage insured', short: 'Insured', dec: 0, min: 0, max: 1, sign: 0, actionable: false, binary: true, desc: 'Default-insured mortgage (1 = yes)' },
  { key: 'self_employed', label: 'Self-employed', short: 'SelfEmp', dec: 0, min: 0, max: 1, sign: 1, actionable: false, binary: true, desc: 'Self-employed borrower (1 = yes)' },
  { key: 'condo', label: 'Condominium', short: 'Condo', dec: 0, min: 0, max: 1, sign: 0, actionable: false, binary: true, desc: 'Condominium property (1 = yes)' },
];
export const F = FEATURES.length;
export const FIDX = Object.fromEntries(FEATURES.map((f, i) => [f.key, i]));

export const REGIONS = ['Major urban', 'Suburban', 'Mid-size city', 'Rural / Northern'];
export const EMPLOYMENT = ['Salaried', 'Self-employed', 'Contract'];
export const PROPERTY = ['Detached', 'Townhouse', 'Condo', 'Multi-unit'];

export function fmtFeature(key, v) {
  const f = FEATURES[FIDX[key]];
  if (v === null || v === undefined || Number.isNaN(v)) return 'missing';
  if (f.binary) return v >= 0.5 ? 'Yes' : 'No';
  const n = Number(v).toLocaleString('en-CA', { minimumFractionDigits: f.dec, maximumFractionDigits: f.dec });
  return `${f.prefix || ''}${n}${f.unit || ''}`;
}

// ---- random numbers ---------------------------------------------------------------------------------------------------
export function rngFrom(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const gauss = () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0; let v = 0;
    while (u === 0) u = next();
    v = next();
    const m = Math.sqrt(-2 * Math.log(u));
    spare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  };
  return { next, gauss };
}
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const pick = (r, weights) => { let t = r.next(); for (let i = 0; i < weights.length; i++) { t -= weights[i]; if (t <= 0) return i; } return weights.length - 1; };
const payFactor = (ratePct, years = 25) => { const m = ratePct / 1200; return m / (1 - (1 + m) ** (-years * 12)); };

// ---- generator --------------------------------------------------------------------------------------------------------
/** Underlying "truth": default (90+ DPD within 12 months) log-odds before the intercept. Non-linear on purpose, so a
 *  linear scorecard leaves performance on the table and the challenger has something to find. */
const SIGNAL = 0.82; // overall strength of the underlying risk signal (keeps AUC in a realistic 0.78-0.85 band)
export function riskLogit(a) {
  const s = (700 - a.credit_score) / 60;
  let z = 0.95 * s + 0.35 * Math.max(0, s) ** 2 + (a.credit_score < 620 ? 0.9 : 0);
  z += 0.016 * Math.max(0, a.ltv - 60) + 0.18 * Math.max(0, a.ltv - 85);
  z += 0.010 * Math.max(0, a.tds - 36) + 0.045 * Math.max(0, a.tds - 46);
  z += 0.012 * a.renewal_shock * (a.tds > 38 ? 2.4 : 0.6);
  z += 0.004 * a.heloc_util + 0.022 * a.heloc_util * (a.ltv > 75 ? 1 : 0);
  z += (a.prior_delinq >= 1 ? 0.7 : 0) + 0.3 * Math.min(a.prior_delinq, 3);
  z += 0.10 * (a.unemp - 6);
  z += -0.025 * a.hpi_12m * (a.ltv / 70) ** 2;
  z += -0.10 * a.insured;
  z += 0.22 * a.self_employed * (a.tds > 40 ? 2 : 1);
  z += 0.08 * a.condo;
  z += -0.45 * Math.log(a.income / 100);
  z += 0.35 * (a.months_on_book < 12 ? 1 : 0);
  return z * SIGNAL;
}

function makeAccount(r, drift) {
  const region = pick(r, [0.45, 0.30, 0.17, 0.08]);
  const employment = pick(r, [0.78, 0.15, 0.07]);
  const property = pick(r, [0.42, 0.18, 0.30, 0.10]);
  const tier = [1.35, 1.1, 0.8, 0.6][region];
  const selfEmp = employment === 1 ? 1 : 0;
  const income = clip(Math.exp(Math.log(98) + 0.45 * r.gauss()) * [1.1, 1.0, 0.92, 0.85][region] * (selfEmp ? 1.05 : 1), 28, 600);
  const credit = clip(Math.round(745 + 62 * r.gauss() - (selfEmp ? 8 : 0) - drift.score), 520, 900);
  const mob = clip(Math.round(Math.abs(r.gauss()) * 40 + (r.next() < 0.35 ? r.next() * 30 : 0)), 0, 144);
  const vintageRate = mob >= 36 && mob <= 66 ? 2.6 + 0.4 * r.gauss() : mob < 36 ? 5.1 + 0.5 * r.gauss() : 3.9 + 0.6 * r.gauss();
  const rate = clip(vintageRate + drift.rate, 1.5, 8.5);
  const ltv = clip(66 + 15 * r.gauss() + drift.ltv - (mob / 144) * 8, 15, 98);
  const price = 520 * tier * [1.15, 0.85, 0.65, 1.5][property] * Math.exp(0.25 * r.gauss());
  const loan = clip(Math.round((ltv / 100) * price), 60, 1800);
  const otherDebt = Math.exp(8.3 + 0.9 * r.gauss());
  const tds = clip(((loan * 1000 * payFactor(rate) * 12 + 5400 + otherDebt) / (income * 1000)) * 100, 8, 75);
  const monthsToRenewal = 60 - (mob % 60);
  const shock = monthsToRenewal <= 12 ? Math.max(0, (payFactor(5.2) / payFactor(rate) - 1) * 100) : 0;
  const heloc = r.next() < 0.22 ? clip(Math.round(45 + 28 * r.gauss()), 3, 100) : 0;
  const lam = clip(0.1 * Math.exp(-(credit - 740) / 60), 0.01, 1.2);
  let delinq = 0;
  { const L = Math.exp(-lam); let p = 1; let k = 0; do { k++; p *= r.next(); } while (p > L && k < 5); delinq = k - 1; }
  const unemp = clip([6.8, 5.8, 6.2, 7.4][region] + drift.unemp + 0.8 * r.gauss(), 3, 12);
  const hpi = clip([-3, -1, 1.5, 3][region] + drift.hpi + 2.2 * r.gauss(), -20, 15);
  const insured = ltv > 78 && r.next() < 0.85 ? 1 : 0;
  const a = {
    credit_score: credit, ltv, tds, income, loan_amount: loan, mortgage_rate: rate, months_on_book: mob,
    renewal_shock: shock, heloc_util: heloc, prior_delinq: delinq, unemp, hpi_12m: hpi, insured,
    self_employed: selfEmp, condo: property === 2 ? 1 : 0,
  };
  return { a, region, employment, property };
}

const NO_DRIFT = { score: 0, rate: 0, ltv: 0, unemp: 0, hpi: 0 };
const RECENT_DRIFT = { score: 9, rate: 0.7, ltv: 4, unemp: 0.9, hpi: -2.2 };

/** Returns raw (dirty) data plus the truth labels. `target` is the portfolio default rate used to set the intercept. */
export function generate({ n = 30000, seed = 2026, recent = 6000, target = 0.05 } = {}) {
  const r = rngFrom(seed);
  const build = (count, drift) => {
    const X = new Float64Array(count * F);
    const logits = new Float64Array(count);
    const seg = { region: new Uint8Array(count), employment: new Uint8Array(count), property: new Uint8Array(count) };
    for (let i = 0; i < count; i++) {
      const { a, region, employment, property } = makeAccount(r, drift);
      for (let j = 0; j < F; j++) X[i * F + j] = a[FEATURES[j].key];
      logits[i] = riskLogit(a) + 1.0 * r.gauss();
      seg.region[i] = region; seg.employment[i] = employment; seg.property[i] = property;
    }
    return { X, logits, seg };
  };
  const main = build(n, NO_DRIFT);
  // intercept so that the mean probability equals the target rate
  let lo = -14; let hi = 2;
  for (let k = 0; k < 60; k++) {
    const mid = (lo + hi) / 2;
    let s = 0;
    for (let i = 0; i < n; i++) s += 1 / (1 + Math.exp(-(main.logits[i] + mid)));
    if (s / n < target) lo = mid; else hi = mid;
  }
  const b0 = (lo + hi) / 2;
  const label = (logits, count) => {
    const y = new Uint8Array(count);
    for (let i = 0; i < count; i++) y[i] = r.next() < 1 / (1 + Math.exp(-(logits[i] + b0))) ? 1 : 0;
    return y;
  };
  const y = label(main.logits, n);
  const rec = build(recent, RECENT_DRIFT);
  const yRecent = label(rec.logits, recent);
  const clean = { X: main.X, y, seg: main.seg, n };
  const dirty = injectIssues(clean, r);
  return { raw: dirty, truth: clean, recent: { X: rec.X, y: yRecent, seg: rec.seg, n: recent }, intercept: b0, seed };
}

/** Real data is messy: missing values, sentinel codes, typos and duplicated records. */
function injectIssues(clean, r) {
  const X = Float64Array.from(clean.X);
  const n = clean.n;
  const ids = new Int32Array(n);
  for (let i = 0; i < n; i++) ids[i] = 100000 + i;
  const inj = { missingIncome: 0, sentinelScore: 0, ltvTypo: 0, badTds: 0, duplicates: 0 };
  for (let i = 0; i < n; i++) {
    const t = r.next();
    if (t < 0.018) { X[i * F + FIDX.income] = NaN; inj.missingIncome++; }
    else if (t < 0.024) { X[i * F + FIDX.credit_score] = r.next() < 0.5 ? 0 : 999; inj.sentinelScore++; }
    else if (t < 0.028) { X[i * F + FIDX.ltv] *= 10; inj.ltvTypo++; }
    else if (t < 0.031) { X[i * F + FIDX.tds] = r.next() < 0.5 ? -1 : 140; inj.badTds++; }
    if (i > 0 && r.next() < 0.004) { ids[i] = ids[i - 1]; inj.duplicates++; }
  }
  return { X, y: clean.y, seg: clean.seg, n, ids, injected: inj };
}

// ---- integrity controls -------------------------------------------------------------------------------------------------
function median(arr) { const a = arr.filter((v) => !Number.isNaN(v)).sort((x, y) => x - y); return a.length ? a[a.length >> 1] : 0; }

export function integrityChecks(raw) {
  const { X, n, ids } = raw;
  const col = (j) => { const c = new Float64Array(n); for (let i = 0; i < n; i++) c[i] = X[i * F + j]; return c; };
  const out = [];
  const add = (category, name, count, action, severity = 'high') => out.push({ category, name, count, rate: count / n, action, status: count === 0 ? 'pass' : severity === 'high' && count / n > 0.01 ? 'fail' : 'warn' });
  let missingCells = 0; let missingRows = 0;
  for (let i = 0; i < n; i++) { let m = 0; for (let j = 0; j < F; j++) if (Number.isNaN(X[i * F + j])) { m++; missingCells++; } if (m) missingRows++; }
  add('Completeness', 'Missing values', missingRows, 'Median-impute from training data and log the fix', 'high');
  for (const f of FEATURES) {
    if (f.binary) continue;
    const c = col(FIDX[f.key]);
    let bad = 0;
    for (let i = 0; i < n; i++) if (!Number.isNaN(c[i]) && (c[i] < f.min || c[i] > f.max)) bad++;
    if (['credit_score', 'ltv', 'tds'].includes(f.key)) add('Validity', `${f.label} outside valid range (${f.min}-${f.max})`, bad, 'Treat as missing, then impute', 'high');
  }
  let dup = 0; const seen = new Set();
  for (let i = 0; i < n; i++) { if (seen.has(ids[i])) dup++; else seen.add(ids[i]); }
  add('Uniqueness', 'Duplicate account IDs', dup, 'Keep first record per account', 'warn');
  let incons = 0;
  for (let i = 0; i < n; i++) { const ltv = X[i * F + FIDX.ltv]; const ins = X[i * F + FIDX.insured]; if (ins === 1 && ltv < 55) incons++; }
  add('Consistency', 'Insured mortgage with LTV below 55%', incons, 'Review with origination team', 'warn');
  let ext = 0;
  for (let i = 0; i < n; i++) { const inc = X[i * F + FIDX.income]; const loan = X[i * F + FIDX.loan_amount]; if (!Number.isNaN(inc) && loan / inc > 10) ext++; }
  add('Plausibility', 'Loan more than 10x income', ext, 'Flag for analyst review (kept)', 'warn');
  return out;
}

/** Cleans in place on a copy: invalid -> missing, drop duplicate IDs, impute medians learned on the supplied training rows. */
export function cleanData(raw, trainIdx = null) {
  const { X, y, seg, n, ids } = raw;
  const keep = []; const seen = new Set();
  for (let i = 0; i < n; i++) { if (seen.has(ids[i])) continue; seen.add(ids[i]); keep.push(i); }
  const m = keep.length;
  const out = new Float64Array(m * F);
  const fixes = { imputed: 0, invalid: 0, droppedDuplicates: n - m };
  for (let k = 0; k < m; k++) {
    for (let j = 0; j < F; j++) {
      let v = X[keep[k] * F + j];
      const f = FEATURES[j];
      if (!Number.isNaN(v) && !f.binary && (v < f.min || v > f.max) && ['credit_score', 'ltv', 'tds'].includes(f.key)) { v = NaN; fixes.invalid++; }
      out[k * F + j] = v;
    }
  }
  const base = trainIdx ? trainIdx.filter((i) => i < m) : [...Array(m).keys()];
  const meds = new Float64Array(F);
  for (let j = 0; j < F; j++) meds[j] = median(base.map((i) => out[i * F + j]));
  for (let k = 0; k < m; k++) for (let j = 0; j < F; j++) if (Number.isNaN(out[k * F + j])) { out[k * F + j] = meds[j]; fixes.imputed++; }
  const yy = new Uint8Array(m); const s = { region: new Uint8Array(m), employment: new Uint8Array(m), property: new Uint8Array(m) };
  keep.forEach((i, k) => { yy[k] = y[i]; s.region[k] = seg.region[i]; s.employment[k] = seg.employment[i]; s.property[k] = seg.property[i]; });
  return { X: out, y: yy, seg: s, n: m, fixes, medians: meds };
}

/** Deterministic 60/20/20 split, stratified on the outcome when `y` is given so every set has the same default rate. */
export function split(n, seed = 7, y = null) {
  const r = rngFrom(seed);
  const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const groups = y ? [[], []] : [[]];
  for (let i = 0; i < n; i++) groups[y ? y[i] : 0].push(i);
  const out = { train: [], val: [], test: [] };
  for (const g of groups) {
    shuffle(g);
    const a = Math.floor(g.length * 0.6); const b = Math.floor(g.length * 0.8);
    out.train.push(...g.slice(0, a)); out.val.push(...g.slice(a, b)); out.test.push(...g.slice(b));
  }
  for (const k of Object.keys(out)) shuffle(out[k]);
  return out;
}

export function take(data, idx) {
  const m = idx.length;
  const X = new Float64Array(m * F); const y = new Uint8Array(m);
  const seg = { region: new Uint8Array(m), employment: new Uint8Array(m), property: new Uint8Array(m) };
  for (let k = 0; k < m; k++) {
    const i = idx[k];
    for (let j = 0; j < F; j++) X[k * F + j] = data.X[i * F + j];
    y[k] = data.y[i];
    if (data.seg) { seg.region[k] = data.seg.region[i]; seg.employment[k] = data.seg.employment[i]; seg.property[k] = data.seg.property[i]; }
  }
  return { X, y, seg, n: m };
}

export function column(d, j) { const c = new Float64Array(d.n); for (let i = 0; i < d.n; i++) c[i] = d.X[i * F + j]; return c; }

/** Histogram of one feature split by outcome, plus default rate per bin. */
export function featureProfile(d, j, bins = 20) {
  const c = column(d, j);
  let lo = Infinity; let hi = -Infinity;
  for (const v of c) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const f = FEATURES[j];
  if (f.binary) { lo = 0; hi = 1; bins = 2; }
  else { const sorted = Float64Array.from(c).sort(); lo = sorted[Math.floor(sorted.length * 0.005)]; hi = sorted[Math.floor(sorted.length * 0.995)]; if (hi === lo) hi = lo + 1; }
  const good = new Array(bins).fill(0); const bad = new Array(bins).fill(0);
  const w = (hi - lo) / bins;
  for (let i = 0; i < d.n; i++) {
    let b = f.binary ? (c[i] >= 0.5 ? 1 : 0) : Math.floor((c[i] - lo) / w);
    b = clip(b, 0, bins - 1);
    if (d.y[i]) bad[b]++; else good[b]++;
  }
  const centers = good.map((_, b) => (f.binary ? b : lo + (b + 0.5) * w));
  const rate = good.map((g, b) => (g + bad[b] >= 30 ? (bad[b] / (g + bad[b])) * 100 : null));
  return { centers, good, bad, rate, lo, hi };
}
