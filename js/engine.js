// Deterministic analytics engine. Both the rules parser and the LLM produce a small structured query; ONLY this file turns
// a query into numbers. The language model never computes or invents a figure. Runs in the browser and in Node tests.

export const SCENARIO_NONE = { rate_bps: 0, unemp_pp: 0, hpi_pct: 0 };
export const SCENARIO_SEVERE = { rate_bps: 150, unemp_pp: 2, hpi_pct: -15 };

// key -> label, unit, decimals, worseHigh (is a higher value worse?), blurb used in prompts and help
export const METRICS = {
  risk: { label: 'Credit risk score', unit: '', dec: 1, worseHigh: true, blurb: 'composite 0-100 score, higher is riskier' },
  dpd90: { label: '90+ day delinquency', unit: '%', dec: 2, worseHigh: true, blurb: 'share of balances 90+ days past due' },
  dpd_trend: { label: '6-month change in delinquency', unit: ' bps', dec: 1, worseHigh: true, blurb: 'change in 90+ delinquency over 6 months' },
  ltv: { label: 'Average LTV', unit: '%', dec: 1, worseHigh: true, blurb: 'average current loan-to-value' },
  dti: { label: 'Average debt service ratio', unit: '%', dec: 1, worseHigh: true, blurb: 'average total debt service ratio' },
  renew: { label: 'Balance renewing in 12 months', unit: '%', dec: 1, worseHigh: true, blurb: 'payment-shock exposure at renewal' },
  unemp: { label: 'Unemployment rate', unit: '%', dec: 1, worseHigh: true, blurb: 'local unemployment' },
  hpi: { label: 'House price change (12m)', unit: '%', dec: 1, worseHigh: false, blurb: 'year over year house price change' },
  score: { label: 'Average credit score', unit: '', dec: 0, worseHigh: false, blurb: 'average bureau score' },
  override: { label: 'Adjudication exception rate', unit: '%', dec: 1, worseHigh: true, blurb: 'share of approvals that were policy exceptions or overrides' },
  dq: { label: 'Data integrity pass rate', unit: '%', dec: 1, worseHigh: false, blurb: 'share of files passing automated data checks' },
  balance: { label: 'Portfolio balance', unit: ' $M', dec: 0, worseHigh: true, blurb: 'outstanding balance in CAD millions' },
  heloc: { label: 'HELOC share', unit: '%', dec: 1, worseHigh: true, blurb: 'share of balance in home equity lines' },
};

export const GROUPS = {
  gta: { label: 'the GTA', ids: [3895, 2253, 2270, 2230, 2236] },
  north: { label: 'Northern Ontario', ids: [2226, 2247, 2249, 2261, 2262, 7654] },
  southwest: { label: 'Southwestern Ontario', ids: [2244, 2242, 2240, 2268, 4913, 5183, 2265, 2266, 7652] },
};

const lc = (t) => String(t).replace(/[A-Za-z]+/g, (w) => (/^[A-Z]{2,}$/.test(w) ? w : w.toLowerCase()));
const clip = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
const fmtNum = (v, d) => Number(v).toLocaleString('en-CA', { minimumFractionDigits: d, maximumFractionDigits: d });

export function fmt(key, v) {
  const m = METRICS[key];
  if (v === null || v === undefined || Number.isNaN(v)) return 'n/a';
  if (key === 'dpd_trend') return `${v > 0 ? '+' : ''}${fmtNum(v, 1)} bps`;
  if (key === 'balance') return v >= 1000 ? `$${fmtNum(v / 1000, 1)}B` : `$${fmtNum(v, 0)}M`;
  return `${fmtNum(v, m.dec)}${m.unit}`;
}

// ---- risk model (must match scripts/build_data.py) ----------------------------------------------------------------
export function riskScore(v, drivers) {
  let total = 0;
  for (const d of drivers) {
    const t = clip((v[d.key] - d.lo) / (d.hi - d.lo));
    total += d.weight * (d.worse_high ? t : 1 - t);
  }
  return Math.round(total * 1000) / 10;
}

export function contributions(v, drivers) {
  return drivers
    .map((d) => {
      const t = clip((v[d.key] - d.lo) / (d.hi - d.lo));
      return { key: d.key, points: Math.round(d.weight * (d.worse_high ? t : 1 - t) * 1000) / 10 };
    })
    .sort((a, b) => b.points - a.points);
}

export function tier(score) {
  if (score >= 58) return { name: 'High', cls: 't4' };
  if (score >= 47) return { name: 'Elevated', cls: 't3' };
  if (score >= 35) return { name: 'Moderate', cls: 't2' };
  return { name: 'Low', cls: 't1' };
}

export const scenarioActive = (s) => !!s && (s.rate_bps !== 0 || s.unemp_pp !== 0 || s.hpi_pct !== 0);

/** Illustrative stress: elasticities are simple and documented in the UI, not calibrated to any real portfolio. */
export function stressRegion(r, s, drivers) {
  if (!scenarioActive(s)) return r;
  const dDti = r.dti * 0.09 * (s.rate_bps / 100) * (r.renew / 100);
  const ltv = r.ltv / Math.max(0.5, 1 + s.hpi_pct / 100);
  const dpd90 = Math.max(
    0.01,
    r.dpd90 * (1 + 0.15 * s.unemp_pp) * (1 + 0.04 * dDti) * (1 + 0.012 * (ltv - r.ltv)),
  );
  const out = {
    ...r,
    dti: r.dti + dDti,
    ltv,
    unemp: r.unemp + s.unemp_pp,
    hpi: r.hpi + s.hpi_pct,
    dpd90,
  };
  out.risk = riskScore(out, drivers);
  return out;
}

// ---- dataset wrapper ------------------------------------------------------------------------------------------------
export function createModel(portfolio, cities = []) {
  const drivers = portfolio.meta.drivers;
  const base = portfolio.regions.map((r) => ({ ...r }));
  const byId = new Map(base.map((r) => [r.id, r]));
  const nameIndex = [];
  for (const r of base) nameIndex.push([r.short.toLowerCase(), r.id]);
  const aliases = {
    'greater toronto': 'gta', 'greater toronto area': 'gta', gta: 'gta',
    'northern ontario': 'north', 'the north': 'north', 'north ontario': 'north',
    'southwestern ontario': 'southwest', 'southwest ontario': 'southwest', 'south west': 'southwest',
  };
  const extra = {
    kitchener: 2265, cambridge: 2265, mississauga: 2253, brampton: 2253, 'niagara falls': 2246, 'st. catharines': 2246,
    guelph: 2266, oshawa: 2230, 'thunder bay': 2262, windsor: 2268, 'sault ste. marie': 2226, kingston: 7655,
  };
  for (const [k, v] of Object.entries(extra)) nameIndex.push([k, v]);
  for (const c of cities) nameIndex.push([c.name.toLowerCase(), c.phu]);
  nameIndex.sort((a, b) => b[0].length - a[0].length);

  function find(text) {
    const t = ` ${String(text).toLowerCase().replace(/[^a-z0-9.\- ]/g, ' ')} `;
    const regions = [];
    let group = null;
    for (const [alias, g] of Object.entries(aliases)) if (t.includes(` ${alias} `)) group = g;
    for (const [name, id] of nameIndex) {
      if (!byId.has(id)) continue;
      const i = t.indexOf(` ${name} `);
      const j = t.indexOf(` ${name}s `);
      if ((i >= 0 || j >= 0) && !regions.some((x) => x.id === id)) regions.push({ id, at: i >= 0 ? i : j });
    }
    regions.sort((a, b) => a.at - b.at);
    return { regions: regions.map((x) => x.id), group };
  }

  function view(scenario) {
    const rows = base.map((r) => {
      const s = stressRegion(r, scenario, drivers);
      const dpdTrend = (s.dpd90 - r.dpd90_hist[r.dpd90_hist.length - 7]) * 100;
      return { ...s, dpd_trend: dpdTrend, tier: tier(s.risk) };
    });
    return rows;
  }

  const weighted = (rows, key) => rows.reduce((a, r) => a + r[key] * r.balance_m, 0) / rows.reduce((a, r) => a + r.balance_m, 0);

  return { portfolio, drivers, base, byId, find, view, weighted, months: portfolio.meta.months };
}

// ---- alerts / watchlist ------------------------------------------------------------------------------------------------
export function alertsFor(r, rows, model) {
  const avg = (k) => model.weighted(rows, k);
  const out = [];
  if (r.dpd_trend >= 3) out.push({ sev: 3, title: 'Delinquency is rising', detail: `90+ day delinquency is up ${fmt('dpd_trend', r.dpd_trend)} in six months.`, action: 'Tighten early-warning review and sample recent originations for this region.' });
  if (r.renew >= 33) out.push({ sev: 2, title: 'Payment shock at renewal', detail: `${fmt('renew', r.renew)} of balance reprices within 12 months (province ${fmt('renew', avg('renew'))}).`, action: 'Prioritise renewal outreach and payment-shock testing for this book.' });
  if (r.ltv >= 66 && r.hpi < -2.5) out.push({ sev: 3, title: 'High LTV with falling prices', detail: `Average LTV ${fmt('ltv', r.ltv)} while prices are ${fmt('hpi', r.hpi)} year over year.`, action: 'Review LTV caps and appraisal controls for new originations.' });
  if (r.override >= 7) out.push({ sev: 3, title: 'Adjudication exceptions above norm', detail: `Exception rate ${fmt('override', r.override)} versus ${fmt('override', avg('override'))} province-wide.`, action: 'Audit overrides and recalibrate adjudicator guidance with an automated exception control.' });
  if (r.dq < 96) out.push({ sev: 2, title: 'Data integrity gap', detail: `Only ${fmt('dq', r.dq)} of files pass automated data checks (province ${fmt('dq', avg('dq'))}).`, action: 'Escalate data remediation and add a completeness check at intake.' });
  if (r.unemp >= 7.5) out.push({ sev: 2, title: 'Weak local labour market', detail: `Unemployment is ${fmt('unemp', r.unemp)}.`, action: 'Apply a labour-market overlay in adjudication and monitoring.' });
  return out.sort((a, b) => b.sev - a.sev);
}

// ---- query validation (guardrail for LLM output, also used by the rules parser) -----------------------------------
const INTENTS = ['rank', 'profile', 'compare', 'stress', 'drivers', 'trend', 'overview', 'reset', 'help'];
const OPS = ['>', '<', 'worse_third', 'better_third'];

export function validateQuery(q, model) {
  if (!q || typeof q !== 'object' || !INTENTS.includes(q.intent)) return null;
  const out = { intent: q.intent };
  const metric = q.metric in METRICS ? q.metric : null;
  out.metric = metric || (q.intent === 'trend' ? 'dpd90' : 'risk');
  out.order = ['worst', 'best', 'high', 'low'].includes(q.order) ? q.order : 'worst';
  out.n = clip(Math.round(Number(q.n) || 5), 1, 10);
  const toIds = (arr) => {
    const ids = [];
    for (const x of Array.isArray(arr) ? arr : []) {
      if (typeof x === 'number' && model.byId.has(x)) ids.push(x);
      else if (typeof x === 'string') for (const id of model.find(x).regions) ids.push(id);
    }
    return [...new Set(ids)];
  };
  out.regions = toIds(q.regions);
  out.scope = q.scope in GROUPS ? q.scope : null;
  out.filters = (Array.isArray(q.filters) ? q.filters : [])
    .filter((f) => f && f.metric in METRICS && OPS.includes(f.op) && (f.op.endsWith('third') || Number.isFinite(Number(f.value))))
    .slice(0, 3)
    .map((f) => ({ metric: f.metric, op: f.op, value: f.op.endsWith('third') ? null : Number(f.value) }));
  const s = q.scenario || {};
  out.scenario = {
    rate_bps: clip(Number(s.rate_bps) || 0, -200, 400),
    unemp_pp: clip(Number(s.unemp_pp) || 0, -2, 6),
    hpi_pct: clip(Number(s.hpi_pct) || 0, -40, 15),
  };
  if (out.intent === 'compare' && out.regions.length < 2) return null;
  return out;
}

// ---- rules parser (offline fallback, and the safety net when the LLM is slow or wrong) ---------------------------------
const NUM_WORDS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10 };
const METRIC_WORDS = [
  ['dpd_trend', /\b(rising|deteriorat\w*|worsening|trend(?:ing)?\s+(?:up|worse)|getting worse|accelerat\w*)\b/],
  ['dpd90', /\b(delinquen\w*|arrears|past due|default\w*|late payments?|90\+?\s*(?:day|dpd))\b/],
  ['ltv', /\b(ltv|loan[- ]to[- ]value|leverage\w*)\b/],
  ['dti', /\b(dti|tds|gds|debt[- ]service|debt[- ]to[- ]income|affordab\w*)\b/],
  ['renew', /\b(renewals?|renewing|payment shock|rate shock|reprice\w*)\b/],
  ['unemp', /\b(unemploy\w*|jobless|labou?r market)\b/],
  ['hpi', /\b(house prices?|home prices?|hpi|prices? (?:fall|drop|decline)\w*)\b/],
  ['override', /\b(overrides?|exceptions?|adjudicat\w*|underwriting quality)\b/],
  ['dq', /\b(data quality|data integrity|data issues?|completeness)\b/],
  ['score', /\b(credit scores?|beacon|fico|bureau)\b/],
  ['balance', /\b(biggest (?:book|exposure|portfolio)|largest (?:book|exposure|portfolio)|exposure|balances?)\b/],
  ['heloc', /\b(helocs?|home equity)\b/],
  ['risk', /\b(credit risk|risk score|risk|riskiest|risky|safest)\b/],
];

function parseNumber(tok) {
  if (tok === undefined) return null;
  const t = String(tok).toLowerCase();
  return /^\d/.test(t) ? Number(t) : NUM_WORDS[t] ?? null;
}

function parseScenario(t) {
  const s = { ...SCENARIO_NONE };
  let m;
  const down = '(?:fall|falls|drop|drops|dropped|down|decline[sd]?|crash\\w*|correct\\w*|slump\\w*|lower|cut|decrease[sd]?)';
  const up = '(?:rise|rises|rose|up|increase[sd]?|hike[sd]?|go(?:es)? up|climb\\w*|jump\\w*|higher|spike\\w*)';
  if ((m = t.match(/([+-]?\d+(?:\.\d+)?)\s*(?:bps|basis points?)/))) s.rate_bps = Number(m[1]);
  else if ((m = t.match(new RegExp(`(?:interest )?rates?\\s*(?:to\\s*)?(?:${up})\\w*\\s*(?:by\\s*)?(\\d+(?:\\.\\d+)?)`)))) s.rate_bps = Number(m[1]) * (Number(m[1]) > 20 ? 1 : 100);
  else if ((m = t.match(new RegExp(`(?:interest )?rates?\\s*(?:${down})\\w*\\s*(?:by\\s*)?(\\d+(?:\\.\\d+)?)`)))) s.rate_bps = -Number(m[1]) * (Number(m[1]) > 20 ? 1 : 100);
  if (s.rate_bps > 0 && new RegExp(`rates?\\s*(?:${down})`).test(t)) s.rate_bps = -s.rate_bps;
  if ((m = t.match(/unemployment[^.\d+-]*([+-]?\d+(?:\.\d+)?)/))) {
    s.unemp_pp = Number(m[1]);
    if (new RegExp(`unemployment\\s*(?:${down})`).test(t)) s.unemp_pp = -Math.abs(s.unemp_pp);
  }
  if ((m = t.match(/(?:prices?|housing|home values?|real estate)[^.\d+-]*([+-]?\d+(?:\.\d+)?)\s*%/)) || (m = t.match(/([+-]?\d+(?:\.\d+)?)\s*%\s*(?:drop|fall|decline|correction|crash)/))) {
    s.hpi_pct = Number(m[1]);
    if (new RegExp(`(?:${down})`).test(t) && !new RegExp(`prices?\\s*(?:${up})`).test(t)) s.hpi_pct = -Math.abs(s.hpi_pct);
  }
  return s;
}

export function parseRules(text, model, ctx = {}) {
  const t = ` ${String(text).toLowerCase().replace(/\s+/g, ' ').trim()} `;
  const found = model.find(t);
  const q = { intent: 'rank', metric: null, order: 'worst', n: 5, filters: [], regions: found.regions, scope: found.group, scenario: { ...SCENARIO_NONE } };
  const here = /\b(here|this (?:region|area|one)|selected|it)\b/.test(t);
  const regionCtx = found.regions.length ? found.regions : (here || /\b(why|explain|trend|tell me|drivers?|summary|profile|snapshot)\b/.test(t)) && ctx.regionId ? [ctx.regionId] : [];

  if (/\b(reset|clear (?:the )?(?:stress|scenario)|baseline|back to normal|remove (?:the )?(?:stress|scenario))\b/.test(t)) return { ...q, intent: 'reset' };
  if (/^\s*(help|\?|what can (?:you|i)|how do (?:i|you))/.test(t)) return { ...q, intent: 'help' };

  const stressCue = /\b(stress\w*|scenario|what[- ]if|shock\w*|suppose|downturn|recession|sensitivity|simulate)\b/.test(t) || /\bif\b.*\b(rates?|unemployment|prices?)\b/.test(t);
  if (stressCue) {
    const s = parseScenario(t);
    q.intent = 'stress';
    q.scenario = scenarioActive(s) ? s : { ...SCENARIO_SEVERE };
    return q;
  }

  const num = t.match(/\b(?:top|worst|best|riskiest|safest|highest|lowest)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b/) || t.match(/\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:worst|best|riskiest|safest|highest|lowest|areas?|regions?)\b/);
  if (num) q.n = parseNumber(num[1]) ?? 5;

  // filters: numeric thresholds, "high X", "rising X"
  const consumed = new Set();
  const thr = /\b(ltv|dti|tds|delinquen\w*|unemploy\w*|renew\w*|exception\w*|overrides?)\b[^.\d]{0,20}?(above|over|greater than|more than|exceeds?|at least|>=?|below|under|less than|<=?)\s*(\d+(?:\.\d+)?)/g;
  let m;
  while ((m = thr.exec(t))) {
    const key = /ltv/.test(m[1]) ? 'ltv' : /dti|tds/.test(m[1]) ? 'dti' : /delinq/.test(m[1]) ? 'dpd90' : /unemp/.test(m[1]) ? 'unemp' : /renew/.test(m[1]) ? 'renew' : 'override';
    q.filters.push({ metric: key, op: /above|over|greater|more|exceed|least|>/.test(m[2]) ? '>' : '<', value: Number(m[3]) });
    consumed.add(key);
  }
  for (const [key, re] of METRIC_WORDS) {
    if (key === 'risk') continue;
    const hi = new RegExp(`\\b(?:high|elevated|heavy|large)\\s+(?:\\w+\\s+)?${re.source.replace(/^\\b\(/, '(').replace(/\)\\b$/, ')')}`).test(t);
    if (hi && key !== 'dpd_trend') { q.filters.push({ metric: key, op: 'worse_third', value: null }); consumed.add(key); }
  }
  if (/\b(rising|deteriorat\w*|worsening|getting worse)\b/.test(t) && !q.filters.some((f) => f.metric === 'dpd_trend')) {
    q.filters.push({ metric: 'dpd_trend', op: '>', value: 0 });
    consumed.add('dpd_trend');
  }
  q.filters = q.filters.slice(0, 3);

  let metric = null;
  for (const [key, re] of METRIC_WORDS) if (!consumed.has(key) && re.test(t)) { metric = key; break; }
  q.metric = metric || 'risk';
  const numeric = q.filters.filter((f) => f.op === '>' || f.op === '<');
  if (!metric && q.filters.length === 1 && numeric.length === 1) {
    q.metric = numeric[0].metric;
    q.order = numeric[0].op === '>' ? 'high' : 'low';
  }

  if (/\b(lowest|least|smallest|fewest)\b/.test(t)) q.order = 'low';
  else if (/\b(highest|most|largest|biggest|top)\b/.test(t) && q.metric !== 'risk') q.order = 'high';
  else if (/\b(best|safest|strongest|healthiest|lowest risk|least risky)\b/.test(t)) q.order = 'best';

  if (/\b(compare|versus|vs\.?|against|difference between)\b/.test(t) && found.regions.length >= 2) return { ...q, intent: 'compare', regions: found.regions.slice(0, 3) };
  if (/\b(overall|overview|whole (?:portfolio|province|book)|ontario as a whole|portfolio health|how are we doing|big picture|executive summary)\b/.test(t) && !found.regions.length) return { ...q, intent: 'overview' };
  if (/\b(why|drivers?|driving|explain|reasons?|what'?s behind|contribut\w*|breakdown|what makes)\b/.test(t) && regionCtx.length) return { ...q, intent: 'drivers', regions: [regionCtx[0]] };
  if (/\b(trend|over time|history|historical|last \d+ months|past (?:year|two years)|evolv\w*|chart)\b/.test(t) && regionCtx.length) return { ...q, intent: 'trend', metric: metric && metric !== 'dpd_trend' ? metric : 'dpd90', regions: [regionCtx[0]] };
  const rankCue = /\b(which|where|worst|best|top|highest|lowest|rank\w*|riskiest|safest|most|least|areas?|regions?|list|show me)\b/.test(t) || q.filters.length > 0;
  if (regionCtx.length && !(rankCue && !found.regions.length)) return { ...q, intent: 'profile', regions: [regionCtx[0]] };
  if (!metric && !rankCue && !q.filters.length) return { ...q, intent: 'help', unknown: true };
  if (q.regions.length && !q.scope && !rankCue) return { ...q, intent: 'profile', regions: [q.regions[0]] };
  return q;
}

// ---- executor ------------------------------------------------------------------------------------------------------------
function percentile(vals, p) {
  const s = [...vals].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.round((s.length - 1) * p)))];
}

const worseness = (key, v) => (METRICS[key].worseHigh ? v : -v);

export function describeFilter(f, thresholds) {
  const m = METRICS[f.metric];
  if (f.op === '>') return `${m.label} above ${fmt(f.metric, f.value)}`;
  if (f.op === '<') return `${m.label} below ${fmt(f.metric, f.value)}`;
  const th = thresholds[f.metric];
  return `${f.op === 'worse_third' ? 'weakest third' : 'strongest third'} on ${lc(m.label)}${th !== undefined ? ` (${f.op === 'worse_third' ? (m.worseHigh ? '≥' : '≤') : (m.worseHigh ? '≤' : '≥')} ${fmt(f.metric, th)})` : ''}`;
}

export function runQuery(q, model, state) {
  const scenario = state.scenario || SCENARIO_NONE;
  const rows = model.view(scenario);
  const byId = new Map(rows.map((r) => [r.id, r]));
  const prov = (k) => model.weighted(rows, k);
  const res = { query: q, scenario: null, highlight: [], select: null, bullets: [], actions: [], facts: [] };

  switch (q.intent) {
    case 'rank': {
      let pool = rows;
      const label = [];
      if (q.scope) { pool = pool.filter((r) => GROUPS[q.scope].ids.includes(r.id)); label.push(GROUPS[q.scope].label); }
      const thresholds = {};
      for (const f of q.filters) {
        if (f.op.endsWith('third')) {
          const w = rows.map((r) => worseness(f.metric, r[f.metric]));
          const cut = percentile(w, f.op === 'worse_third' ? 2 / 3 : 1 / 3);
          thresholds[f.metric] = METRICS[f.metric].worseHigh ? cut : -cut;
          pool = pool.filter((r) => (f.op === 'worse_third' ? worseness(f.metric, r[f.metric]) >= cut : worseness(f.metric, r[f.metric]) <= cut));
        } else pool = pool.filter((r) => (f.op === '>' ? r[f.metric] > f.value : r[f.metric] < f.value));
      }
      const key = q.metric;
      const dir = q.order === 'high' ? -1 : q.order === 'low' ? 1 : q.order === 'best' ? (METRICS[key].worseHigh ? 1 : -1) : (METRICS[key].worseHigh ? -1 : 1);
      const sorted = [...pool].sort((a, b) => dir * (a[key] - b[key]));
      const top = sorted.slice(0, q.n);
      const word = { worst: 'Weakest', best: 'Strongest', high: 'Highest', low: 'Lowest' }[q.order];
      res.type = 'rank';
      res.metric = key;
      res.title = `${word} regions by ${lc(METRICS[key].label)}`;
      res.subtitle = [...label, ...q.filters.map((f) => describeFilter(f, thresholds))].join(' · ') || `All ${rows.length} regions`;
      res.rows = top.map((r) => ({ id: r.id, name: r.short, value: r[key], display: fmt(key, r[key]), tier: r.tier, share: null }));
      const vals = rows.map((r) => r[key]);
      res.scale = { min: Math.min(...vals), max: Math.max(...vals) };
      res.highlight = top.map((r) => r.id);
      res.metricKey = key;
      if (!top.length) {
        res.headline = 'No region matches all of those conditions.';
        res.facts.push('No region matched the requested filters.');
      } else {
        const t = top[0];
        res.headline = `${t.short} ranks first at ${fmt(key, t[key])}${key === 'risk' ? ` (${t.tier.name.toLowerCase()} tier)` : ''}; ${top.length} region${top.length > 1 ? 's' : ''} shown out of ${pool.length} matching.`;
        res.facts.push(`${res.title}${res.subtitle ? ` (${res.subtitle})` : ''}.`);
        top.forEach((r, i) => {
          const parts = [`${METRICS[key].label} ${fmt(key, r[key])}`];
          if (key !== 'risk') parts.push(`risk score ${r.risk}`);
          if (key !== 'dpd90') parts.push(`90+ delinquency ${fmt('dpd90', r.dpd90)}`);
          if (key !== 'ltv') parts.push(`LTV ${fmt('ltv', r.ltv)}`);
          res.facts.push(`${i + 1}. ${r.short}: ${parts.join('; ')}.`);
        });
        res.facts.push(`Province benchmark for ${lc(METRICS[key].label)}: ${fmt(key, prov(key))}.`);
        const exposed = top.reduce((a, r) => a + r.balance_m, 0);
        res.bullets.push(`These regions hold ${fmt('balance', exposed)} (${Math.round((100 * exposed) / rows.reduce((a, r) => a + r.balance_m, 0))}% of the portfolio).`);
        res.facts.push(res.bullets[0]);
        res.actions.push(...alertsFor(t, rows, model).slice(0, 2).map((a) => `${t.short}: ${a.action}`));
      }
      break;
    }
    case 'profile': {
      const r = byId.get(q.regions[0]);
      res.type = 'profile';
      res.region = r.id;
      res.select = r.id;
      res.highlight = [r.id];
      res.title = `${r.short}: risk profile`;
      res.row = r;
      res.alerts = alertsFor(r, rows, model);
      const rank = [...rows].sort((a, b) => b.risk - a.risk).findIndex((x) => x.id === r.id) + 1;
      res.rank = rank;
      res.headline = `${r.short} scores ${r.risk} (${r.tier.name.toLowerCase()} tier), ranked ${rank} of ${rows.length} for credit risk.`;
      res.facts.push(res.headline, `90+ delinquency ${fmt('dpd90', r.dpd90)} versus ${fmt('dpd90', prov('dpd90'))} province; 6-month change ${fmt('dpd_trend', r.dpd_trend)}.`, `LTV ${fmt('ltv', r.ltv)}, debt service ${fmt('dti', r.dti)}, ${fmt('renew', r.renew)} of balance renewing in 12 months, unemployment ${fmt('unemp', r.unemp)}.`, `Balance ${fmt('balance', r.balance_m)} across ${r.accounts.toLocaleString('en-CA')} accounts.`);
      for (const a of res.alerts.slice(0, 3)) res.facts.push(`${a.title}: ${a.detail}`);
      res.actions.push(...res.alerts.slice(0, 2).map((a) => a.action));
      break;
    }
    case 'compare': {
      const picks = q.regions.map((id) => byId.get(id)).filter(Boolean);
      res.type = 'compare';
      res.title = `Comparison: ${picks.map((r) => r.short).join(' vs ')}`;
      res.regionRows = picks;
      res.keys = ['risk', 'dpd90', 'dpd_trend', 'ltv', 'dti', 'renew', 'unemp', 'override', 'dq'];
      res.highlight = picks.map((r) => r.id);
      const [a, b] = picks;
      const hi = picks.reduce((x, y) => (x.risk >= y.risk ? x : y));
      res.headline = `${hi.short} carries more credit risk (${hi.risk}) than ${picks.find((r) => r !== hi).short} (${picks.find((r) => r !== hi).risk}).`;
      res.facts.push(res.headline);
      for (const r of picks) res.facts.push(`${r.short}: risk ${r.risk}, 90+ delinquency ${fmt('dpd90', r.dpd90)}, 6-month change ${fmt('dpd_trend', r.dpd_trend)}, LTV ${fmt('ltv', r.ltv)}, debt service ${fmt('dti', r.dti)}, renewing ${fmt('renew', r.renew)}, exceptions ${fmt('override', r.override)}.`);
      if (a && b) {
        const gaps = res.keys.slice(1).map((k) => ({ k, gap: Math.abs(worseness(k, a[k]) - worseness(k, b[k])) / (Math.max(...rows.map((r) => r[k])) - Math.min(...rows.map((r) => r[k])) || 1) })).sort((x, y) => y.gap - x.gap);
        res.bullets.push(`Biggest gap: ${METRICS[gaps[0].k].label.toLowerCase()} (${fmt(gaps[0].k, a[gaps[0].k])} vs ${fmt(gaps[0].k, b[gaps[0].k])}).`);
        res.facts.push(res.bullets[0]);
      }
      break;
    }
    case 'drivers': {
      const r = byId.get(q.regions[0]);
      res.type = 'drivers';
      res.region = r.id;
      res.select = r.id;
      res.highlight = [r.id];
      res.title = `What drives ${r.short}'s score of ${r.risk}`;
      const c = contributions(r, model.drivers);
      res.rows = c.map((x) => ({ key: x.key, name: METRICS[x.key].label, points: x.points, display: `${x.points.toFixed(1)} pts`, value: fmt(x.key, r[x.key]), prov: fmt(x.key, prov(x.key)) }));
      const top = c.slice(0, 2);
      res.headline = `${METRICS[top[0].key].label} and ${METRICS[top[1].key].label.toLowerCase()} contribute most to ${r.short}'s score of ${r.risk}.`;
      res.facts.push(res.headline, ...res.rows.map((x) => `${x.name}: ${x.value} (province ${x.prov}), adds ${x.display}.`));
      res.alerts = alertsFor(r, rows, model);
      res.actions.push(...res.alerts.slice(0, 2).map((a) => a.action));
      break;
    }
    case 'trend': {
      const r = byId.get(q.regions[0]);
      const key = q.metric === 'risk' ? 'risk' : 'dpd90';
      res.type = 'trend';
      res.region = r.id;
      res.select = r.id;
      res.highlight = [r.id];
      res.metricKey = key;
      res.title = `${r.short}: ${lc(METRICS[key].label)}, last 24 months`;
      const series = key === 'risk' ? r.risk_hist : r.dpd90_hist;
      const provSeries = model.months.map((_, i) => model.base.reduce((a, x) => a + (key === 'risk' ? x.risk_hist[i] : x.dpd90_hist[i]) * x.balance_m, 0) / model.base.reduce((a, x) => a + x.balance_m, 0));
      res.series = { labels: model.months, region: series, province: provSeries, name: r.short };
      const first = series[0];
      const last = series[series.length - 1];
      res.headline = `${r.short} ${lc(METRICS[key].label)} moved from ${fmt(key, first)} to ${fmt(key, last)} over 24 months; the province moved from ${fmt(key, provSeries[0])} to ${fmt(key, provSeries[provSeries.length - 1])}.`;
      res.facts.push(res.headline);
      break;
    }
    case 'stress': {
      const s = q.scenario;
      res.type = 'stress';
      res.scenario = s;
      const bRows = model.view(SCENARIO_NONE);
      const sRows = model.view(s);
      const totalBal = bRows.reduce((a, r) => a + r.balance_m, 0);
      const bDelq = bRows.reduce((a, r) => a + (r.dpd90 / 100) * r.balance_m, 0);
      const sDelq = sRows.reduce((a, r) => a + (r.dpd90 / 100) * r.balance_m, 0);
      const bScore = model.weighted(bRows, 'risk');
      const sScore = model.weighted(sRows, 'risk');
      const moved = sRows.filter((r, i) => r.tier.name !== bRows[i].tier.name).length;
      const hot = (rs) => rs.filter((r) => r.risk >= 47).reduce((a, r) => a + r.balance_m, 0) / totalBal;
      const impact = sRows.map((r, i) => ({ id: r.id, name: r.short, delta: r.risk - bRows[i].risk, from: bRows[i].risk, to: r.risk, dpd: r.dpd90, tier: r.tier })).sort((a, b) => b.delta - a.delta);
      res.title = 'Stress scenario';
      res.subtitle = scenarioText(s);
      res.kpis = [
        { label: 'Portfolio risk score', from: bScore, to: sScore, fmt: (v) => v.toFixed(1) },
        { label: '90+ delinquent balance', from: bDelq, to: sDelq, fmt: (v) => `$${Math.round(v)}M` },
        { label: 'Balance in elevated/high tiers', from: hot(bRows) * 100, to: hot(sRows) * 100, fmt: (v) => `${Math.round(v)}%` },
      ];
      res.rows = impact.slice(0, 6).map((x) => ({ id: x.id, name: x.name, value: x.delta, display: `+${x.delta.toFixed(1)} pts`, tier: x.tier, share: `${x.from.toFixed(0)} → ${x.to.toFixed(0)}` }));
      res.scale = { min: 0, max: Math.max(...impact.map((x) => x.delta), 1) };
      res.highlight = impact.slice(0, 5).map((x) => x.id);
      res.headline = `Under this scenario the portfolio risk score moves from ${bScore.toFixed(1)} to ${sScore.toFixed(1)}, 90+ day delinquent balances rise by $${Math.round(sDelq - bDelq)}M, and ${moved} region${moved === 1 ? '' : 's'} change risk tier.`;
      res.facts.push(res.subtitle, res.headline, ...impact.slice(0, 5).map((x) => `${x.name}: risk ${x.from.toFixed(0)} to ${x.to.toFixed(0)}.`));
      res.actions.push('Pre-position collections capacity and renewal outreach in the most affected regions.', 'Review LTV caps for new originations where prices are most exposed.');
      break;
    }
    case 'overview': {
      res.type = 'overview';
      res.title = 'Portfolio overview';
      const totalBal = rows.reduce((a, r) => a + r.balance_m, 0);
      const accounts = rows.reduce((a, r) => a + r.accounts, 0);
      const high = rows.filter((r) => r.risk >= 58);
      const elev = rows.filter((r) => r.risk >= 47 && r.risk < 58);
      const rising = [...rows].sort((a, b) => b.dpd_trend - a.dpd_trend)[0];
      res.kpis = [
        { label: 'Portfolio balance', value: fmt('balance', totalBal) },
        { label: 'Accounts', value: accounts.toLocaleString('en-CA') },
        { label: '90+ day delinquency', value: fmt('dpd90', prov('dpd90')) },
        { label: 'Avg risk score', value: prov('risk').toFixed(1) },
        { label: 'In high/elevated tiers', value: `${Math.round((100 * [...high, ...elev].reduce((a, r) => a + r.balance_m, 0)) / totalBal)}% of balance` },
        { label: 'Fastest deteriorating', value: `${rising.short} (${fmt('dpd_trend', rising.dpd_trend)})` },
      ];
      res.headline = `The portfolio holds ${fmt('balance', totalBal)} with 90+ day delinquency at ${fmt('dpd90', prov('dpd90'))}; ${high.length} regions are in the high tier and ${rising.short} is deteriorating fastest.`;
      res.facts.push(res.headline, ...res.kpis.map((k) => `${k.label}: ${k.value}.`));
      res.highlight = high.map((r) => r.id);
      break;
    }
    case 'reset':
      res.type = 'reset';
      res.title = 'Scenario cleared';
      res.scenario = { ...SCENARIO_NONE };
      res.headline = 'Back to the baseline view.';
      break;
    default:
      res.type = 'help';
      res.title = 'Ask me about credit risk';
      res.headline = q.unknown ? "I couldn't map that to a credit-risk question." : 'Here is what I can do.';
  }
  if (scenarioActive(scenario) && !['stress', 'reset', 'help', 'trend'].includes(res.type)) {
    res.subtitle = `${res.subtitle ? `${res.subtitle} · ` : ''}Under active stress scenario`;
    res.facts.push(`These figures are under a stress scenario (${scenarioText(scenario)}).`);
  }
  return res;
}

export function scenarioText(s) {
  const parts = [];
  if (s.rate_bps) parts.push(`rates ${s.rate_bps > 0 ? '+' : ''}${s.rate_bps} bps`);
  if (s.unemp_pp) parts.push(`unemployment ${s.unemp_pp > 0 ? '+' : ''}${s.unemp_pp} pts`);
  if (s.hpi_pct) parts.push(`house prices ${s.hpi_pct > 0 ? '+' : ''}${s.hpi_pct}%`);
  return parts.length ? `Scenario: ${parts.join(', ')}` : 'No scenario';
}

/** Every number in an AI-written summary must already appear in the facts it was given. */
export function numbersGrounded(text, facts) {
  const hay = String(facts).replace(/,/g, '');
  const nums = String(text).replace(/,/g, '').match(/\d+(?:\.\d+)?/g) || [];
  return nums.every((n) => hay.includes(n));
}
