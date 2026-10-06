import {
  METRICS, SCENARIO_NONE, SCENARIO_SEVERE, createModel, parseRules, runQuery, alertsFor, scenarioActive, scenarioText, fmt,
} from './engine.js';
import { connect, disconnect, savedToken, llmLive, llmModel, interpret, narrate } from './llm.js';

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const NS = 'http://www.w3.org/2000/svg';
const MAP_METRICS = ['risk', 'dpd90', 'dpd_trend', 'ltv', 'dti', 'renew', 'unemp'];
const CHIP_LABEL = { risk: 'Risk score', dpd90: '90+ delinquency', dpd_trend: 'Delinquency trend', ltv: 'LTV', dti: 'Debt service', renew: 'Renewal exposure', unemp: 'Unemployment' };

const state = { scenario: { ...SCENARIO_NONE }, metric: 'risk', regionId: null, zoom: 'south', rows: [], highlight: [], busy: false };
let model; let geo; let cities;
const charts = new Set();

// ------------------------------------------------------------------------------------------------ loading
async function load() {
  const get = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(p); return r.json(); });
  const [portfolio, g, c] = await Promise.all([get('data/portfolio.json'), get('data/regions.geojson'), get('data/cities.json')]);
  geo = g; cities = c;
  model = createModel(portfolio, c);
}

// ------------------------------------------------------------------------------------------------ colour ramp
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function ramp(t) {
  const stops = ['--seq-100', '--seq-300', '--seq-500', '--seq-700'].map((v) => hexToRgb(css(v)));
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function metricRange() {
  const base = model.view(SCENARIO_NONE);
  const key = state.metric;
  const vals = [...base, ...state.rows].map((r) => r[key]);
  return { min: Math.min(...vals), max: Math.max(...vals) };
}

// ------------------------------------------------------------------------------------------------ map
const WIDTH = 1000;
const KX = Math.cos((49 * Math.PI) / 180);
const VIEWS = { all: null, south: [-83.6, 41.6, -74.0, 46.3] };
const proj = ([lon, lat]) => [lon * KX, -lat];
let mapApi;

function buildMap() {
  const box = [Infinity, Infinity, -Infinity, -Infinity];
  for (const f of geo.features) for (const poly of f.geometry.coordinates) for (const ring of poly) for (const [lon, lat] of ring) {
    box[0] = Math.min(box[0], lon); box[1] = Math.min(box[1], lat); box[2] = Math.max(box[2], lon); box[3] = Math.max(box[3], lat);
  }
  const vb = (bb) => { const [x1, y1] = proj([bb[0], bb[3]]); const [x2, y2] = proj([bb[2], bb[1]]); return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }; };
  const full = vb(box);
  const k = WIDTH / full.w;
  const xy = (lon, lat) => { const [x, y] = proj([lon, lat]); return [(x - full.x) * k, (y - full.y) * k]; };
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'map-svg');
  svg.setAttribute('role', 'group');
  svg.setAttribute('aria-label', 'Map of Ontario regions coloured by the selected credit risk metric. Use Tab and Enter to select a region.');
  const g = document.createElementNS(NS, 'g');
  const cityG = document.createElementNS(NS, 'g');
  const badgeG = document.createElementNS(NS, 'g');
  svg.append(g, cityG, badgeG);
  const paths = new Map();
  const centers = new Map();
  const tip = $('#tip');

  for (const f of geo.features) {
    let d = '';
    let best = null;
    for (const poly of f.geometry.coordinates) for (const ring of poly) {
      let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
      ring.forEach((pt, i) => {
        const [x, y] = xy(pt[0], pt[1]);
        d += `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      });
      d += 'Z';
      const area = (maxX - minX) * (maxY - minY);
      if (!best || area > best.area) best = { area, c: [(minX + maxX) / 2, (minY + maxY) / 2] };
    }
    const id = f.properties.id;
    centers.set(id, best.c);
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('d', d);
    p.setAttribute('class', 'reg');
    p.setAttribute('fill-rule', 'evenodd');
    p.setAttribute('tabindex', '0');
    p.setAttribute('role', 'button');
    p.dataset.id = id;
    p.addEventListener('click', () => selectRegion(id));
    p.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectRegion(id); } });
    const show = (e) => showTip(id, e, p);
    p.addEventListener('pointermove', show);
    p.addEventListener('pointerenter', show);
    p.addEventListener('pointerleave', () => { tip.hidden = true; });
    p.addEventListener('focus', () => showTip(id, null, p));
    p.addEventListener('blur', () => { tip.hidden = true; });
    g.appendChild(p);
    paths.set(id, p);
  }

  const cityNodes = [];
  const LEFT = new Set(['Hamilton', 'Sarnia', 'Kitchener', 'Sault Ste. Marie', 'Ottawa', 'Kingston', 'Sudbury', 'London']);
  for (const c of cities.filter((x) => x.label)) {
    const [x, y] = xy(c.lon, c.lat);
    const gg = document.createElementNS(NS, 'g');
    gg.setAttribute('class', 'city');
    const dot = document.createElementNS(NS, 'circle');
    const t = document.createElementNS(NS, 'text');
    t.textContent = c.name;
    gg.append(dot, t);
    cityG.appendChild(gg);
    cityNodes.push({ c, x, y, dot, t, left: LEFT.has(c.name) });
  }

  function applyScale() {
    const v = svg.viewBox.baseVal;
    const px = svg.clientWidth || 600;
    const u = v.width / px;
    for (const n of cityNodes) {
      const show = (state.zoom === 'all' ? n.c.label.includes('A') : n.c.label.includes('S'));
      n.dot.parentNode.style.display = show ? '' : 'none';
      n.dot.setAttribute('cx', n.x); n.dot.setAttribute('cy', n.y); n.dot.setAttribute('r', 3 * u);
      n.t.setAttribute('x', n.x + (n.left ? -6 : 6) * u);
      n.t.setAttribute('y', n.y + 4 * u);
      n.t.setAttribute('text-anchor', n.left ? 'end' : 'start');
      n.t.style.fontSize = `${11 * u}px`;
      n.t.style.strokeWidth = `${3 * u}px`;
    }
    badgeG.querySelectorAll('.badge').forEach((b) => {
      b.querySelector('circle').setAttribute('r', 11 * u);
      b.querySelector('text').style.fontSize = `${12 * u}px`;
    });
  }

  function setZoom(z) {
    state.zoom = z;
    const bb = VIEWS[z] || box;
    const a = vb(bb);
    const x0 = (a.x - full.x) * k; const y0 = (a.y - full.y) * k;
    svg.setAttribute('viewBox', `${x0.toFixed(1)} ${y0.toFixed(1)} ${(a.w * k).toFixed(1)} ${(a.h * k).toFixed(1)}`);
    applyScale();
  }

  function showBadges(ids) {
    badgeG.replaceChildren();
    ids.forEach((id, i) => {
      const c = centers.get(id);
      if (!c) return;
      const b = document.createElementNS(NS, 'g');
      b.setAttribute('class', 'badge');
      const ci = document.createElementNS(NS, 'circle');
      ci.setAttribute('cx', c[0]); ci.setAttribute('cy', c[1]);
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('x', c[0]); t.setAttribute('y', c[1]);
      t.textContent = String(i + 1);
      b.append(ci, t);
      badgeG.appendChild(b);
    });
    applyScale();
  }

  $('#map').replaceChildren(svg);
  setZoom(state.zoom);
  window.addEventListener('resize', applyScale);
  mapApi = { paths, setZoom, showBadges, applyScale };
}

function showTip(id, e, el) {
  const r = state.rows.find((x) => x.id === id);
  const tip = $('#tip');
  if (!r) return;
  const wrap = $('.map-wrap').getBoundingClientRect();
  const box = el.getBoundingClientRect();
  const m = state.metric;
  tip.innerHTML = `<b>${esc(r.short)}</b>${esc(METRICS[m].label)}: ${esc(fmt(m, r[m]))}<br>Risk ${r.risk} &middot; ${esc(r.tier.name)}<br>${esc(fmt('balance', r.balance_m))} balance`;
  tip.hidden = false;
  const x = (e ? e.clientX : box.left + box.width / 2) - wrap.left;
  const y = (e ? e.clientY : box.top + box.height / 2) - wrap.top;
  tip.style.left = `${Math.min(Math.max(8, x + 12), wrap.width - tip.offsetWidth - 8)}px`;
  tip.style.top = `${Math.min(Math.max(8, y + 12), wrap.height - tip.offsetHeight - 8)}px`;
}

function paintMap() {
  const key = state.metric;
  const { min, max } = metricRange();
  const worseHigh = METRICS[key].worseHigh;
  for (const r of state.rows) {
    const p = mapApi.paths.get(r.id);
    if (!p) continue;
    let t = max === min ? 0.5 : (r[key] - min) / (max - min);
    if (!worseHigh) t = 1 - t;
    p.style.fill = ramp(t);
    p.setAttribute('aria-label', `${r.short}: ${METRICS[key].label} ${fmt(key, r[key])}, risk tier ${r.tier.name}`);
    p.classList.toggle('sel', r.id === state.regionId);
    p.classList.toggle('hl', state.highlight.includes(r.id));
  }
  $('#lg-lo').textContent = `${fmt(key, worseHigh ? min : max)} lower risk`;
  $('#lg-hi').textContent = `higher risk ${fmt(key, worseHigh ? max : min)}`;
}

// ------------------------------------------------------------------------------------------------ table
let sortKey = 'risk'; let sortDir = -1;
const TBL_COLS = [['short', 'Region'], ['risk', 'Risk'], ['dpd90', '90+ DPD'], ['dpd_trend', '6m Δ'], ['ltv', 'LTV'], ['dti', 'DSR'], ['renew', 'Renewing'], ['unemp', 'Unemp.'], ['balance_m', 'Balance']];
function paintTable() {
  const rows = [...state.rows].sort((a, b) => sortDir * (typeof a[sortKey] === 'string' ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey]));
  const cell = (r, k) => (k === 'short' ? esc(r.short) : k === 'balance_m' ? fmt('balance', r.balance_m) : esc(fmt(k, r[k])));
  $('#tbl').innerHTML = `<thead><tr>${TBL_COLS.map(([k, l]) => `<th data-k="${k}" aria-sort="${k === sortKey ? (sortDir > 0 ? 'ascending' : 'descending') : 'none'}">${l}${k === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr data-id="${r.id}">${TBL_COLS.map(([k]) => `<td>${cell(r, k)}</td>`).join('')}</tr>`).join('')}</tbody>`;
}

// ------------------------------------------------------------------------------------------------ charts
function lineChart(canvas, labels, a, b, fmtKey, nameA, nameB) {
  const ink2 = css('--ink2'); const grid = css('--grid');
  const ch = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: nameA, data: a, borderColor: css('--seq-400'), backgroundColor: css('--seq-400'), borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, tension: 0.25 },
        { label: nameB, data: b, borderColor: css('--muted'), borderDash: [5, 4], borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.25 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 600 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', align: 'end', labels: { color: ink2, boxWidth: 14, boxHeight: 2, usePointStyle: false, font: { size: 12 } } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(fmtKey, c.parsed.y)}` } },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: ink2, maxTicksLimit: 6, font: { size: 11 } }, border: { color: css('--axis') } },
        y: { grid: { color: grid }, ticks: { color: ink2, font: { size: 11 }, callback: (v) => fmt(fmtKey, v) }, border: { display: false } },
      },
    },
  });
  charts.add(ch);
  return ch;
}
function dropCharts(container) {
  for (const ch of [...charts]) if (!container || container.contains(ch.canvas) || !document.body.contains(ch.canvas)) { ch.destroy(); charts.delete(ch); }
}

// ------------------------------------------------------------------------------------------------ detail card
function paintSel(r) {
  const s = $('#map-sel');
  if (!r) { s.hidden = true; return; }
  const icon = { t1: '✔', t2: '●', t3: '▲', t4: '⬣' }[r.tier.cls];
  s.hidden = false;
  s.innerHTML = `<b>${esc(r.short)}</b><span class="tier ${r.tier.cls}">${icon} ${r.tier.name}</span><span>Risk <b>${r.risk.toFixed(1)}</b></span><span>90+ DPD <b>${esc(fmt('dpd90', r.dpd90))}</b></span><a href="#detail" class="link">details &darr;</a>`;
}

function paintDetail() {
  const el = $('#detail');
  if (!state.regionId) { el.hidden = true; dropCharts(el); paintSel(null); return; }
  const r = state.rows.find((x) => x.id === state.regionId);
  paintSel(r);
  const base = model.byId.get(r.id);
  const baseRow = model.view(SCENARIO_NONE).find((x) => x.id === r.id);
  const delta = r.risk - baseRow.risk;
  const prov = (k) => model.weighted(state.rows, k);
  const kpi = (label, val, sub, cls = '') => `<div class="kpi"><span>${label}</span><b>${val}</b>${sub ? `<small class="${cls}">${sub}</small>` : ''}</div>`;
  const icon = { t1: '✔', t2: '●', t3: '▲', t4: '⬣' }[r.tier.cls];
  const alerts = alertsFor(r, state.rows, model);
  dropCharts(el);
  el.hidden = false;
  el.innerHTML = `
    <div class="d-head"><h2>${esc(r.short)}</h2><span class="tier ${r.tier.cls}">${icon} ${r.tier.name} risk</span><span class="note">${r.accounts.toLocaleString('en-CA')} accounts &middot; ${esc(fmt('balance', r.balance_m))}</span></div>
    <div class="kpis">
      ${kpi('Risk score', r.risk.toFixed(1), delta ? `${delta > 0 ? '+' : ''}${delta.toFixed(1)} stress` : `Ontario ${prov('risk').toFixed(1)}`, delta > 0 ? 'up' : '')}
      ${kpi('90+ delinquency', fmt('dpd90', r.dpd90), `Ontario ${fmt('dpd90', prov('dpd90'))}`)}
      ${kpi('6-month change', fmt('dpd_trend', r.dpd_trend), '', r.dpd_trend > 3 ? 'up' : '')}
      ${kpi('Avg LTV', fmt('ltv', r.ltv), `Ontario ${fmt('ltv', prov('ltv'))}`)}
      ${kpi('Debt service', fmt('dti', r.dti), `Ontario ${fmt('dti', prov('dti'))}`)}
      ${kpi('Renewing in 12m', fmt('renew', r.renew), '')}
      ${kpi('Exception rate', fmt('override', r.override), `Ontario ${fmt('override', prov('override'))}`)}
      ${kpi('Data checks passed', fmt('dq', r.dq), '')}
    </div>
    <div class="chart-box"><canvas aria-label="Risk score trend for ${esc(r.short)} versus Ontario, 24 months" role="img"></canvas></div>
    <div class="alerts">${alerts.length ? alerts.slice(0, 3).map((a) => `<div class="alert"><b>${esc(a.title)}</b>${esc(a.detail)}<em>&rarr; ${esc(a.action)}</em></div>`).join('') : '<div class="alert"><b>No active watchlist flags</b><em>All monitored indicators are within normal ranges for this region.</em></div>'}</div>`;
  const provSeries = model.months.map((_, i) => model.base.reduce((a, x) => a + x.risk_hist[i] * x.balance_m, 0) / model.base.reduce((a, x) => a + x.balance_m, 0));
  lineChart($('canvas', el), model.months, base.risk_hist, provSeries, 'risk', `${r.short} risk score`, 'Ontario (balance-weighted)');
}

// ------------------------------------------------------------------------------------------------ refresh
function refresh() {
  state.rows = model.view(state.scenario);
  paintMap();
  paintTable();
  paintDetail();
  const on = scenarioActive(state.scenario);
  $('#scenario-bar').hidden = !on;
  $('#scenario-text').textContent = scenarioText(state.scenario);
  $('#s-rate').value = state.scenario.rate_bps; $('#s-unemp').value = state.scenario.unemp_pp; $('#s-hpi').value = state.scenario.hpi_pct;
  $('#v-rate').textContent = `${state.scenario.rate_bps > 0 ? '+' : ''}${state.scenario.rate_bps} bps`;
  $('#v-unemp').textContent = `${state.scenario.unemp_pp > 0 ? '+' : ''}${state.scenario.unemp_pp} pts`;
  $('#v-hpi').textContent = `${state.scenario.hpi_pct > 0 ? '+' : ''}${state.scenario.hpi_pct}%`;
  paintContext();
}

function selectRegion(id) {
  state.regionId = id;
  refresh();
}
function setHighlight(ids) { state.highlight = ids || []; mapApi.showBadges(state.highlight); paintMap(); }

function paintContext() {
  const c = $('#ctx');
  if (state.regionId) { c.hidden = false; c.textContent = `📍 ${model.byId.get(state.regionId).short} selected`; } else c.hidden = true;
  const sg = [];
  if (state.regionId) { const n = model.byId.get(state.regionId).short; sg.push(`Why is ${n} scored this way?`, `Show the delinquency trend for ${n}`, `Tell me about ${n}`); }
  sg.push('Where is credit risk worst?', 'Which areas have high LTV and rising delinquency?', 'Compare Toronto and Ottawa', 'What if rates rise 200 bps and house prices fall 15%?', 'Give me a portfolio overview', 'Which regions have the most exceptions?');
  $('#suggest').innerHTML = sg.slice(0, 5).map((s) => `<button type="button" class="sg">${esc(s)}</button>`).join('');
}

// ------------------------------------------------------------------------------------------------ chat rendering
const msgs = () => $('#msgs');
function scrollDown() { const m = msgs(); m.scrollTop = m.scrollHeight; }
function addUser(text) { const d = document.createElement('div'); d.className = 'msg user'; d.textContent = text; msgs().appendChild(d); scrollDown(); }
function addBot(html) { const d = document.createElement('div'); d.className = 'msg bot'; d.innerHTML = html; msgs().appendChild(d); scrollDown(); return d; }

const THINK = (t) => `<span class="think"><i></i><i></i><i></i>&nbsp;${esc(t)}</span>`;

function barsHtml(res) {
  const maxAbs = Math.max(...res.rows.map((r) => Math.abs(r.value)), 1e-9);
  return `<div class="bars">${res.rows.map((r, i) => `<div class="bar-row" data-id="${r.id}" tabindex="0" role="button" aria-label="${esc(r.name)} ${esc(r.display)}"><span class="rk">${i + 1}</span><span class="nm">${esc(r.name)}</span><span class="track"><span class="fill" data-w="${(Math.abs(r.value) / maxAbs) * 100}"></span></span><span class="val">${esc(r.display)}${r.share ? `<small>${esc(r.share)}</small>` : ''}</span></div>`).join('')}</div>`;
}
function driversHtml(res) {
  const max = Math.max(...res.rows.map((r) => r.points), 1e-9);
  return `<div class="bars">${res.rows.map((r) => `<div class="drv-row"><span class="nm">${esc(r.name)}</span><span class="track"><span class="fill" data-w="${(r.points / max) * 100}"></span></span><span class="val">${esc(r.display)}<small>${esc(r.value)} vs ${esc(r.prov)}</small></span></div>`).join('')}</div>`;
}
function compareHtml(res) {
  const [a, b] = res.regionRows;
  const head = res.regionRows.map((r) => `<th>${esc(r.short)}</th>`).join('');
  const body = res.keys.map((k) => {
    const worst = res.regionRows.reduce((w, r) => ((METRICS[k].worseHigh ? r[k] > w[k] : r[k] < w[k]) ? r : w));
    const tied = a && b && a[k] === b[k];
    return `<tr><td>${esc(METRICS[k].label)}</td>${res.regionRows.map((r) => `<td class="${!tied && r === worst ? 'worse' : ''}">${esc(fmt(k, r[k]))}</td>`).join('')}</tr>`;
  }).join('');
  return `<table class="mini-tbl"><thead><tr><th></th>${head}</tr></thead><tbody>${body}</tbody></table><p class="note">Bold = weaker of the two on that measure.</p>`;
}
function stressHtml(res) {
  const k = res.kpis.map((x) => `<div class="kpi"><span>${esc(x.label)}</span><b>${esc(x.fmt(x.to))}</b> <small class="arrow">from ${esc(x.fmt(x.from))}</small></div>`).join('');
  return `<div class="kpi-row">${k}</div><p class="actions-h">Most affected regions (risk score points)</p>${barsHtml(res)}`;
}
function profileHtml(res) {
  const r = res.row;
  const k = (l, v) => `<div class="kpi"><span>${l}</span><b>${esc(v)}</b></div>`;
  return `<div class="kpi-row">${k('Risk score', r.risk.toFixed(1))}${k('90+ delinquency', fmt('dpd90', r.dpd90))}${k('6-month change', fmt('dpd_trend', r.dpd_trend))}${k('Avg LTV', fmt('ltv', r.ltv))}${k('Debt service', fmt('dti', r.dti))}${k('Balance', fmt('balance', r.balance_m))}</div>${res.alerts.length ? `<p class="actions-h">Watchlist</p><div class="alerts">${res.alerts.slice(0, 3).map((a) => `<div class="alert"><b>${esc(a.title)}</b>${esc(a.detail)}</div>`).join('')}</div>` : ''}`;
}
function overviewHtml(res) { return `<div class="kpi-row">${res.kpis.map((x) => `<div class="kpi"><span>${esc(x.label)}</span><b style="font-size:15px">${esc(x.value)}</b></div>`).join('')}</div>`; }
function helpHtml() {
  return `<ul class="help-list"><li>Rank regions: <i>"Which areas have the worst credit risk?"</i></li><li>Combine filters: <i>"High LTV and rising delinquency"</i></li><li>Explain a region: click the map, then <i>"Why is it high?"</i></li><li>Compare: <i>"Compare Peel and Ottawa"</i></li><li>Stress test: <i>"What if unemployment rises 2 points?"</i></li></ul>`;
}

function renderResult(el, res, source, question) {
  const body = {
    rank: () => (res.rows.length ? barsHtml(res) : ''),
    profile: () => profileHtml(res), compare: () => compareHtml(res), drivers: () => driversHtml(res),
    stress: () => stressHtml(res), overview: () => overviewHtml(res), trend: () => '<div class="chart-box"><canvas role="img" aria-label="Trend chart"></canvas></div>', help: helpHtml, reset: () => '',
  }[res.type]();
  const how = res.type === 'help' || res.type === 'reset' ? '' : `<details class="how"><summary>How the AI read this</summary><pre>${esc(JSON.stringify({ interpreted_by: source, query: stripQuery(res.query) }, null, 2))}</pre></details>`;
  el.innerHTML = `<h3>${esc(res.title)}</h3>${res.subtitle ? `<div class="subtitle">${esc(res.subtitle)}</div>` : ''}<p class="headline">${esc(res.headline)}</p><div class="ai-slot"></div>${body}${res.actions && res.actions.length ? `<p class="actions-h">Suggested next steps</p><ul class="actions">${res.actions.slice(0, 3).map((a) => `<li>${esc(a)}</li>`).join('')}</ul>` : ''}${how}`;
  requestAnimationFrame(() => requestAnimationFrame(() => el.querySelectorAll('.fill').forEach((f) => { f.style.width = `${f.dataset.w}%`; })));
  if (res.type === 'trend') lineChart($('canvas', el), res.series.labels, res.series.region, res.series.province, res.metricKey, res.series.name, 'Ontario (balance-weighted)');
  if (res.type === 'help') el.querySelectorAll('.help-list i').forEach((i) => { i.style.cursor = 'pointer'; i.addEventListener('click', () => ask(i.textContent.replace(/^"|"$/g, ''))); });
  scrollDown();
}
function stripQuery(q) {
  const o = { intent: q.intent };
  if (['rank'].includes(q.intent)) Object.assign(o, { metric: q.metric, order: q.order, n: q.n, filters: q.filters, scope: q.scope });
  if (['profile', 'compare', 'drivers', 'trend'].includes(q.intent)) Object.assign(o, { regions: q.regions.map((id) => model.byId.get(id).short) }, q.intent === 'trend' ? { metric: q.metric } : {});
  if (q.intent === 'stress') o.scenario = q.scenario;
  return o;
}

// ------------------------------------------------------------------------------------------------ ask flow
async function ask(text) {
  const question = String(text || '').trim();
  if (!question || state.busy) return;
  state.busy = true;
  $('#send').disabled = true;
  addUser(question);
  const ctx = { regionId: state.regionId, regionName: state.regionId ? model.byId.get(state.regionId).short : '' };
  const bot = addBot(THINK(llmLive() ? `Interpreting with ${llmModel().split('/').pop()}…` : 'Reading your question…'));
  const rules = parseRules(question, model, ctx);
  let q = null;
  let source = 'rules engine';
  if (llmLive()) {
    const aiQ = await interpret(question, model, ctx);
    if (aiQ && aiQ.intent !== 'help') {
      if (['profile', 'drivers', 'trend'].includes(aiQ.intent) && !aiQ.regions.length && ctx.regionId) aiQ.regions = [ctx.regionId];
      if (['profile', 'drivers', 'trend'].includes(aiQ.intent) && !aiQ.regions.length) { /* fall through to rules */ } else { q = aiQ; source = `AI (${llmModel()})`; }
    } else if (aiQ === null) source = 'rules engine (AI unavailable)';
  }
  if (!q) q = rules;
  const res = runQuery(q, model, { scenario: state.scenario });
  if (res.type === 'stress') state.scenario = { ...res.scenario };
  if (res.type === 'reset') state.scenario = { ...SCENARIO_NONE };
  if (res.select) state.regionId = res.select;
  refresh();
  setHighlight(res.highlight);
  renderResult(bot, res, source, question);
  bot.querySelectorAll('.bar-row').forEach((row) => {
    const go = () => { selectRegion(Number(row.dataset.id)); setHighlight([Number(row.dataset.id)]); };
    row.addEventListener('click', go);
    row.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  });
  state.busy = false;
  $('#send').disabled = false;
  if (llmLive() && res.facts.length) {
    narrate(question, res).then((t) => {
      const slot = $('.ai-slot', bot);
      if (t && slot) { slot.innerHTML = `<div class="ai-sum">${esc(t)}<small>AI summary &middot; every number verified against the data</small></div>`; scrollDown(); }
    });
  }
}

// ------------------------------------------------------------------------------------------------ guided demo
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function tour() {
  if (state.busy) return;
  const btn = $('#tour-btn');
  btn.disabled = true;
  const input = $('#q');
  const steps = [
    ['Give me a portfolio overview'],
    ['Which areas have the worst credit risk?'],
    ['Why is Toronto scored so high?'],
    ['Which areas have high LTV and rising delinquency?'],
    ['What if rates rise 200 bps and house prices fall 15%?'],
    ['Reset the scenario'],
  ];
  for (const [text] of steps) {
    input.value = '';
    for (const ch of text) { input.value += ch; await sleep(18); }
    await sleep(250);
    input.value = '';
    await ask(text);
    while (state.busy) await sleep(100);
    await sleep(2800);
  }
  btn.disabled = false;
}

// ------------------------------------------------------------------------------------------------ boot
function wire() {
  $('#metric-chips').innerHTML = MAP_METRICS.map((m) => `<button type="button" class="chip" data-m="${m}" aria-pressed="${m === state.metric}">${CHIP_LABEL[m]}</button>`).join('');
  $('#metric-chips').addEventListener('click', (e) => {
    const b = e.target.closest('.chip'); if (!b) return;
    state.metric = b.dataset.m;
    document.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
    paintMap();
  });
  document.querySelectorAll('[data-zoom]').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('[data-zoom]').forEach((x) => x.classList.toggle('on', x === b));
    mapApi.setZoom(b.dataset.zoom); mapApi.showBadges(state.highlight);
  }));
  const slider = () => { state.scenario = { rate_bps: Number($('#s-rate').value), unemp_pp: Number($('#s-unemp').value), hpi_pct: Number($('#s-hpi').value) }; refresh(); };
  ['#s-rate', '#s-unemp', '#s-hpi'].forEach((s) => $(s).addEventListener('input', slider));
  $('#preset-severe').addEventListener('click', () => { state.scenario = { ...SCENARIO_SEVERE }; refresh(); });
  const reset = () => { state.scenario = { ...SCENARIO_NONE }; refresh(); };
  $('#preset-reset').addEventListener('click', reset);
  $('#scenario-reset').addEventListener('click', reset);
  $('#ask').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#q').value; $('#q').value = ''; ask(v); });
  $('#suggest').addEventListener('click', (e) => { const b = e.target.closest('.sg'); if (b) ask(b.textContent); });
  $('#tbl').addEventListener('click', (e) => {
    const th = e.target.closest('th'); const tr = e.target.closest('tbody tr');
    if (th) { const k = th.dataset.k; sortDir = k === sortKey ? -sortDir : (k === 'short' ? 1 : -1); sortKey = k; paintTable(); }
    else if (tr) selectRegion(Number(tr.dataset.id));
  });
  $('#tour-btn').addEventListener('click', tour);
  $('#theme-btn').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('crc-theme', next); } catch (e) { /* ignore */ }
    syncTheme(); refresh();
  });
}
function syncTheme() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  $('#theme-btn').textContent = dark ? 'Light' : 'Dark';
  $('#theme-btn').setAttribute('aria-pressed', String(dark));
}

async function boot() {
  try { await load(); } catch (e) {
    $('#main').innerHTML = '<div class="card"><h2>The data could not be loaded</h2><p class="note">Reload the page to try again.</p></div>';
    return;
  }
  buildMap();
  wire();
  syncTheme();
  state.rows = model.view(state.scenario);
  refresh();
  addBot(`<h3>Hi, I'm your credit risk copilot.</h3><p class="headline">Click any region on the map, or ask me a question in plain language. I can rank regions, explain what drives a score, compare areas and run what-if stress tests.</p><p class="note">Try a suggestion below, press <b>Guided demo</b> for a 60-second tour, or <b>Connect AI</b> (top right) to ask in your own words.</p>`);
  wireAi();
  const saved = savedToken();
  if (saved) { await connect(saved, true); }
  paintAi();
}

function paintAi() {
  const live = llmLive();
  $('#ai-pill').classList.toggle('live', live);
  $('#ai-text').textContent = live ? `AI live · ${llmModel().split('/').pop()}` : 'Connect AI · rules mode';
}

function wireAi() {
  const pop = $('#ai-pop');
  const pill = $('#ai-pill');
  const toggle = (open) => { pop.hidden = !open; pill.setAttribute('aria-expanded', String(open)); if (open) $('#ai-token').focus(); };
  pill.addEventListener('click', () => toggle(pop.hidden));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') toggle(false); });
  document.addEventListener('click', (e) => { if (!pop.hidden && !e.target.closest('.ai-wrap')) toggle(false); });
  $('#ai-connect').addEventListener('click', async () => {
    const btn = $('#ai-connect');
    btn.disabled = true; $('#ai-msg').textContent = 'Checking token…';
    const r = await connect($('#ai-token').value, $('#ai-remember').checked);
    btn.disabled = false;
    $('#ai-msg').textContent = r.ok ? 'Connected. Ask me anything.' : r.reason;
    if (r.ok) { $('#ai-token').value = ''; setTimeout(() => toggle(false), 700); }
    paintAi();
  });
  $('#ai-off').addEventListener('click', () => { disconnect(); $('#ai-msg').textContent = 'Disconnected. Using the rules engine.'; paintAi(); });
}
boot();
