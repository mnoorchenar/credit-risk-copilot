// Shared DOM and charting helpers (browser only).
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
export const num = (v, d = 3) => Number(v).toFixed(d);
export const money = (v) => (Math.abs(v) >= 1e9 ? `$${(v / 1e9).toFixed(2)}B` : Math.abs(v) >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : Math.abs(v) >= 1e3 ? `$${(v / 1e3).toFixed(0)}k` : `$${v.toFixed(0)}`);
export const hexRgb = (h) => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };

export const colors = () => ({
  champion: css('--c-champ'), challenger: css('--c-chal'), good: css('--c-good'), bad: css('--c-bad'),
  ink: css('--ink'), ink2: css('--ink2'), muted: css('--muted'), grid: css('--grid'), axis: css('--axis'), surface: css('--surface'),
  lo: css('--div-lo'), mid: css('--div-mid'), hi: css('--div-hi'), accent: css('--accent'),
});

/** Interpolates blue -> grey -> red for t in [0,1]. */
export function diverge(t) {
  const c = colors();
  const a = hexRgb(c.lo); const m = hexRgb(c.mid); const b = hexRgb(c.hi);
  const [p, q, u] = t < 0.5 ? [a, m, t * 2] : [m, b, (t - 0.5) * 2];
  return `rgb(${p.map((v, i) => Math.round(v + (q[i] - v) * u)).join(',')})`;
}

const registry = new Map();
export function chart(canvas, config) {
  const el = typeof canvas === 'string' ? $(canvas) : canvas;
  if (!el) return null;
  const old = registry.get(el);
  if (old) old.destroy();
  const c = colors();
  const base = {
    responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
    plugins: { legend: { labels: { color: c.ink2, boxWidth: 14, boxHeight: 3, font: { size: 12 } } }, tooltip: { backgroundColor: c.ink, titleColor: c.surface, bodyColor: c.surface } },
  };
  const merged = { ...config, options: deepMerge(base, config.options || {}) };
  const ch = new Chart(el, merged);
  registry.set(el, ch);
  return ch;
}
export function destroyCharts(root) {
  for (const [el, ch] of [...registry]) if (!root || root.contains(el) || !document.body.contains(el)) { ch.destroy(); registry.delete(el); }
}
function deepMerge(a, b) {
  const out = { ...a };
  for (const k of Object.keys(b)) out[k] = b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object' ? deepMerge(a[k], b[k]) : b[k];
  return out;
}
export function axes({ x = {}, y = {}, xTitle, yTitle } = {}) {
  const c = colors();
  const mk = (o, title, grid) => ({
    grid: { color: grid ? c.grid : 'transparent', drawTicks: false }, border: { color: c.axis, display: !grid },
    ticks: { color: c.ink2, font: { size: 11 }, maxTicksLimit: 8 },
    title: title ? { display: true, text: title, color: c.ink2, font: { size: 11 } } : { display: false }, ...o,
  });
  return { x: mk(x, xTitle, false), y: mk(y, yTitle, true) };
}

export function busy(msg, frac) {
  const el = $('#busy');
  if (!msg) { el.hidden = true; return; }
  el.hidden = false;
  $('#busy-msg').textContent = msg;
  $('#busy-bar').style.width = `${Math.round((frac || 0) * 100)}%`;
}

export const kpi = (label, value, sub = '', cls = '') => `<div class="kpi ${cls}"><span>${esc(label)}</span><b>${value}</b>${sub ? `<small>${sub}</small>` : ''}</div>`;
export const badge = (status, text) => `<span class="st ${status}">${{ pass: '✔', warn: '▲', fail: '✖', info: 'ℹ' }[status] || ''} ${esc(text)}</span>`;
