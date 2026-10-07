// Floating "Ask about this page" assistant plus the shared message renderer used by the Copilot tab.
import { $, esc } from './common.js';
import { ask, getHistory, resetChat, hasHistory, scopeMessage } from './chat.js';
import { PAGES, SUGGESTIONS, pageFacts } from './pagefacts.js';
import { llmLive, llmModel } from './llm.js';

let cfg = null; // { getS, getStates, getTab }
let open = false;
let pending = false;
let tab = 'data';

const ICON = '<svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true"><path fill="currentColor" d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H9l-4.2 3.5a.6.6 0 0 1-1-.46V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z"/><circle cx="8" cy="10.5" r="1.3" fill="#fff"/><circle cx="12" cy="10.5" r="1.3" fill="#fff"/><circle cx="16" cy="10.5" r="1.3" fill="#fff"/></svg>';

/** Renders a page's conversation into a container (used by the widget and the Copilot tab). */
export function renderMessages(el, t, { busy = false } = {}) {
  const hist = getHistory(t);
  el.innerHTML = hist.map((m, i) => m.role === 'user'
    ? `<div class="m u">${esc(m.text)}</div>`
    : `<div class="m b">${esc(m.text)}<small>${esc(m.src || '')}</small>${m.sources && m.sources.length && !/Outside/.test(m.src || '') ? `<details class="srcs"><summary>Based on ${m.sources.length} fact${m.sources.length > 1 ? 's' : ''} from this page</summary><ul>${m.sources.map((s) => `<li>${esc(s)}</li>`).join('')}</ul></details>` : ''}</div>`).join('')
    + (busy ? '<div class="m b"><span class="think"><i></i><i></i><i></i></span></div>' : '');
  el.scrollTop = el.scrollHeight;
}

function paint() {
  if (!cfg) return;
  const el = $('#cw-msgs');
  $('#cw-title').textContent = `Ask about ${PAGES[tab]}`;
  $('#cw-sub').textContent = llmLive() ? `AI · ${llmModel().split('/').pop()}` : 'Answers from this page\'s facts · connect AI for written answers';
  const hist = getHistory(tab);
  if (!hist.length && !pending) {
    el.innerHTML = `<div class="m b">Hi! I can see what is on the <b>${esc(PAGES[tab])}</b> page right now, including your current selections, and I answer only from those numbers. Ask me anything about it.<small>Tip: change a slider or pick an account, then ask again.</small></div>`;
  } else renderMessages(el, tab, { busy: pending });
  $('#cw-sg').innerHTML = SUGGESTIONS[tab].map((s) => `<button type="button" class="sg">${esc(s)}</button>`).join('');
  $('#cw-reset').disabled = !hasHistory(tab) || pending;
  $('#cw-send').disabled = pending;
}

async function send(text) {
  const q = String(text || '').trim();
  if (!q || pending || !cfg) return;
  const S = cfg.getS(); if (!S) return;
  pending = true;
  const t = tab;
  paintPendingUser(q);
  try { await ask(S, t, q, cfg.getStates); } finally { pending = false; }
  if (t === tab) paint();
}

function paintPendingUser(q) {
  const el = $('#cw-msgs');
  if (!getHistory(tab).length) el.innerHTML = '';
  el.insertAdjacentHTML('beforeend', `<div class="m u">${esc(q)}</div><div class="m b"><span class="think"><i></i><i></i><i></i></span></div>`);
  el.scrollTop = el.scrollHeight;
  $('#cw-send').disabled = true; $('#cw-reset').disabled = true;
}

function setOpen(v) {
  open = v;
  $('#chatw').hidden = !v;
  $('#fab').setAttribute('aria-expanded', String(v));
  $('#fab').classList.toggle('on', v);
  if (v) { paint(); setTimeout(() => $('#cw-q').focus(), 50); }
}

export function setTab(t) {
  tab = t;
  const fab = $('#fab');
  if (!fab) return;
  fab.hidden = t === 'copilot';
  if (t === 'copilot' && open) setOpen(false);
  if (open) paint();
}

export function refresh() { if (open) paint(); }

export function init(config) {
  cfg = config;
  const root = document.createElement('div');
  root.innerHTML = `<button type="button" class="fab" id="fab" aria-label="Ask about this page" aria-expanded="false" aria-controls="chatw" title="Ask about this page" hidden>${ICON}<span class="fab-tip">Ask about this page</span></button>
  <section class="chatw" id="chatw" role="dialog" aria-label="Page assistant" hidden>
    <header class="cw-h"><div><b id="cw-title">Ask about this page</b><small id="cw-sub"></small></div>
      <div class="cw-btns"><button type="button" class="btn sm" id="cw-reset" title="Clear this page's conversation">↻ Reset</button><button type="button" class="cw-x" id="cw-x" aria-label="Close chat">×</button></div></header>
    <div class="cw-msgs msgs" id="cw-msgs" aria-live="polite"></div>
    <div class="cw-sg" id="cw-sg"></div>
    <form class="cw-form" id="cw-form" autocomplete="off"><input type="text" id="cw-q" maxlength="240" placeholder="Ask about the results on this page…" aria-label="Your question"><button class="btn primary" id="cw-send" type="submit">Send</button></form>
    <footer class="cw-f"><button type="button" class="link" id="cw-all">Clear all pages</button><details id="cw-see"><summary>What can I see?</summary><div class="cw-facts" id="cw-facts"></div></details></footer>
  </section>`;
  document.body.append(...root.children);
  $('#fab').addEventListener('click', () => setOpen(!open));
  $('#cw-x').addEventListener('click', () => setOpen(false));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && open) setOpen(false); });
  $('#cw-form').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#cw-q').value; $('#cw-q').value = ''; send(v); });
  $('#cw-sg').addEventListener('click', (e) => { const b = e.target.closest('.sg'); if (b) send(b.textContent); });
  $('#cw-reset').addEventListener('click', () => { resetChat(tab); paint(); document.dispatchEvent(new CustomEvent('chat-reset', { detail: { tab } })); });
  $('#cw-all').addEventListener('click', () => { resetChat(); paint(); document.dispatchEvent(new CustomEvent('chat-reset', { detail: { tab: null } })); });
  $('#cw-see').addEventListener('toggle', () => {
    if (!$('#cw-see').open) return;
    const S = cfg.getS();
    const facts = S ? pageFacts(S, tab, cfg.getStates()) : [];
    $('#cw-facts').innerHTML = `<p class="note">${facts.length} facts from the ${esc(PAGES[tab])} page, as of now. The assistant may only use these.</p><ul>${facts.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>`;
  });
  document.addEventListener('ai-changed', refresh);
}

export function show() { const f = $('#fab'); if (f) f.hidden = tab === 'copilot'; }
export { scopeMessage };
