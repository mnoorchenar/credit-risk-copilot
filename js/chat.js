// Chat engine shared by the floating page assistant and the Copilot tab: per-page history, relevance ranking of the page's
// facts (a light retrieval step), and an answer path that prefers the AI but never lets it quote numbers or claims the app
// did not compute. No DOM access.
import { answer, llmLive, llmModel } from './llm.js';
import { PAGES, SUGGESTIONS, pageFacts } from './pagefacts.js';
import { computeGovernance } from './governance.js';

const store = new Map(); // tab -> [{ role: 'user' | 'bot', text, src?, sources? }]

export const getHistory = (tab) => store.get(tab) || [];
export function resetChat(tab) { if (tab) store.delete(tab); else store.clear(); }
export const hasHistory = (tab) => (store.get(tab) || []).length > 0;

export function scopeMessage(tab) {
  const ex = SUGGESTIONS[tab].slice(0, 2).map((q) => `"${q}"`).join(' or ');
  return tab === 'copilot'
    ? `I can only answer questions about this model's data, performance, explanations, decision cutoffs and validation results. Try: ${ex}.`
    : `I can only answer questions about the ${PAGES[tab]} page and this model's results. Try: ${ex}.`;
}

const STOP = new Set(['the', 'a', 'an', 'is', 'are', 'of', 'to', 'we', 'it', 'this', 'that', 'how', 'what', 'which', 'does', 'do', 'and', 'for', 'in', 'on', 'than', 'with', 'should', 'much', 'be', 'me', 'my', 'can', 'you', 'tell', 'about', 'there', 'any', 'why', 'was', 'were', 'has', 'have', 'if', 'i', 'show', 'give', 'explain', 'mean', 'means']);
const SYN = {
  overfit: ['overfitting', 'gap', 'memoris', 'validation'], overfitting: ['gap', 'memoris', 'validation'], underfit: ['underfitting'],
  learning: ['learning rate', 'step'], rate: ['rate'], depth: ['depth', 'interactions'], trees: ['trees', 'iterations'],
  accurate: ['auc', 'gini'], accuracy: ['auc', 'gini'], good: ['auc', 'gini', 'performance'], performance: ['auc', 'ks', 'gini'], ranking: ['auc'],
  calibrated: ['calibration'], calibration: ['calibration', 'slope'], stable: ['psi', 'drift', 'stability'], stability: ['psi', 'drift'], drift: ['psi', 'drift'],
  deploy: ['verdict', 'recommend', 'check'], deployment: ['verdict', 'check'], ready: ['verdict'], risk: ['risk', 'pd'], risky: ['riskiest', 'risk'],
  loss: ['loss', 'exposure'], cutoff: ['cutoff', 'decline', 'approve'], cutoffs: ['cutoff', 'band'], threshold: ['cutoff', 'operating'], decline: ['decline', 'declined'], approve: ['approve', 'approved', 'approval'],
  rules: ['rules', 'swap'], policy: ['rules', 'swap'], compare: ['versus', 'difference', 'champion', 'challenger'], comparison: ['versus', 'champion', 'challenger'],
  features: ['importance', 'shap', 'drivers'], drivers: ['shap', 'importance'], drive: ['shap', 'drivers', 'importance'], important: ['importance', 'shap'], importance: ['shap', 'permutation'],
  reasons: ['reasons', 'raising'], recourse: ['change the decision', 'would change'], change: ['would change', 'recourse'], account: ['account'],
  missing: ['missing', 'imputed', 'impute'], duplicates: ['duplicate'], integrity: ['integrity', 'control'], problems: ['integrity', 'control', 'fixed', 'flagged'],
  recession: ['stress', 'unemployment'], stress: ['stress', 'unemployment', 'prices'], segment: ['segment', 'approval'], fairness: ['segment', 'approval'],
  decile: ['decile'], riskiest: ['decile', 'riskiest'], roc: ['auc'], ks: ['ks'], brier: ['brier'], lift: ['lift', 'decile'],
};
const stem = (w) => (w.length > 5 ? w.slice(0, 5) : w);

/** Orders facts by overlap with the question (stable). `s` is the number of matching terms. */
export function rank(question, facts) {
  const words = String(question).toLowerCase().replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((w) => w.length > 1 && !STOP.has(w));
  const terms = [...new Set(words.flatMap((w) => [w, ...(SYN[w] || [])]))].map((t) => (t.includes(' ') ? t : stem(t)));
  return facts.map((f, i) => { const l = f.toLowerCase(); return { f, i, s: terms.reduce((a, t) => a + (l.includes(t) ? 1 : 0), 0) }; })
    .sort((a, b) => b.s - a.s || a.i - b.i);
}

/** Answers a question for a page, appends both turns to the page history and returns the bot turn. */
export async function ask(S, tab, question, getStates) {
  const q = String(question || '').trim().slice(0, 300);
  const hist = store.get(tab) || [];
  store.set(tab, hist);
  const prior = hist.slice(-6).filter((m) => m.text);
  hist.push({ role: 'user', text: q });
  const facts = pageFacts(S, tab, getStates());
  const ranked = rank(q, facts);
  const matches = ranked.filter((x) => x.s > 0);
  const sources = matches.slice(0, 3).map((x) => x.f);
  let text = null; let src = '';
  if (llmLive()) {
    const first = S.models.challenger ? 'challenger' : 'champion';
    const g = computeGovernance(S, first);
    text = await answer(q, ranked.map((x) => x.f), { warn: g.summary.warn, fail: g.summary.fail }, { history: prior, page: tab === 'copilot' ? 'Copilot' : PAGES[tab], scope: scopeMessage(tab) });
    if (text) src = `AI · ${llmModel()} · numbers verified`;
  }
  if (!text) {
    if (!sources.length) { text = scopeMessage(tab); src = 'Outside this page'; }
    else { text = `From this page: ${sources.slice(0, 2).join(' ')}`; src = llmLive() ? 'AI answer rejected (unverified numbers or an overstated claim); showing the matching facts' : 'AI not connected: showing the facts that match your question'; }
  }
  const turn = { role: 'bot', text, src, sources };
  hist.push(turn);
  return turn;
}
