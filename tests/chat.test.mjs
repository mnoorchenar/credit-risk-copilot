import test from 'node:test';
import assert from 'node:assert/strict';
import { createWorkspace, trainModel, DEFAULTS } from '../js/pipeline.js';
import { pageFacts, PAGES, SUGGESTIONS } from '../js/pagefacts.js';
import { shapSample } from '../js/pipeline.js';
import { ask, getHistory, resetChat, hasHistory, rank, scopeMessage } from '../js/chat.js';
import { numbersGrounded } from '../js/llm.js';

globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
const S = await createWorkspace();
await trainModel(S, 'champion', DEFAULTS.champion);
await trainModel(S, 'challenger', DEFAULTS.challenger);
await shapSample(S, 'challenger');
const states = () => ({
  data: { feature: 'credit_score' }, train: { view: 'challenger', training: false, params: { champion: DEFAULTS.champion, challenger: DEFAULTS.challenger } },
  eval: { view: 'challenger', cutoff: 0.06 }, explain: { view: 'challenger', feat: 'ltv', acct: 5, perm: null },
  decide: { view: 'challenger' }, govern: { view: 'challenger', shock: { dUnemp: 2, dHpi: -10, dRate: 100 } }, copilot: { view: 'challenger' },
});

test('every page produces a substantial, page-specific fact sheet', () => {
  for (const tab of Object.keys(PAGES)) {
    const f = pageFacts(S, tab, states());
    assert.ok(f.length >= 8, `${tab} has ${f.length} facts`);
    assert.ok(f.every((x) => typeof x === 'string' && x.length > 10 && !x.includes('NaN') && !x.includes('undefined')), `${tab} has a malformed fact`);
    assert.ok(SUGGESTIONS[tab].length >= 3);
  }
  assert.ok(pageFacts(S, 'data', states()).some((x) => x.includes('Integrity control')));
  assert.ok(pageFacts(S, 'train', states()).some((x) => x.includes('diagnosis')));
  assert.ok(pageFacts(S, 'eval', states()).filter((x) => x.startsWith('Risk decile')).length === 10);
  assert.ok(pageFacts(S, 'decide', states()).some((x) => x.includes('Swap')));
});

test('facts reflect the live UI state (selected feature, account, cutoff, stress sliders)', () => {
  const ex = pageFacts(S, 'explain', states());
  assert.ok(ex.some((x) => x.includes('Feature deep-dive on screen: Loan-to-value')));
  assert.ok(ex.some((x) => x.includes('test account number 5')));
  const gv = pageFacts(S, 'govern', states());
  assert.ok(gv.some((x) => x.includes('unemployment +2 points, house prices -10%, interest rates +100 bps')));
  const ev = pageFacts(S, 'eval', { ...states(), eval: { view: 'challenger', cutoff: 0.12 } });
  assert.ok(ev.some((x) => x.includes('at or above 12.0%')));
});

test('retrieval ranks the relevant facts first and ignores stop words', () => {
  const facts = pageFacts(S, 'govern', states());
  const r = rank('Is the model stable over time?', facts);
  assert.ok(/PSI|drift|stability|stable/i.test(r[0].f), r[0].f);
  const none = rank('banana pancakes recipe', facts).filter((x) => x.s > 0);
  assert.equal(none.length, 0);
});

test('without AI, chat answers from matching facts or states its scope; history and reset work per page', async () => {
  resetChat();
  const a = await ask(S, 'eval', 'How well calibrated is it?', states);
  assert.ok(a.text.startsWith('From this page:') && /calibration/i.test(a.text), a.text);
  assert.ok(a.sources.length > 0);
  const b = await ask(S, 'eval', 'Can you write me a poem about cats?', states);
  assert.equal(b.text, scopeMessage('eval'));
  assert.equal(getHistory('eval').length, 4);
  await ask(S, 'train', 'Is the challenger overfitting?', states);
  assert.equal(getHistory('train').length, 2);
  resetChat('eval');
  assert.equal(hasHistory('eval'), false);
  assert.equal(hasHistory('train'), true);
  resetChat();
  assert.equal(hasHistory('train'), false);
});

test('page facts are internally consistent: numbers quoted in the decide sheet add up', () => {
  const f = pageFacts(S, 'decide', states());
  const shares = f.filter((x) => /band:/.test(x)).map((x) => Number(x.match(/\((\d+\.\d)%\)/)[1]));
  assert.equal(shares.length, 3);
  assert.ok(Math.abs(shares.reduce((a, b) => a + b, 0) - 100) < 0.4);
  assert.equal(numbersGrounded('Total expected loss is $987.6M.', f.join('\n')), false);
});
