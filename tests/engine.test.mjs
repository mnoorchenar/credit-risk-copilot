import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createModel, parseRules, validateQuery, runQuery, riskScore, numbersGrounded, SCENARIO_NONE, SCENARIO_SEVERE } from '../js/engine.js';

const load = (p) => JSON.parse(readFileSync(new URL(`../data/${p}`, import.meta.url), 'utf8'));
const portfolio = load('portfolio.json');
const model = createModel(portfolio, load('cities.json'));
const state = { scenario: SCENARIO_NONE };
const parse = (t, ctx) => parseRules(t, model, ctx);
const id = (short) => model.base.find((r) => r.short === short).id;

test('JS risk score matches the Python build for every region', () => {
  for (const r of portfolio.regions) assert.equal(riskScore(r, portfolio.meta.drivers), r.risk, r.short);
});

test('worst credit risk question ranks Toronto first', () => {
  const q = parse('Which areas have the worst credit risk score?');
  assert.equal(q.intent, 'rank');
  assert.equal(q.metric, 'risk');
  const res = runQuery(q, model, state);
  assert.equal(res.rows[0].name, 'Toronto');
  assert.equal(res.rows.length, 5);
});

test('top N and metric parsing', () => {
  const q = parse('show me the top 3 regions by delinquency');
  assert.equal(q.metric, 'dpd90');
  assert.equal(q.n, 3);
  assert.equal(runQuery(q, model, state).rows.length, 3);
});

test('best / safest flips the order', () => {
  const res = runQuery(parse('which are the safest regions'), model, state);
  assert.equal(res.rows[0].name, 'Huron Perth');
});

test('compound filters: high LTV and rising delinquency', () => {
  const q = parse('which areas have high LTV and rising delinquency?');
  assert.equal(q.intent, 'rank');
  assert.ok(q.filters.some((f) => f.metric === 'ltv' && f.op === 'worse_third'));
  assert.ok(q.filters.some((f) => f.metric === 'dpd_trend' && f.op === '>'));
  const res = runQuery(q, model, state);
  for (const r of res.rows) {
    const row = model.base.find((b) => b.id === r.id);
    assert.ok(row.ltv >= 66, `${r.name} ltv`);
  }
});

test('numeric threshold filter', () => {
  const q = parse('regions with LTV above 68');
  assert.deepEqual(q.filters[0], { metric: 'ltv', op: '>', value: 68 });
  const res = runQuery(q, model, state);
  assert.ok(res.rows.length > 0);
  for (const r of res.rows) assert.ok(r.value > 68);
});

test('region groups scope the ranking', () => {
  const q = parse('worst risk in northern ontario');
  assert.equal(q.scope, 'north');
  const res = runQuery(q, model, state);
  for (const r of res.rows) assert.ok([2226, 2247, 2249, 2261, 2262, 7654].includes(r.id));
});

test('profile, city names and selected-region context', () => {
  assert.equal(parse('how is Peel doing').intent, 'profile');
  assert.deepEqual(parse('tell me about Mississauga').regions, [id('Peel')]);
  const q = parse('why is the score high here', { regionId: id('Toronto') });
  assert.equal(q.intent, 'drivers');
  assert.deepEqual(q.regions, [id('Toronto')]);
  assert.equal(parse('trend of delinquency here', { regionId: id('Ottawa') }).intent, 'trend');
});

test('compare', () => {
  const q = parse('compare Toronto and Ottawa');
  assert.equal(q.intent, 'compare');
  assert.equal(runQuery(q, model, state).regionRows.length, 2);
});

test('stress scenarios parse and raise risk', () => {
  let q = parse('what if rates rise 200 bps and unemployment goes up 2 points');
  assert.equal(q.intent, 'stress');
  assert.equal(q.scenario.rate_bps, 200);
  assert.equal(q.scenario.unemp_pp, 2);
  q = parse('stress test a 15% drop in house prices');
  assert.equal(q.scenario.hpi_pct, -15);
  q = parse('run a stress test');
  assert.deepEqual(q.scenario, SCENARIO_SEVERE);
  const res = runQuery(q, model, state);
  assert.ok(res.kpis[0].to > res.kpis[0].from);
  assert.ok(res.rows[0].value > 0);
  assert.equal(parse('reset the scenario').intent, 'reset');
});

test('overview and help', () => {
  assert.equal(parse('give me an overall portfolio summary').intent, 'overview');
  assert.equal(parse('help').intent, 'help');
  assert.equal(parse('banana pancakes').intent, 'help');
});

test('validateQuery rejects junk and clamps LLM output', () => {
  assert.equal(validateQuery({ intent: 'drop table' }, model), null);
  assert.equal(validateQuery({ intent: 'compare', regions: ['Toronto'] }, model), null);
  const v = validateQuery({ intent: 'stress', scenario: { rate_bps: 9999, unemp_pp: -50, hpi_pct: -99 } }, model);
  assert.deepEqual(v.scenario, { rate_bps: 400, unemp_pp: -2, hpi_pct: -40 });
  const r = validateQuery({ intent: 'rank', metric: 'nonsense', n: 99, filters: [{ metric: 'ltv', op: 'DROP', value: 1 }] }, model);
  assert.equal(r.metric, 'risk');
  assert.equal(r.n, 10);
  assert.equal(r.filters.length, 0);
  const c = validateQuery({ intent: 'compare', regions: ['Toronto', 'Peel'] }, model);
  assert.equal(c.regions.length, 2);
});

test('numbersGrounded flags invented figures', () => {
  assert.ok(numbersGrounded('Toronto is at 71.0 with 8.7B', 'Toronto 71.0 balance $8.7B'));
  assert.ok(!numbersGrounded('Toronto is at 83', 'Toronto 71.0'));
});
