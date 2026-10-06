import test from 'node:test';
import assert from 'node:assert/strict';
import { F } from '../js/data.js';
import { createWorkspace, trainModel, DEFAULTS } from '../js/pipeline.js';
import { computeGovernance, buildFacts, templateMemo, modelCard } from '../js/governance.js';
import { bandTable, swapSet, policyApproves, recourse, reasons, stressTest } from '../js/explain.js';
import { shapRow } from '../js/ml.js';
import { contradicts, numbersGrounded } from '../js/llm.js';

globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
const S = await createWorkspace();
await trainModel(S, 'champion', DEFAULTS.champion);
await trainModel(S, 'challenger', DEFAULTS.challenger);

test('workspace: stratified split keeps one default rate and models reach a realistic AUC', () => {
  const r = (d) => d.y.reduce((a, b) => a + b, 0) / d.n;
  assert.ok(Math.abs(r(S.tr) - r(S.te)) < 0.002);
  for (const k of ['champion', 'challenger']) { const a = S.models[k].metrics.test.auc; assert.ok(a > 0.78 && a < 0.9, `${k} auc ${a}`); }
  assert.ok(S.models.challenger.metrics.test.auc > S.models.champion.metrics.test.auc - 0.01);
});

test('decision bands partition the book and loss maths is consistent', () => {
  const e = S.models.challenger; const rows = bandTable(e.preds.test, S.te.y, S.ead, 0.25, 0.03, 0.1);
  assert.equal(rows.reduce((a, r) => a + r.n, 0), S.te.n);
  assert.ok(Math.abs(rows.reduce((a, r) => a + r.share, 0) - 1) < 1e-9);
  assert.ok(rows[0].badRate < rows[1].badRate && rows[1].badRate < rows[2].badRate, 'bad rate must rise across bands');
});

test('swap-set analysis: same approval volume, cells add up, model avoids defaults', () => {
  const e = S.models.challenger; const pol = new Uint8Array(S.te.n);
  for (let i = 0; i < S.te.n; i++) pol[i] = policyApproves(S.te.X, i * F) ? 1 : 0;
  const sw = swapSet(S.te.y, pol, e.preds.test);
  const total = Object.values(sw.cells).reduce((a, c) => a + c[0], 0);
  assert.equal(total, S.te.n);
  assert.equal(sw.cells.both[0] + sw.cells.modelOnly[0], sw.approvals);
  assert.equal(sw.cells.both[0] + sw.cells.policyOnly[0], sw.approvals);
  assert.ok(sw.modelBadRate <= sw.policyBadRate, `model ${sw.modelBadRate} vs rules ${sw.policyBadRate}`);
});

test('reason codes and recourse are consistent with the model', () => {
  const e = S.models.challenger; const p = e.preds.test;
  let hi = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[hi]) hi = i;
  const { phi } = shapRow(e.model, S.te.X, hi * F, F);
  const rs = reasons(phi, S.te.X, hi * F);
  assert.ok(rs.up.length > 0 && rs.up[0].phi >= rs.up[rs.up.length - 1].phi);
  const rc = recourse(e.model, S.te.X, hi * F, 0.5);
  for (const o of rc.options) assert.ok(o.pd < 0.5, `recourse pd ${o.pd}`);
  let lo = 0; for (let i = 1; i < p.length; i++) if (p[i] < p[lo]) lo = i;
  assert.deepEqual(recourse(e.model, S.te.X, lo * F, 0.5).options, []);
});

test('stress test: worse macro conditions raise PD and loss, no shock changes nothing', () => {
  const e = S.models.challenger;
  const base = stressTest(e.model, S.te.X, S.te.n, S.ead, 0.25, {}, S.te.seg.region);
  const same = stressTest(e.model, S.te.X, S.te.n, S.ead, 0.25, { dUnemp: 0, dHpi: 0, dRate: 0 }, S.te.seg.region);
  const bad = stressTest(e.model, S.te.X, S.te.n, S.ead, 0.25, { dUnemp: 3, dHpi: -20, dRate: 200 }, S.te.seg.region);
  assert.ok(Math.abs(base.meanPd - same.meanPd) < 1e-12);
  assert.ok(bad.meanPd > base.meanPd * 1.05 && bad.expLoss > base.expLoss);
});

test('governance scorecard, fact sheet, memo and model card are complete and honest', () => {
  const g = computeGovernance(S, 'challenger');
  assert.equal(g.checks.length, 9);
  assert.equal(g.summary.pass + g.summary.warn + g.summary.fail, 9);
  assert.ok(g.maxShapErr < 1e-6);
  const facts = buildFacts(S, 'challenger').join('\n');
  assert.ok(facts.includes('Governance verdict:'));
  assert.ok(facts.includes(g.verdict));
  const memo = templateMemo(S, 'challenger');
  assert.ok(memo.includes('Recommendation:'));
  assert.ok(numbersGrounded(memo.replace(/\d+ of \d+/g, ''), facts + memo));
  assert.ok(modelCard(S, 'challenger').includes('Validation checks'));
});

test('semantic guard rejects "all checks passed" when warnings exist but allows honest wording', () => {
  assert.equal(contradicts('It passes all governance checks.', { warn: 2, fail: 0 }), true);
  assert.equal(contradicts('There are no warnings.', { warn: 1, fail: 0 }), true);
  assert.equal(contradicts('Approve with conditions; two checks carry warnings.', { warn: 2, fail: 0 }), false);
  assert.equal(contradicts('All checks passed.', { warn: 0, fail: 0 }), false);
});
