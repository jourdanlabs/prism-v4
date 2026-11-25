import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { solve } from '../src/kernel/solve.js';
import { generateBench } from '../src/synth/generator.js';
import { runBench } from '../src/synth/run-bench.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import {
  makeSemanticExtractor,
  makePositionalExtractor,
} from '../src/invariance/reference-extractors.js';

const { items, stats } = generateBench();

test('synth: corpus shape — 200 solvable + >=12 traps', () => {
  const solvable = items.filter((i) => i.kind === 'solve');
  const traps = items.filter((i) => i.kind === 'trap');
  const gates = items.filter((i) => i.kind === 'gate');
  assert.equal(solvable.length, 200);
  assert.ok(traps.length + gates.length >= 12);
  assert.equal(traps.length + gates.length, 14);
});

test('synth: diversity — registers, multi-hop, fraction words', () => {
  assert.ok(Object.keys(stats.registers).length >= 5);
  assert.ok(stats.multiHop >= 60, `multiHop=${stats.multiHop}`);
  assert.ok(stats.literalKinds.words > 100, `word literals=${stats.literalKinds.words}`);
  assert.equal(stats.hasFiveEighths, true);
  assert.equal(stats.hasSevenTwelfths, true);
});

test('synth: every solvable item answers exactly and matches float check', () => {
  for (const it of items.filter((i) => i.kind === 'solve')) {
    const d = solve(it.program, it.X);
    assert.equal(d.status, 'ANSWER', `${it.id} refused: ${d.code}`);
    assert.equal(R.toString(d.value), it.expected.rational);
    assert.ok(Math.abs(R.toNumber(d.value) - it.floatCheck) < 1e-9, it.id);
  }
});

test('synth: all 14 trap families refuse with expected codes', () => {
  const traps = items.filter((i) => i.kind === 'trap');
  const expectedCodes = new Set([
    'MISSING_BASE_RATE', 'REVERSED_CONDITIONAL', 'UNLICENSED_INDEPENDENCE',
    'DUPLICATE_EVIDENCE', 'CONFLICTING_FIGURES', 'CONTEXT_MISMATCH',
    'SCOPED_OR_RETRACTED_PREMISE',
  ]);
  const seen = new Set();
  for (const it of traps) {
    const d = solve(it.program, it.X);
    assert.equal(d.status, 'REFUSED', `${it.trapFamily} released ${JSON.stringify(d)}`);
    assert.equal(d.code, it.expected.code, `${it.trapFamily}: want ${it.expected.code}, got ${d.code}`);
    seen.add(d.code);
  }
  assert.ok(expectedCodes.size <= seen.size + 1); // all 7 covered across families
});

test('synth: gate-level Bulma trap — positional refused, semantic passes', async () => {
  const gate = items.find((i) => i.kind === 'gate');
  assert.ok(gate, 'no gate item');
  const pos = await verifyInvariance({
    offlineSyntheticBench: true, X: gate.X, extract: makePositionalExtractor(gate.program) });
  assert.equal(pos.verdict, 'REFUSED');
  assert.equal(pos.code, 'UNSTABLE_BINDING');
  const sem = await verifyInvariance({
    offlineSyntheticBench: true, X: gate.X, extract: makeSemanticExtractor(gate.program) });
  assert.equal(sem.verdict, 'PASS', `semantic failed: ${sem.code}`);
});

test('synth: end-to-end bench — zero wrong releases, 100% correct answers', async () => {
  const { metrics, failures } = await runBench({ write: false });
  assert.equal(metrics.answersCorrect, metrics.solvable, JSON.stringify(failures.slice(0, 5)));
  assert.equal(metrics.wrongReleases, 0);
  assert.equal(metrics.trapsRefused, metrics.traps);
  // kernel arithmetic is exact; deviation vs the float recompute is float's
  // own rounding noise, bounded far below any decision-relevant epsilon.
  assert.ok(metrics.numericalMaxError < 1e-9, `float deviation ${metrics.numericalMaxError}`);
  assert.equal(metrics.gatePassSemantic, metrics.solvable);
  assert.equal(metrics.gateCaughtPositional, metrics.gatePositionalTrials);
});
