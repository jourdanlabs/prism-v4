// Use the same generator, default seed, and row count as npm run bench.
import { solve } from '../../../src/kernel/solve.js';
import { solveCredal } from '../../../src/kernel/credal.js';
import { toJSON, toProgram } from '../../../src/kernel/parser.js';
import { toString } from '../../../src/kernel/rational.js';
import { verifyInvariance } from '../../../src/invariance/gate.js';
import { makePositionalExtractor } from '../../../src/invariance/reference-extractors.js';
import { generateBench } from '../../../src/synth/generator.js';

function intervalStrings(interval) {
  return Object.fromEntries(Object.entries(interval).map(([key, value]) => [
    key,
    typeof value === 'string' ? value : toString(value),
  ]));
}

function verdict(decision) {
  if (decision.status === 'ANSWER') {
    const { status, value, interval, target, via, credal } = decision;
    return { status, value: toString(value), interval: intervalStrings(interval),
      target, via, support: [credal.eMin, credal.eMax] };
  }
  const { status, code, details } = decision;
  return { status, code, ...(details.interval ? { interval: intervalStrings(details.interval) } : {}) };
}

const { items } = generateBench({ seed: 20261122, count: 200 });
for (const item of items) {
  const { id, kind, program, X, expected, trapFamily } = item;
  const kernel = verdict(solve(program));
  const withText = verdict(solve(program, X));
  let gate;
  if (kind === 'gate') {
    const decision = await verifyInvariance({
      X, extract: makePositionalExtractor(program), offlineSyntheticBench: true,
    });
    gate = { status: decision.verdict, code: decision.code };
  }
  console.log(JSON.stringify({
    id, kind, trapFamily, program, ast: toJSON(toProgram(program)),
    expected, kernel, withText, gate, polytope: verdict(solveCredal(program)),
  }));
}
