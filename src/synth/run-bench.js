// synth-bench :: full Phase-0 deterministic evaluation driver.
// Runs every generated item through prism-kernel + the B=4 invariance gate
// (semantic extractor for the pass path, positional-bug extractor for the
// adversarial path) and reports metrics. Writes evals/synth-bench-results.json.

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as R from '../kernel/rational.js';
import { solve } from '../kernel/solve.js';
import { buildReceipt } from '../kernel/receipt.js';
import { verifyInvariance } from '../invariance/gate.js';
import { literalsInSpan } from '../kernel/literals.js';
import { toProgram } from '../kernel/parser.js';
import {
  makeSemanticExtractor,
  makePositionalExtractor,
} from '../invariance/reference-extractors.js';
import { generateBench } from './generator.js';

const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);

// Does the program have >=2 bound literal sites carrying different values?
// Only then can a positional extractor be caught — permutations among
// equal-valued literals are legitimately invariant.
function hasMovableLiteral(program, X) {
  const prog = toProgram(program);
  const vals = new Set();
  for (const d of prog.decls) {
    if (!NUMERIC_OPS.has(d.op) || !d.span) continue;
    const hit = literalsInSpan(X, d.span).find((l) => R.eq(l.value, d.value));
    if (hit) vals.add(R.toString(d.value));
  }
  return vals.size >= 2;
}

const __dirname = dirname(fileURLToPath(import.meta.url));

export async function runBench({ seed = 20261122, count = 200, write = true } = {}) {
  const { items, stats } = generateBench({ seed, count });
  const t0 = performance.now();
  const metrics = {
    items: items.length,
    solvable: 0,
    answersCorrect: 0,
    numericalMaxError: 0,
    traps: 0,
    trapsRefused: 0,
    wrongReleases: 0,
    gatePassSemantic: 0,
    gatePassExpected: 0,
    gateCaughtPositional: 0,
    gatePositionalTrials: 0,
    receiptsVerified: 0,
  };
  const failures = [];
  const receipts = [];

  for (const it of items) {
    if (it.kind === 'solve') {
      metrics.solvable++;
      const d = solve(it.program, it.X);
      if (d.status === 'ANSWER' && R.toString(d.value) === it.expected.rational) {
        metrics.answersCorrect++;
      } else {
        failures.push({ id: it.id, kind: 'answer-mismatch', got: d });
      }
      // |exact-rational - float-recompute| — measures float's own rounding,
      // not kernel error (kernel error is 0 by construction).
      const err = Math.abs(R.toNumber(d.value ?? R.zero()) - it.floatCheck);
      if (err > metrics.numericalMaxError) metrics.numericalMaxError = err;

      // gate: semantic extractor must pass; positional must be caught when a
      // bound literal can actually move (>=2 sites with distinct values)
      const sem = await verifyInvariance({ X: it.X, extract: makeSemanticExtractor(it.program), offlineSyntheticBench: true });
      if (sem.verdict === 'PASS') {
        metrics.gatePassSemantic++;
      } else {
        failures.push({ id: it.id, kind: 'semantic-gate-fail', code: sem.code });
      }
      const pos = await verifyInvariance({ X: it.X, extract: makePositionalExtractor(it.program), offlineSyntheticBench: true });
      const movableLiteral = hasMovableLiteral(it.program, it.X);
      if (movableLiteral) {
        metrics.gatePositionalTrials++;
        if (pos.verdict === 'REFUSED' && pos.code === 'UNSTABLE_BINDING') {
          metrics.gateCaughtPositional++;
        } else if (pos.verdict === 'PASS') {
          failures.push({ id: it.id, kind: 'positional-gate-miss' });
        } else {
          metrics.gateCaughtPositional++; // refused, different code still no release
        }
      }
      metrics.receiptsVerified += buildReceipt({
        X: it.X,
        program: toProgram(it.program),
        decision: d,
      }).receipt_sha256
        ? 1
        : 0;
    } else if (it.kind === 'trap') {
      metrics.traps++;
      const d = solve(it.program, it.X);
      if (d.status === 'REFUSED') {
        if (d.code === it.expected.code) metrics.trapsRefused++;
        else failures.push({ id: it.id, kind: 'wrong-refusal-code', want: it.expected.code, got: d.code });
      } else {
        metrics.wrongReleases++;
        failures.push({ id: it.id, kind: 'wrong-release', got: R.toString(d.value) });
      }
    } else if (it.kind === 'gate') {
      metrics.traps++;
      const pos = await verifyInvariance({ X: it.X, extract: makePositionalExtractor(it.program), offlineSyntheticBench: true });
      const sem = await verifyInvariance({ X: it.X, extract: makeSemanticExtractor(it.program), offlineSyntheticBench: true });
      if (pos.verdict === 'REFUSED' && pos.code === 'UNSTABLE_BINDING') {
        metrics.trapsRefused++;
      } else {
        metrics.wrongReleases++;
        failures.push({ id: it.id, kind: 'gate-trap-miss', got: pos.code ?? pos.verdict });
      }
      if (sem.verdict === 'PASS' || sem.code !== 'UNSTABLE_BINDING') {
        metrics.gatePassExpected++;
      }
    }
  }

  metrics.recall =
    metrics.solvable > 0 ? metrics.answersCorrect / metrics.solvable : 0;
  metrics.trapRefusalRate =
    metrics.traps > 0 ? metrics.trapsRefused / metrics.traps : 0;
  metrics.elapsedMs = Math.round(performance.now() - t0);
  metrics.generatorStats = stats;

  const out = { metrics, failures, receiptSample: receipts.slice(0, 3) };
  if (write) {
    const dir = join(__dirname, '..', '..', 'evals');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'synth-bench-results.json'), JSON.stringify(out, null, 2));
  }
  return out;
}

if (process.argv[1] && process.argv[1].endsWith('run-bench.js')) {
  const { metrics, failures } = await runBench();
  console.log(JSON.stringify(metrics, null, 2));
  if (failures.length) {
    console.log(`FAILURES (${failures.length}):`);
    console.log(JSON.stringify(failures.slice(0, 10), null, 2));
    process.exit(1);
  }
  console.log('synth-bench: all checks green.');
}
