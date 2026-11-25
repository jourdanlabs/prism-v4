// live Phase-0 eval :: drive a local open-weights model through the real
// pipeline — constrained factor extraction -> prism-kernel -> B=4 Slot-Swap
// gate — on a slice of synth-bench. Zero vendor taint; deterministic seed.
//
// Metrics reported (ARCH-SPEC §6 go/no-go):
//   - wrongReleases   (must be 0 on every trap — hard safety property)
//   - answerable recall (target >= 60% vs October's 0-13%)
//   - slotSwapVerdicts on the Bulma item + a solvable item

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as R from '../kernel/rational.js';
import { solve } from '../kernel/solve.js';
import { toProgram } from '../kernel/parser.js';
import { verifyInvariance } from '../invariance/gate.js';
import { ollamaAvailable } from './ollama.js';
import { makeOllamaExtractor } from './extractor.js';
import { generateBench } from '../synth/generator.js';

// Traps whose refusal lives only in the mutated *program*: the rendered X is
// a clean solvable passage, so a faithful extractor answering it is CORRECT.
// These are verified on their canonical bytecode (deterministically), not on
// live model output.
const PROGRAM_LEVEL_TRAPS = new Set([
  'duplicate_sensor',
  'conflicting_figures',
  'context_mismatch',
  'stated_vs_computed',
  // text asserts independence in prose; only the stripped *program* lacks
  // COND_INDEP — a correct extractor should answer the text.
  'unlicensed_independence',
]);

const __dirname = dirname(fileURLToPath(import.meta.url));
// non-thinking models first: thinking-model variants can exhaust num_predict
// inside message.thinking and return empty content under format:'json'.
const PREFERRED_MODELS = [
  'gemma4:26b',
  'twil-8b-balanced:latest',
  'twil-8b-verbose:latest',
  'hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M',
];

export async function runLiveEval({
  host = 'http://localhost:11434',
  model = null,
  nSolvable = 8,
  seed = 20261122,
  write = true,
} = {}) {
  const available = await ollamaAvailable(host);
  if (!available) {
    return { skipped: true, reason: `no ollama at ${host}` };
  }
  const chosen =
    model ??
    process.env.PRISM_OLLAMA_MODEL ??
    PREFERRED_MODELS.find((m) => available.includes(m)) ??
    available[0];

  const { items } = generateBench({ seed, count: 200 });
  // deterministic diverse slice of solvable items
  const solvable = items.filter((i) => i.kind === 'solve');
  const step = Math.floor(solvable.length / nSolvable);
  const slice = Array.from({ length: nSolvable }, (_, i) => solvable[i * step]);
  const allTraps = items.filter((i) => i.kind === 'trap');
  const textTraps = allTraps.filter((t) => !PROGRAM_LEVEL_TRAPS.has(t.trapFamily));
  const bytecodeTraps = allTraps.filter((t) => PROGRAM_LEVEL_TRAPS.has(t.trapFamily));
  const gateItem = items.find((i) => i.kind === 'gate');

  const calls = [];
  const extract = makeOllamaExtractor({ model: chosen, host, calls });
  const t0 = Date.now();
  const metrics = {
    model: chosen,
    host,
    attempted: 0,
    correctAnswers: 0,
    wrongAnswers: 0, // gate-PASS answer that disagrees — wrong release!
    refusedSolvable: 0, // gate/kernel refused a solvable item (safe, costs recall)
    traps: textTraps.length, // text-legible traps only; bytecode traps separate
    trapRefusals: 0,
    trapWrongReleases: 0,
    bytecodeTraps: bytecodeTraps.length,
    bytecodeRefusals: 0,
    slotSwap: {},
  };
  const details = [];

  // A RELEASE requires the full B=4 gate to PASS with an ANSWER. Everything
  // else — kernel refusal, gate refusal, extractor crash — is a safe failure.
  const gateRun = async (it, maxSwaps) => {
    try {
      return await verifyInvariance({ X: it.X, extract, maxSwaps });
    } catch (e) {
      return { verdict: 'EXTRACTION_FAILURE', code: 'EXTRACTION_FAILURE', detail: e.message };
    }
  };

  for (const it of slice) {
    metrics.attempted++;
    const g = await gateRun(it, 2); // main loop MUST exercise Slot-Swap (>=2)
    const d = g.decision;
    if (g.verdict === 'PASS' && d?.status === 'ANSWER') {
      if (R.toString(d.value) === it.expected.rational) {
        metrics.correctAnswers++;
        details.push({ id: it.id, outcome: 'correct', value: R.toString(d.value) });
      } else {
        metrics.wrongAnswers++;
        details.push({
          id: it.id, outcome: 'WRONG_ANSWER',
          got: R.toString(d.value), want: it.expected.rational,
        });
      }
    } else {
      metrics.refusedSolvable++;
      details.push({
        id: it.id,
        outcome: g.verdict === 'PASS' ? 'kernel-refused' : `gate-${g.verdict}`,
        code: g.code ?? d?.code ?? null,
      });
    }
  }

  for (const it of textTraps) {
    const g = await gateRun(it, 2); // main loop MUST exercise Slot-Swap (>=2)
    if (g.verdict === 'PASS' && g.decision?.status === 'ANSWER') {
      metrics.trapWrongReleases++;
      details.push({
        id: it.id, outcome: 'TRAP_WRONG_RELEASE', family: it.trapFamily,
        got: R.toString(g.decision.value),
      });
    } else {
      metrics.trapRefusals++;
      details.push({
        id: it.id, outcome: 'refused', family: it.trapFamily,
        code: g.code ?? g.decision?.code ?? null, want: it.expected.code,
      });
    }
  }

  // Program-level traps: X is clean text — the refusal lives in the emitted
  // bytecode. Verify the kernel refuses the canonical program (no model call).
  for (const it of bytecodeTraps) {
    const prog = toProgram(it.program, it.X.length);
    const d = solve(prog, it.X);
    if (d.status === 'REFUSED') {
      metrics.bytecodeRefusals++;
      details.push({ id: it.id, outcome: 'bytecode-refused', code: d.code, family: it.trapFamily });
    } else {
      details.push({ id: it.id, outcome: 'BYTECODE_TRAP_MISSED', family: it.trapFamily });
    }
  }

  // Slot-Swap probe on the Bulma gate item with the live extractor.
  // An extractor throw on the original text is a safe failure (nothing
  // releases) — record it rather than crash the eval.
  if (gateItem) {
    try {
      const g = await verifyInvariance({ X: gateItem.X, extract, maxSwaps: 4 });
      metrics.slotSwap.bulma = { verdict: g.verdict, code: g.code ?? null };
    } catch (e) {
      metrics.slotSwap.bulma = { verdict: 'EXTRACTION_FAILURE', error: e.message.slice(0, 200) };
    }
    // and one solvable item for the pass side
    const s0 = slice[0];
    try {
      const g2 = await verifyInvariance({ X: s0.X, extract, maxSwaps: 2 });
      metrics.slotSwap.solvable = { id: s0.id, verdict: g2.verdict, code: g2.code ?? null };
    } catch (e) {
      metrics.slotSwap.solvable = { id: s0.id, verdict: 'EXTRACTION_FAILURE', error: e.message.slice(0, 200) };
    }
  }

  metrics.recall = metrics.attempted > 0 ? metrics.correctAnswers / metrics.attempted : 0;
  metrics.trapRefusalRate = metrics.trapRefusals / metrics.traps;
  metrics.elapsedS = (Date.now() - t0) / 1000;
  metrics.calls = calls.length;

  const out = { skipped: false, metrics, details };
  if (write) {
    const dir = join(__dirname, '..', '..', 'evals');
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, 'live-results.json'), JSON.stringify(out, null, 2));
  }
  return out;
}

if (process.argv[1] && process.argv[1].endsWith('run-live-eval.js')) {
  const res = await runLiveEval();
  if (res.skipped) {
    console.log('live eval skipped:', res.reason);
    process.exit(0);
  }
  console.log(JSON.stringify(res.metrics, null, 2));
  console.log(JSON.stringify(res.details, null, 2).slice(0, 4000));
}
