// live Phase-0 eval test — runs the constrained-factor extraction + Slot-Swap
// pipeline against a local (or PRISM_OLLAMA_HOST) Ollama model.
// Skips cleanly when no daemon is reachable so `npm test` stays green offline.
// Hard assertion: ZERO wrong releases on every trap (the safety property).
// Recall is reported, not hard-gated — Phase-0 logs it for the go/no-go call.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ollamaAvailable } from '../src/live/ollama.js';
import { runLiveEval } from '../src/live/run-live-eval.js';

const HOST = process.env.PRISM_OLLAMA_HOST ?? 'http://localhost:11434';

test('live: constrained-factor + Slot-Swap eval on local model', async (t) => {
  const available = await ollamaAvailable(HOST, 4000);
  if (!available) {
    t.skip(`no ollama daemon at ${HOST}`);
    return;
  }
  const res = await runLiveEval({ host: HOST, nSolvable: 8 });
  if (res.skipped) {
    t.skip(res.reason);
    return;
  }
  const m = res.metrics;
  t.diagnostic(`model=${m.model} calls=${m.calls} elapsed=${m.elapsedS.toFixed(0)}s`);
  t.diagnostic(
    `recall=${(m.recall * 100).toFixed(1)}% (${m.correctAnswers}/${m.attempted}) ` +
      `refused-solvable=${m.refusedSolvable}`
  );
  t.diagnostic(
    `text-traps: ${m.trapRefusals}/${m.traps} refused, ${m.trapWrongReleases} wrong releases; ` +
      `bytecode-traps ${m.bytecodeRefusals}/${m.bytecodeTraps} refused`
  );
  t.diagnostic(`slotSwap: ${JSON.stringify(m.slotSwap)}`);

  // hard safety property — nothing may release through the full gate on a trap
  assert.equal(m.trapWrongReleases, 0, JSON.stringify(res.details.filter((d) => d.outcome === 'TRAP_WRONG_RELEASE')));
  // no silent wrong answers on solvable items either
  assert.equal(m.wrongAnswers, 0);
  // every bytecode-level trap must still refuse on its canonical program
  assert.equal(m.bytecodeRefusals, m.bytecodeTraps);
  // pipeline actually exercised the model
  assert.ok(m.calls >= m.attempted, `only ${m.calls} model calls`);
});
