// bulma-live-probe :: replay Bulma's independent-gate passages through the
// live Ollama extractor + full B=4 gate (Slot-Swap >= 2 per the R2 contract).
// Expected outcomes come from Bulma's hand-derived oracles.
//   node src/live/run-bulma-probe.js [--model gemma4:26b]

import { verifyInvariance } from '../invariance/gate.js';
import { makeOllamaExtractor } from './extractor.js';
import { ollamaAvailable } from './ollama.js';
import * as R from '../kernel/rational.js';
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const PASSAGES = [
  {
    id: 'parenthetical',
    expect: { answer: '27/35' },
    X: 'Every pallet was either Cedar or Quartz. Cedar (the 40% share belongs to Quartz) made up 60% of pallets. The beacon lit for nine tenths of Cedar pallets and eight twentieths of Quartz pallets. This pallet\u2019s beacon lit. What is the probability that it was Cedar?',
  },
  {
    id: 'multihop',
    expect: { answer: '7/16' },
    X: 'Every intake belongs to exactly one of two lanes, Amber or Indigo. The Amber lane accounts for 25%, and Indigo accounts for 75%. Amber intakes trigger a chime with probability 70% and a flash with probability 60%. Indigo intakes trigger a chime with probability 30% and a flash with probability 60%. The chime and flash are independent within each lane. This intake triggered both the chime and the flash. What is the probability it belongs to Amber?',
  },
  {
    id: 'negative',
    expect: { answer: '1/7' },
    X: 'Each container is either Larch or Basalt. Larch accounts for one third of all containers and Basalt for two thirds. The siren sounds for three quarters of Larch containers and one quarter of Basalt containers. The siren remained silent for the container in front of us. What is the probability it is Larch?',
  },
  {
    id: 'retracted',
    expect: { refuse: 'SCOPED_OR_RETRACTED_PREMISE' },
    X: 'Every vial is Elm or Flax. The log read, \u201cElm makes up 75% of vials and Flax 25%.\u201d The counts came from a tabletop exercise and were struck from the record. The lamp lights for 80% of Elm vials and 20% of Flax vials. This vial\u2019s lamp lit. What is the probability it is Elm?',
  },
  {
    id: 'reversed',
    expect: { refuse: 'REVERSED_CONDITIONAL' },
    X: 'Every parcel is Flint or Moss. Flint makes up 30% and Moss 70% of the parcels. Sort the parcels whose bell rang into a pile: 80% of that pile is Flint, and 20% is Moss. The light shines for 50% of Flint parcels and 10% of Moss parcels. No joint rates are available. This parcel\u2019s bell rang and its light shone. What is the probability it is Flint given both signals?',
  },
  {
    id: 'negated-license',
    expect: { refuse: 'UNLICENSED_INDEPENDENCE' },
    X: 'Every crate belongs to Saffron or Teal. Saffron makes up 25% of crates and Teal 75%. A Saffron crate triggers the tone with probability 80% and the blink with probability 60%. A Teal crate triggers the tone with probability 20% and the blink with probability 40%. The tone and blink are not conditionally independent within either cohort. This crate triggered both the tone and blink. What is the probability it is Saffron?',
  },
  {
    id: 'mixed-populations',
    expect: { refuse: 'CONTEXT_MISMATCH' },
    X: 'Every South intake is Opal or Jade. In South, Opal accounts for 20% and Jade 80%. The only alarm figures come from North: there, 90% of Opal intakes and 30% of Jade intakes trigger the alarm. A South intake triggered the alarm. What is the probability that the South intake is Opal?',
  },
  {
    id: 'ash-fir-direction',
    expect: { refuseAny: true },
    X: 'Every bead is Ash or Fir. The lot is 40% Ash and 60% Fir. Filter the lot to beads whose tag flashed: 80% of those beads are Ash and 20% are Fir. What is the chance a randomly selected bead\u2019s tag flashes?',
  },
];

// Round-2 fresh passages (BULMA-GATE-PRISM-PHASE0-R2-2a36542). The two R2
// wrong releases were zero-support self-conditioning releasing 1/1 and a
// collapsed day/night CtxId releasing 1/2; both must now refuse typed.
// linked-event-target's oracle answer is 1/1, but the signals are
// deterministically wired to the same switch — the factor grammar has no
// linkage decl, so a refusal is the fail-closed outcome (never PASS 7/11).
const PASSAGES_R2 = [
  {
    id: 'comparison-clause',
    expect: { answer: '28/31' },
    X: 'Every sampler is either Garnet or Sable. Sable, compared to the 30% share for Garnet, accounts for 70% of samplers. The lamp glows for 80% of Sable samplers and for 20% of Garnet samplers. This sampler\u2019s lamp glowed. What is the probability it is Sable?',
  },
  {
    id: 'observed-negative',
    expect: { answer: '1/4' },
    X: 'Every crate is either Birch or Juniper. Birch accounts for 40% of crates and Juniper accounts for 60%. The lamp lights for 75% of Birch crates and 50% of Juniper crates. This crate was observed with its lamp dark. What is the probability the crate is Birch?',
  },
  {
    id: 'linked-event-target',
    expect: { answer: '1/1', refuseOk: true },
    X: 'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. The lamp lights for 80% of Ash panels and 20% of Fir panels. The buzzer sounds for 80% of Ash panels and 20% of Fir panels. The lamp and buzzer are wired to the same switch and always activate together on the same panel. This panel was observed with its lamp lit. What is the probability its buzzer sounded?',
  },
  {
    id: 'zero-support-self-conditioning',
    expect: { refuse: 'DEGENERATE_PARTITION' },
    X: 'Every panel is either Rowan or Maple. Rowan accounts for 40% of panels and Maple accounts for 60%. The lamp lights for 0% of Rowan panels and 0% of Maple panels. This panel was observed with its lamp lit. Given that this panel\u2019s lamp lit, what is the probability its lamp lit?',
  },
  {
    id: 'active-voice-reversal',
    expect: { refuse: 'INCOMPLETE_FACTORIZATION' },
    X: 'Every card is either Mica or Slate. Mica accounts for 40% of all cards and Slate accounts for 60% of all cards. In the collection of cards that carry the seal, Mica accounts for 80% and Slate accounts for 20%. What is the probability that a randomly selected card carries the seal?',
  },
  {
    id: 'unverified-independence',
    expect: { refuse: 'UNLICENSED_INDEPENDENCE' },
    X: 'Every meter is either Coral or Flint. Coral accounts for 40% of meters and Flint accounts for 60%. The hum occurs for 80% of Coral meters and 20% of Flint meters. The blink occurs for 70% of Coral meters and 30% of Flint meters. Conditional independence of the hum and blink remains an open question within each type. This meter was observed to hum and blink. What is the probability it is Coral?',
  },
  {
    id: 'lowercase-populations',
    expect: { refuse: 'CONTEXT_MISMATCH' },
    X: 'Every day shift item is either Pebble or Moss. On the day shift, Pebble accounts for 25% of items and Moss accounts for 75%. On the night shift, the alarm sounds for 90% of Pebble items and 30% of Moss items. A day shift item was observed with its alarm sounding. What is the probability this day shift item is Pebble?',
  },
];

export async function runBulmaProbe({
  host = 'http://localhost:11434',
  model = 'gemma4:26b',
  maxSwaps = 2,
  round = 'r1',
  write = true,
} = {}) {
  const available = await ollamaAvailable(host);
  if (!available) {
    return { skipped: true, reason: `no ollama at ${host}` };
  }
  const chosen = available.includes(model) ? model : available[0];
  const extract = makeOllamaExtractor({ model: chosen, host });
  const results = [];
  const metrics = { correctReleases: 0, wrongReleases: 0, refusals: 0, extractionFailures: 0, oracleMisses: [] };
  const passages = round === 'r2' ? PASSAGES_R2 : round === 'all' ? [...PASSAGES, ...PASSAGES_R2] : PASSAGES;
  for (const p of passages) {
    let g;
    try {
      g = await verifyInvariance({ X: p.X, extract, maxSwaps });
    } catch (e) {
      g = { verdict: 'EXTRACTION_FAILURE', code: 'EXTRACTION_FAILURE', detail: e.message };
    }
    const released = g.verdict === 'PASS' && g.decision?.status === 'ANSWER';
    const value = released ? R.toString(g.decision.value) : null;
    let outcome;
    if (released) {
      if (p.expect.answer && value === p.expect.answer) {
        outcome = 'correct';
        metrics.correctReleases++;
      } else {
        outcome = 'WRONG_RELEASE';
        metrics.wrongReleases++;
        metrics.oracleMisses.push(p.id);
      }
    } else {
      metrics.refusals++;
      if (p.expect.answer && !p.expect.refuseOk) {
        outcome = 'refused-solvable';
        metrics.oracleMisses.push(p.id);
      } else if (p.expect.refuseOk) {
        // oracle answer exists but cannot be derived without modeling the
        // linkage — a typed refusal is the fail-closed outcome
        outcome = `refused-safe(${g.code ?? g.verdict})`;
      } else {
        const want = p.expect.refuse;
        outcome = want && g.code === want ? 'typed-refusal' : want ? `refused(${g.code})` : 'refused';
      }
    }
    results.push({ id: p.id, expect: p.expect, verdict: g.verdict, code: g.code ?? g.decision?.code ?? null, value, outcome });
  }
  const out = { model: chosen, maxSwaps, metrics, results };
  if (write) {
    writeFileSync(join(__dirname, '../../evals/bulma-live-probe.json'), JSON.stringify(out, null, 2));
  }
  return out;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const modelIdx = process.argv.indexOf('--model');
  const model = modelIdx >= 0 ? process.argv[modelIdx + 1] : 'gemma4:26b';
  const roundIdx = process.argv.indexOf('--round');
  const round = roundIdx >= 0 ? process.argv[roundIdx + 1] : 'r1';
  const res = await runBulmaProbe({ model, round });
  if (res.skipped) {
    console.log(`skipped: ${res.reason}`);
    process.exit(0);
  }
  for (const r of res.results) {
    console.log(`${r.id.padEnd(20)} ${r.outcome.padEnd(18)} verdict=${r.verdict} code=${r.code} value=${r.value}`);
  }
  console.log(JSON.stringify(res.metrics));
}
