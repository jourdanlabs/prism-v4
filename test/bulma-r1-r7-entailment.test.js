// bulma R1–R7 entailment replay — every adversarial passage that ever
// released a wrong answer through the public gate, replayed through the
// independent Qwen3.8-27B semantic entailment witness + the exact credal
// kernel. Live test: skips cleanly when Ollama or the witness model is
// unreachable so `npm test` stays green offline.
//
//   safety property : ZERO wrong releases on every trap
//   recall property : affirmative + matched same-shift controls release 1/1
//
// Each case calls checkSemanticEntailment once per program (the single
// compact batch prompt), then feeds the SHA-256-bound receipt to BOTH
// public entry points — verifyInvariance and solveWithWitness — which
// re-verify the receipt instead of re-calling the model.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ollamaAvailable } from '../src/live/ollama.js';
import {
  checkSemanticEntailment,
  passageSha256,
  programCanonicalSha256,
  solveWithWitness,
  verifyEntailmentWitness,
  WITNESS_MODEL,
} from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { ctxWitnessCheck } from '../src/invariance/ownership.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags, contextLabelPolarity, validateLicenseDecls } from '../src/kernel/sentences.js';
import * as R from '../src/kernel/rational.js';

const HOST = process.env.PRISM_OLLAMA_HOST ?? 'http://127.0.0.1:11434';
const fmt = (v) => (v ? `${v.n}/${v.d}` : null);

// ---------------------------------------------------------------------------
// fixture builder — the Ash/Fir lamp/buzzer chassis every Bulma round used.
// base: 40/60 priors, 40%/20% lamp marginals, 40%/20% buzzer marginals; the
// only stated joint fact is the link sentence under test. An unsupported
// link means the kernel sees an underdetermined interval — a released 1/1
// is ALWAYS the extraction bug, never the arithmetic.
// ---------------------------------------------------------------------------
const X_BASE = (link, obs, pre = '') =>
  'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
  'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
  'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
  `${pre ? `${pre} ` : ''}${link} ${obs} ` +
  'What is the probability its buzzer sounded?';

function bcFor(X, { linkSentence, linkOp = 'EQUIV', obsSentence }) {
  const at = (s) => {
    const i = X.indexOf(s);
    if (i < 0) throw new Error(`fixture needle missing: "${s.slice(0, 40)}"`);
    return `[${i},${i + s.length}]`;
  };
  const bz = X.indexOf('The buzzer sounds');
  return `HYP h0 ${at('Ash')}
HYP h1 ${at('Fir')}
PART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}
EVENT e0 ${at('lamp')}
EVENT e1 ${at('buzzer')}
PRIOR h0 2/5 ctx0 ${at('40% of panels')}
PRIOR h1 3/5 ctx0 ${at('60%.')}
LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}
LIK e1 POS GIVEN h0 2/5 ctx0 [${X.indexOf('40% of Ash', bz)},${X.indexOf('40% of Ash', bz) + 10}]
LIK e1 POS GIVEN h1 1/5 ctx0 [${X.indexOf('20% of Fir', bz)},${X.indexOf('20% of Fir', bz) + 10}]
DET_LINK e0 ${linkOp} e1 ctx0 ${at(linkSentence)}
OBS e0 POS ctx0 ${at(obsSentence)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
}

// compiler-derived sentence ledger, exactly what the wired extractor emits
function withLedger(X, bytecode) {
  const prog = toProgram(bytecode);
  const lines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return bytecode.replace(/(^|\n)(ASK_PROB)/, `$1${lines.join('\n')}\n$2`);
}

const DEFAULT_OBS = 'This panel was observed with its lamp lit.';

const MUTEX_S = 'The lamp and buzzer cannot both activate.';
const IMPLIES_S = 'Whenever the lamp lights, the buzzer sounds.';
const BUZZER_RATE = 'The buzzer sounds for 40% of Ash panels and 20% of Fir panels.';

async function inv(X, bc) {
  const bytecode = withLedger(X, bc);
  return verifyInvariance({ X, extract: makeSemanticExtractor(bytecode), offlineSyntheticBench: true });
}
function assertPass(r, expected) {
  assert.equal(r.verdict, 'PASS');
  assert.equal(r.decision.status, 'ANSWER');
  assert.equal(fmt(r.decision.value), expected);
  assert.equal(fmt(r.decision.interval.min), expected);
  assert.equal(fmt(r.decision.interval.max), expected);
}
function assertRefused(r) {
  assert.equal(r.verdict, 'REFUSED');
  assert.equal(r.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
}
function alteredRate(linkSentence, linkOp, obsSentence, replacement) {
  const X = X_BASE(linkSentence, obsSentence).replace(BUZZER_RATE, replacement);
  return [X, bcFor(X, { linkSentence, linkOp, obsSentence })];
}
function doubleObsBc(X, linkSentence, linkOp, obsSentence, polarity) {
  const bc = bcFor(X, { linkSentence, linkOp, obsSentence });
  const obsLine = bc.split('\n').find((line) => line.startsWith('OBS e0 POS'));
  assert.ok(obsLine);
  const second = obsLine.replace('OBS e0 POS', `OBS e1 ${polarity}`);
  return bc
    .replace(obsLine, `${obsLine}\n${second}`)
    .replace('ASK_PROB e1 GIVEN (e0:POS) ctx0', `ASK_PROB h0 GIVEN (e0:POS e1:${polarity}) ctx0`);
}

test('base-rate restrictive relative: benign location preserves result, material panel refuses', async () => {
  const benign = 'The buzzer sounds for 40% of Ash panels whose cabinets stand beside the east wall and 20% of Fir panels.';
  assertPass(await inv(...alteredRate(MUTEX_S, 'MUTEX', DEFAULT_OBS, benign)), '0/1');
  assertPass(await inv(...alteredRate(IMPLIES_S, 'IMPLIES', DEFAULT_OBS, benign)), '1/1');
  const bothOn = 'The buzzer sounds for 40% of Ash panels including one whose lamp lights while its buzzer sounds and 20% of Fir panels.';
  assertRefused(await inv(...alteredRate(MUTEX_S, 'MUTEX', DEFAULT_OBS, bothOn)));
  const lampNoBuzzer = 'The buzzer sounds for 40% of Ash panels including one whose lamp lights while its buzzer remains silent and 20% of Fir panels.';
  assertRefused(await inv(...alteredRate(IMPLIES_S, 'IMPLIES', DEFAULT_OBS, lampNoBuzzer)));
});

test('whole OBS: harmless setting preserves result, other indicator state refuses', async () => {
  const benignObs = 'This panel was observed with its lamp lit beside the east wall.';
  {
    const X = X_BASE(MUTEX_S, benignObs);
    const bc = bcFor(X, { linkSentence: MUTEX_S, linkOp: 'MUTEX', obsSentence: benignObs });
    const direct = await solveWithWitness(toProgram(withLedger(X, bc), X.length), X, { offlineSyntheticBench: true });
    assert.equal(direct.status, 'ANSWER');
    assert.equal(fmt(direct.value), '0/1');
    assert.equal(fmt(direct.interval.min), '0/1');
    assert.equal(fmt(direct.interval.max), '0/1');
    assertPass(await inv(X, bc), '0/1');
  }
  {
    const X = X_BASE(IMPLIES_S, benignObs);
    assertPass(await inv(X, bcFor(X, { linkSentence: IMPLIES_S, linkOp: 'IMPLIES', obsSentence: benignObs })), '1/1');
  }
  const toneObs = 'This panel was observed with its lamp lit while the other indicator sounds an audible tone.';
  {
    const X = X_BASE(MUTEX_S, toneObs);
    assertRefused(await inv(X, bcFor(X, { linkSentence: MUTEX_S, linkOp: 'MUTEX', obsSentence: toneObs })));
  }
  const silentObs = 'This panel was observed with its lamp lit while the other indicator makes no sound.';
  {
    const X = X_BASE(IMPLIES_S, silentObs);
    assertRefused(await inv(X, bcFor(X, { linkSentence: IMPLIES_S, linkOp: 'IMPLIES', obsSentence: silentObs })));
  }
});

test('double OBS same span: matching polarity gives 4/7, mismatched polarity refuses', async () => {
  const toneObs = 'This panel was observed with its lamp lit while the other indicator sounds an audible tone.';
  const X = X_BASE(IMPLIES_S, toneObs);
  assertPass(await inv(X, doubleObsBc(X, IMPLIES_S, 'IMPLIES', toneObs, 'POS')), '4/7');
  assertRefused(await inv(X, doubleObsBc(X, IMPLIES_S, 'IMPLIES', toneObs, 'NEG')));
  const silentObs = 'This panel was observed with its lamp lit while the other indicator makes no sound.';
  const X2 = X_BASE(IMPLIES_S, silentObs);
  assertRefused(await inv(X2, doubleObsBc(X2, IMPLIES_S, 'IMPLIES', silentObs, 'POS')));
});
test('double OBS: intrinsically contradictory same-instant residual about the same second indicator refuses in both directions, distinct ambient clock passes 4/7', async () => {
  const contradictoryObs = 'This panel was observed with its lamp lit while the other indicator sounds an audible tone and that same indicator makes no sound throughout this inspection.';
  const XContraImplies = X_BASE(IMPLIES_S, contradictoryObs);
  assertRefused(await inv(XContraImplies, doubleObsBc(XContraImplies, IMPLIES_S, 'IMPLIES', contradictoryObs, 'POS')));
  const XContraMutex = X_BASE(MUTEX_S, contradictoryObs);
  assertRefused(await inv(XContraMutex, doubleObsBc(XContraMutex, MUTEX_S, 'MUTEX', contradictoryObs, 'NEG')));
  const reversedObs = 'This panel was observed with its lamp lit while the other indicator makes no sound and that same indicator sounds an audible tone throughout this inspection.';
  const XRevImplies = X_BASE(IMPLIES_S, reversedObs);
  assertRefused(await inv(XRevImplies, doubleObsBc(XRevImplies, IMPLIES_S, 'IMPLIES', reversedObs, 'POS')));
  const XRevMutex = X_BASE(MUTEX_S, reversedObs);
  assertRefused(await inv(XRevMutex, doubleObsBc(XRevMutex, MUTEX_S, 'MUTEX', reversedObs, 'NEG')));
  const ambientObs = 'This panel was observed with its lamp lit while the other indicator remains silent and makes no sound throughout this inspection although the nearby wall clock rings audibly throughout this inspection.';
  const XAmbient = X_BASE(MUTEX_S, ambientObs);
  assertPass(await inv(XAmbient, doubleObsBc(XAmbient, MUTEX_S, 'MUTEX', ambientObs, 'NEG')), '4/7');
});
const mk = ({ id, link, linkOp, obs = DEFAULT_OBS, pre = '', expect, note }) => {
  const X = X_BASE(link, obs, pre);
  return { id, X, bc: withLedger(X, bcFor(X, { linkSentence: link, linkOp, obsSentence: obs })), expect, note };
};

const OBS_NEGATED_SHIFT =
  'This panel was observed with its lamp lit and is not a night-shift panel.';
const OBS_OUTSIDE_SHIFT =
  'This panel was observed with its lamp lit and is outside the night shift.';
const OBS_DAY_SHIFT = 'A day-shift panel was observed with its lamp lit.';
const OBS_NIGHT_SHIFT = 'A night-shift panel was observed with its lamp lit.';
const OBS_CURRENT = 'This panel is currently observed with its lamp lit.';
const LINK_NIGHT = 'On night-shift panels, the lamp and buzzer always activate together.';

// ---------------------------------------------------------------------------
// The full Bulma R1–R7 adversarial set + unseen twists + controls.
//   expect.refuse: the gate must refuse — released answer is a wrong release
//   expect.answer: exact kernel value that must release
//   expect.entailed: what the witness ITSELF must report (the semantic leg)
// ---------------------------------------------------------------------------
const CASES = [
  // ---- R2: negated link + day/night anaphora + partial cue --------------
  mk({
    id: 'r2-negated-link-as-equiv',
    link: 'The lamp and buzzer never activate together.',
    expect: { refuse: true, entailed: false },
    note: 'was 1/1 — "never together" ledgered EQUIV',
  }),
  mk({
    id: 'r2-sometimes-link-as-equiv',
    link: 'The lamp and buzzer sometimes activate together and sometimes activate separately.',
    expect: { refuse: true, entailed: false },
    note: 'was 1/1 — partial co-activation is not equivalence',
  }),
  mk({
    id: 'r2-mutex-faithful-control',
    link: 'The lamp and buzzer never activate together.',
    linkOp: 'MUTEX',
    expect: { answer: '0/1', entailed: true },
    note: 'the same sentence faithfully encoded releases 0/1',
  }),
  mk({
    id: 'r2-night-link-day-obs',
    link: LINK_NIGHT,
    obs: OBS_DAY_SHIFT,
    expect: { refuse: true, entailed: false },
    note: 'was 1/1 — night-shift link cannot serve a day-shift observation',
  }),
  // ---- R3: partial co-activation ----------------------------------------
  mk({
    id: 'r3-partial-coactivation',
    link: 'The lamp and buzzer activate together on some panels, and each also activates by itself.',
    expect: { refuse: true, entailed: false },
    note: 'was 1/1 — explicit solo mass rules out equivalence',
  }),
  // ---- R4: quantifier-wrapped cues --------------------------------------
  mk({
    id: 'r4-subset-universal',
    link: 'On a subset of panels, the lamp and buzzer always activate together, and elsewhere both solo patterns occur.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r4-existential-wiring',
    link: 'At least one panel has a lamp and buzzer that are wired to the same switch.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r4-modal-wiring',
    link: 'It is possible that the lamp and buzzer are wired to the same switch on every panel.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r4-hypothesis-scoped',
    link: 'The lamp and buzzer always activate together on Ash panels.',
    expect: { refuse: true, entailed: false },
  }),
  // ---- R5: unrestricted quantified noun phrase --------------------------
  mk({
    id: 'r5-common-relay-subset',
    link: 'On every panel with a common relay, the lamp and buzzer always activate together.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r5-all-intact',
    link: 'The lamp and buzzer always activate together on all intact panels.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r5-both-hyp-fitted',
    link: 'On Ash and Fir panels fitted with a common relay, the lamp and buzzer always activate together.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r5-except-damaged',
    link: 'On every panel except damaged panels, the lamp and buzzer always activate together.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r5-prototype-possessive',
    link: "Every prototype's lamp and buzzer are wired to the same switch.",
    expect: { refuse: true, entailed: false },
  }),
  // ---- R6: exception inside device NP + foreign possessives -------------
  mk({
    id: 'r6-except-testing',
    link: 'The lamp and buzzer are wired to the same switch except during testing.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r6-refurbished-scope',
    link: 'On every Refurbished panel, the lamp and buzzer always activate together.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r6-clinic-possessive',
    link: "Every clinic's lamp and buzzer are wired to the same switch.",
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'r6-crew-possessive',
    link: "Every crew's lamp and buzzer are wired to the same switch.",
    expect: { refuse: true, entailed: false },
  }),
  // ---- R7: the five live blockers ---------------------------------------
  mk({
    id: 'r7-both-damaged-panels',
    link: 'Both the lamp and buzzer on damaged panels always activate together.',
    expect: { refuse: true, entailed: false },
    note: 'ANA_SUBJ swallowed the subset',
  }),
  mk({
    id: 'r7-both-locked-enclosure',
    link: 'Both the lamp and buzzer inside the locked enclosure are wired to the same switch.',
    expect: { refuse: true, entailed: false },
    note: 'ANA_SUBJ swallowed the enclosure restriction',
  }),
  mk({
    id: 'r7-fluff-clinic',
    pre: 'The clinic is open.',
    link: 'In every clinic, the lamp and buzzer always activate together.',
    expect: { refuse: true, entailed: false },
    note: 'FLUFF-supplied noun self-grounded the foreign population',
  }),
  mk({
    id: 'r7-were-wired',
    link: 'The lamp and buzzer were wired to the same switch.',
    obs: OBS_CURRENT,
    expect: { refuse: true, entailed: false },
    note: 'past wiring leaves present wiring unspecified',
  }),
  mk({
    id: 'r7-negated-shift-membership',
    link: LINK_NIGHT,
    obs: OBS_NEGATED_SHIFT,
    expect: { refuse: true, entailed: false },
    note: '"is not a night-shift panel" counted as positive membership',
  }),
  // ---- unseen adversarial twists ----------------------------------------
  mk({
    id: 'twist-used-to-be-wired',
    link: 'The lamp and buzzer used to be wired to the same switch.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'twist-except-tuesdays',
    link: 'The lamp and buzzer are wired to the same switch except on Tuesdays.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'twist-unless-wet',
    link: 'The lamp and buzzer are wired to the same switch unless the panel is wet.',
    expect: { refuse: true, entailed: false },
  }),
  mk({
    id: 'twist-outside-night-shift',
    link: LINK_NIGHT,
    obs: OBS_OUTSIDE_SHIFT,
    expect: { refuse: true, entailed: false },
  }),
  // ---- controls that must still release ----------------------------------
  mk({
    id: 'control-affirmative',
    link: 'The lamp and buzzer always activate together.',
    expect: { answer: '1/1', entailed: true },
  }),
  mk({
    id: 'control-wired-same-switch',
    link: 'The lamp and buzzer are wired to the same switch.',
    expect: { answer: '1/1', entailed: true },
  }),
  mk({
    id: 'control-matched-night-shift',
    link: LINK_NIGHT,
    obs: OBS_NIGHT_SHIFT,
    expect: { answer: '1/1', entailed: true },
    note: 'matched same-shift link + observation releases exactly',
  }),
];

test('Bulma R7 passages refuse in both public paths before any Ollama request', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => {
    throw new Error('unexpected witness request');
  });
  const cases = CASES.filter((c) => c.id.startsWith('r7-'));
  assert.equal(cases.length, 5);
  for (const c of cases) {
    const expected = c.id === 'r7-negated-shift-membership'
      ? 'CONTEXT_MISMATCH'
      : 'UNLICENSED_INDEPENDENCE';
    const gate = await verifyInvariance({ X: c.X, extract: makeSemanticExtractor(c.bc) });
    const decision = await solveWithWitness(c.bc, c.X);
    assert.equal(gate.verdict, 'REFUSED', c.id);
    assert.equal(gate.code, expected, c.id);
    assert.equal(decision.status, 'REFUSED', c.id);
    assert.equal(decision.code, expected, c.id);
  }
  assert.equal(fetch.mock.callCount(), 0);
});

test('canonical relation subjects and wiring distinguish scope and tense', () => {
  const cases = [
    ['Both the lamp and buzzer always activate together.', true],
    ['Both outputs are wired to the same switch.', true],
    ['Both of them share a relay.', true],
    ['These two signals always activate together.', true],
    ['Those two outputs always move in lockstep.', true],
    ['These two signals on damaged panels always activate together.', false],
    ['These two signals inside the locked enclosure are wired to the same switch.', false],
    ['Both outputs located beneath the deck always activate together.', false],
    ['Both outputs beside damaged panels are wired to the same switch.', false],
    ['The lamp and buzzer were wired to the same switch.', false],
    ['The lamp and buzzer shared a relay.', false],
  ];
  for (const [link, licensed] of cases) {
    const c = mk({ link, obs: OBS_CURRENT });
    const invalid = validateLicenseDecls(c.X, toProgram(c.bc));
    assert.equal(invalid.length === 0, licensed, link);
  }
});

test('a FLUFF ledger entry cannot contribute a population noun even if the text looks material', () => {
  const c = mk({
    pre: 'The clinic has its own lamp and buzzer.',
    link: 'In every clinic, the lamp and buzzer always activate together.',
  });
  const prog = toProgram(c.bc);
  const clinicTag = prog.decls.find((d) => d.op === 'SENT_TAG' && d.idx === 4);
  clinicTag.tag = 'FLUFF';
  assert.equal(validateLicenseDecls(c.X, prog).length, 1);
});

test('an observation can deny membership even when the question names that population', () => {
  const excludeLc = new Set(['lamp', 'buzzer', 'ash', 'fir']);
  for (const text of [
    OBS_NEGATED_SHIFT,
    'This panel isn’t a night-shift panel.',
    'It is a night-shift panel on paper, but it is not a night-shift panel.',
  ]) {
    assert.deepEqual(contextLabelPolarity(text, excludeLc), {
      affirmed: new Set(),
      excluded: new Set(['night shift']),
    });
  }
  assert.deepEqual(contextLabelPolarity(OBS_NIGHT_SHIFT, excludeLc), {
    affirmed: new Set(['night shift']),
    excluded: new Set(),
  });

  const c = CASES.find((c) => c.id === 'r7-negated-shift-membership');
  const X = c.X.replace('its buzzer', 'the night-shift panel’s buzzer');
  const { decls } = toProgram(c.bc);
  const decision = { status: 'ANSWER', witnesses: decls };
  const findings = ctxWitnessCheck({ X, decision, excludeLc });
  assert.equal(findings.length, 1);
  assert.equal(findings[0].code, 'CONTEXT_MISMATCH');
  assert.match(findings[0].detail, /affirmatively denied/);
});

test('matching hashes cannot forge a witness when Ollama is unreachable', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => {
    throw new Error('connection refused');
  });
  const c = CASES.find((c) => c.id === 'control-wired-same-switch');
  const prog = toProgram(c.bc);
  const unavailable = await checkSemanticEntailment(prog, c.X);
  assert.equal(unavailable.code, 'ENTAILMENT_WITNESS_UNAVAILABLE');
  assert.equal(verifyEntailmentWitness(unavailable, prog, c.X).code, unavailable.code);
  assert.throws(() => { unavailable.entailed = true; }, TypeError);

  const forged = {
    witnessModel: WITNESS_MODEL,
    passageSha256: passageSha256(c.X),
    programCanonicalSha256: programCanonicalSha256(prog),
    entailed: true,
  };
  const expected = 'ENTAILMENT_WITNESS_UNAUTHENTICATED';
  assert.equal(verifyEntailmentWitness(forged, prog, c.X).code, expected);
  assert.equal((await solveWithWitness(prog, c.X, { entailmentWitness: forged })).code, expected);
  assert.equal((await verifyInvariance({
    X: c.X,
    extract: makeSemanticExtractor(c.bc),
    entailmentWitness: forged,
  })).code, expected);
  assert.equal(fetch.mock.callCount(), 1);
});

test('issued witness round-trips through both public paths and retains its bindings', async (t) => {
  const response = {
    entailed: true,
    checks: [{ id: 'L1', entailed: true }, { id: 'M1', entailed: true }],
  };
  const fetch = t.mock.method(globalThis, 'fetch', async () =>
    Response.json({ message: { content: JSON.stringify(response) } })
  );
  const c = CASES.find((c) => c.id === 'control-wired-same-switch');
  const prog = toProgram(c.bc);
  const witness = await checkSemanticEntailment(prog, c.X);
  assert.deepEqual(verifyEntailmentWitness(witness, prog, c.X), { ok: true });
  assert.ok(Object.isFrozen(witness.checks[0]));
  assert.equal(verifyEntailmentWitness({ ...witness }, prog, c.X).code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED');
  assert.equal(verifyEntailmentWitness(witness, prog, c.X + ' ').code, 'ENTAILMENT_WITNESS_MISMATCH');
  const altered = toProgram(c.bc.replace('ASK_PROB e1', 'ASK_PROB NOT e1'));
  assert.equal(verifyEntailmentWitness(witness, altered, c.X).code, 'ENTAILMENT_WITNESS_MISMATCH');
  const gate = await verifyInvariance({ X: c.X, extract: makeSemanticExtractor(c.bc), entailmentWitness: witness });
  const decision = await solveWithWitness(prog, c.X, { entailmentWitness: witness });
  assert.equal(gate.verdict, 'PASS');
  assert.equal(fmt(gate.decision.value), '1/1');
  assert.equal(decision.status, 'ANSWER');
  assert.equal(fmt(decision.value), '1/1');
  assert.equal(fetch.mock.callCount(), 1);
});

test('witness authentication and issuance survive WeakSet prototype replacement', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ message: {
    content: JSON.stringify({ entailed: true, checks: [
      { id: 'L1', entailed: true }, { id: 'M1', entailed: true },
    ] }),
  } }));
  const c = CASES.find(({ id }) => id === 'control-wired-same-switch');
  const prog = toProgram(c.bc);
  const forged = {
    witnessModel: WITNESS_MODEL,
    witnessHost: 'http://127.0.0.1:11434',
    passageSha256: passageSha256(c.X),
    programCanonicalSha256: programCanonicalSha256(prog),
    entailed: true,
  };
  const { has, add } = WeakSet.prototype;
  let result;
  let genuine;
  try {
    WeakSet.prototype.has = () => true;
    WeakSet.prototype.add = function () { return this; };
    result = verifyEntailmentWitness(forged, prog, c.X);
    genuine = await checkSemanticEntailment(prog, c.X);
  } finally {
    WeakSet.prototype.has = has;
    WeakSet.prototype.add = add;
  }
  assert.equal(result.code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED');
  assert.deepEqual(verifyEntailmentWitness(genuine, prog, c.X), { ok: true });
});

test('top-level true cannot substitute for empty, omitted, or duplicate witness checks', async (t) => {
  let checks;
  t.mock.method(globalThis, 'fetch', async () => Response.json({ message: {
    content: JSON.stringify({ entailed: true, checks }),
  } }));
  const c = CASES.find(({ id }) => id === 'control-wired-same-switch');
  const prog = toProgram(c.bc);
  for (const supplied of [
    [], undefined, [{ id: 'M1', entailed: true }],
    [{ id: 'M1', entailed: true }, { id: 'L1', entailed: false }, { id: 'L1', entailed: true }],
  ]) {
    checks = supplied;
    const witness = await checkSemanticEntailment(prog, c.X);
    assert.equal(witness.entailed, false);
    const opts = { entailmentWitness: witness };
    assert.equal((await solveWithWitness(prog, c.X, opts)).code, 'SEMANTIC_NOT_ENTAILED');
    assert.equal((await verifyInvariance({
      X: c.X, extract: makeSemanticExtractor(c.bc), ...opts,
    })).code, 'SEMANTIC_NOT_ENTAILED');
  }
});

test('custom witness endpoint requires opt-in at issuance and consumption', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => Response.json({ message: {
    content: JSON.stringify({ entailed: true, checks: [
      { id: 'L1', entailed: true }, { id: 'M1', entailed: true },
    ] }),
  } }));
  const c = CASES.find(({ id }) => id === 'control-wired-same-switch');
  const prog = toProgram(c.bc);
  const gate = { X: c.X, extract: makeSemanticExtractor(c.bc) };
  const host = 'http://127.0.0.1:54321';
  const model = 'stub-model';
  const unavailable = 'ENTAILMENT_WITNESS_UNAVAILABLE';
  for (const [options, gateOptions] of [[{ host }, { ollamaHost: host }], [{ model }, { witnessModel: model }]]) {
    const witness = await checkSemanticEntailment(prog, c.X, options);
    assert.equal(witness.code, unavailable);
    assert.match(witness.reason, /allowCustomWitnessEndpoint/);
    assert.equal((await solveWithWitness(prog, c.X, options)).code, unavailable);
    assert.equal((await verifyInvariance({ ...gate, ...gateOptions })).code, unavailable);
  }
  assert.equal(fetch.mock.callCount(), 0);

  const allowCustomWitnessEndpoint = true;
  const witness = await checkSemanticEntailment(prog, c.X, { host, allowCustomWitnessEndpoint });
  assert.equal(witness.entailed, true);
  const replay = { entailmentWitness: witness };
  assert.equal((await solveWithWitness(prog, c.X, replay)).code, 'ENTAILMENT_WITNESS_MISMATCH');
  assert.equal((await verifyInvariance({ ...gate, ...replay })).code, 'ENTAILMENT_WITNESS_MISMATCH');
  assert.equal((await solveWithWitness(prog, c.X, { host, ...replay })).code, unavailable);
  assert.equal((await verifyInvariance({ ...gate, ollamaHost: host, ...replay })).code, unavailable);
  assert.equal((await solveWithWitness(prog, c.X, { host, allowCustomWitnessEndpoint, ...replay })).status, 'ANSWER');
  assert.equal((await verifyInvariance({
    ...gate, ollamaHost: host, allowCustomWitnessEndpoint, ...replay,
  })).verdict, 'PASS');
  assert.equal((await solveWithWitness(prog, c.X, { host, model, allowCustomWitnessEndpoint })).status, 'ANSWER');
  assert.equal((await verifyInvariance({
    ...gate, ollamaHost: host, witnessModel: model, allowCustomWitnessEndpoint,
  })).verdict, 'PASS');
  assert.equal(fetch.mock.callCount(), 3);
  assert.equal(fetch.mock.calls[2].arguments[0], `${host}/api/chat`);
  assert.equal(JSON.parse(fetch.mock.calls[2].arguments[1].body).model, model);
});

test('bulma R1–R7 + twists: zero wrong releases through live Qwen witness + credal kernel', async (t) => {
  const models = await ollamaAvailable(HOST, 4000);
  if (!models) {
    t.skip(`no ollama daemon at ${HOST}`);
    return;
  }
  if (!models.some((m) => m === WITNESS_MODEL || m.startsWith('hf.co/unsloth/Qwen3.8-27B'))) {
    t.skip(`witness model ${WITNESS_MODEL} not on ${HOST}`);
    return;
  }

  const failures = [];
  let wrongReleases = 0;
  const config = { host: HOST, allowCustomWitnessEndpoint: true };
  for (const c of CASES) {
    const prog = toProgram(c.bc, c.X.length);
    // the independent witness — one batch call per program
    const witness = await checkSemanticEntailment(prog, c.X, config);
    // both public entry points consume the SAME verified receipt
    const gate = await verifyInvariance({
      X: c.X,
      extract: makeSemanticExtractor(c.bc),
      entailmentWitness: witness,
      ollamaHost: HOST,
      allowCustomWitnessEndpoint: true,
    });
    const sw = await solveWithWitness(prog, c.X, { ...config, entailmentWitness: witness });

    t.diagnostic(
      `${c.id}: witness=${witness.entailed} (${witness.code ?? 'ok'}) ` +
        `gate=${gate.verdict}${gate.code ? `/${gate.code}` : ''} ` +
        `solve=${sw.status}${sw.code ? `/${sw.code}` : ''}` +
        (sw.status === 'ANSWER' ? `=${fmt(sw.value)}` : '')
    );

    const released =
      (gate.verdict === 'PASS' && gate.decision?.status === 'ANSWER' ? fmt(gate.decision.value) : null) ??
      (sw.status === 'ANSWER' ? fmt(sw.value) : null);

    if (c.expect.refuse) {
      if (gate.verdict === 'PASS' || sw.status === 'ANSWER') {
        wrongReleases++;
        failures.push(`${c.id}: WRONG RELEASE ${released} — ${c.note ?? ''}`);
      }
      if (c.expect.entailed === false && witness.entailed !== false) {
        failures.push(`${c.id}: witness failed to reject (entailed=${witness.entailed})`);
      }
    } else {
      // release control: witness must entail, and BOTH entry points must
      // release the exact kernel value — nothing more, nothing less.
      if (c.expect.entailed === true && witness.entailed !== true) {
        failures.push(
          `${c.id}: witness rejected control (${witness.code}: ${witness.reason})`
        );
      }
      if (gate.verdict !== 'PASS' || fmt(gate.decision?.value) !== c.expect.answer) {
        failures.push(
          `${c.id}: gate did not release ${c.expect.answer} (${gate.verdict}/${gate.code ?? 'no-code'})`
        );
      }
      if (sw.status !== 'ANSWER' || fmt(sw.value) !== c.expect.answer) {
        failures.push(
          `${c.id}: solveWithWitness did not release ${c.expect.answer} (${sw.status}/${sw.code ?? 'no-code'})`
        );
      }
    }
  }

  assert.equal(wrongReleases, 0, JSON.stringify(failures));
  assert.deepEqual(failures, [], JSON.stringify(failures, null, 2));
});
