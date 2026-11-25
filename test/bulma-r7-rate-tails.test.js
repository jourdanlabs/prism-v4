// Bulma R7 final-gate regressions (Toph-authored, cold-frame domain).
// Round 7 released when a counterexample clause was appended to a
// PRIOR or LIK sentence: exact span-interval accounting only ran on
// OBS sentences, and a rate span covers only its numeric literal, so
// text after the unit noun was never judged. The fix is structural:
// concessive/temporal subordinators split rate sentences, the text
// beyond the rightmost rate span (and under a concessive leading
// frame) must be inert on its own, and the entailment witness sweeps
// every sentence for a reported counterexample to a license. Every
// sentence below is written fresh for this suite.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags, sentences, subClauses } from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import {
  checkSemanticEntailment,
  licenseCounterexampleFindings,
  solveWithWitness,
  verifyEntailmentWitness,
} from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

const CORE = 'this frame was noted with its shutter open';
const LINKS = {
  EQUIV: 'The shutter and vent share a hinge.',
  IMPLIES: 'The shutter always triggers the vent.',
  MUTEX: 'The shutter and vent cannot both activate.',
  COND_INDEP: 'The shutter and vent are conditionally independent within each frame type.',
};

function build({
  rel = 'EQUIV', wholeObsSpan = false, numbers = 'A',
  partSentence = 'Every cold frame is either Cedar or Birch.',
  priorSentence, p0, p1,
  lik0Sentence, lik1Sentence, obsSentence, obsSpanText,
  query = 'What is the probability its vent opened?',
}) {
  const B = numbers === 'B';
  p0 ??= B ? '40% of frames' : '25% of frames';
  p1 ??= B ? '60%.' : '75%.';
  const S2 = priorSentence ??
    (B ? 'Cedar accounts for 40% of frames and Birch accounts for 60%.'
       : 'Cedar accounts for 25% of frames and Birch accounts for 75%.');
  const S3 = lik0Sentence ??
    (B ? 'The shutter opens for 30% of Cedar frames and 20% of Birch frames.'
       : 'The shutter opens for 50% of Cedar frames and 20% of Birch frames.');
  const S4 = lik1Sentence ??
    (B ? 'The vent opens for 30% of Cedar frames and 20% of Birch frames.'
       : 'The vent opens for 50% of Cedar frames and 20% of Birch frames.');
  const X = [partSentence, S2, S3, S4, LINKS[rel], obsSentence, query].join(' ');
  const at = (s, from = 0) => {
    const i = X.indexOf(s, from);
    if (i < 0) throw new Error('needle ' + s);
    return `[${i},${i + s.length}]`;
  };
  const s4 = X.indexOf(S4);
  const l0 = B ? '3/10' : '1/2';
  const link = rel === 'COND_INDEP'
    ? `COND_INDEP (e0 e1) GIVEN * ctx0 ${at(LINKS[rel])}`
    : `DET_LINK e0 ${rel} e1 ctx0 ${at(LINKS[rel])}`;
  const bc = `HYP h0 ${at('Cedar')}
HYP h1 ${at('Birch')}
PART_EXHAUSTIVE (h0 h1) ${at('Cedar or Birch')}
EVENT e0 ${at('shutter')}
EVENT e1 ${at('vent')}
PRIOR h0 ${B ? '2/5' : '1/4'} ctx0 ${at(p0)}
PRIOR h1 ${B ? '3/5' : '3/4'} ctx0 ${at(p1)}
LIK e0 POS GIVEN h0 ${l0} ctx0 ${at(B ? '30% of Cedar' : '50% of Cedar')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Birch')}
LIK e1 POS GIVEN h0 ${l0} ctx0 ${at(B ? '30% of Cedar' : '50% of Cedar', s4)}
LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Birch', s4)}
${link}
OBS e0 POS ctx0 ${at(wholeObsSpan ? obsSentence.replace(/\.$/, '') : obsSpanText)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc);
  const tagLines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tagLines.join('\n')}\n$2`) };
}

const CLEAN_OBS = { obsSentence: `At the morning check ${CORE}.`, obsSpanText: CORE };

// ---------------- traps: counterexample appended to a rate sentence ----
const RATE_TRAPS = [
  { // co-activation appended to the second likelihood sentence (MUTEX)
    rel: 'MUTEX', ...CLEAN_OBS,
    lik1Sentence: 'The vent opens for 50% of Cedar frames and 20% of Birch frames even as the shutter and the vent swung open together at the spring trial.',
  },
  { // solo activation appended to the second likelihood sentence (IMPLIES)
    rel: 'IMPLIES', ...CLEAN_OBS,
    lik1Sentence: 'The vent opens for 50% of Cedar frames and 20% of Birch frames following an inspection where the shutter opened while the vent stayed shut.',
  },
  { // solo activation appended to the priors sentence (EQUIV)
    rel: 'EQUIV', ...CLEAN_OBS,
    priorSentence: 'Cedar accounts for 25% of frames and Birch for 75% despite a logged instance of the shutter opening on its own.',
    p0: '25% of frames', p1: '75%',
  },
  { // counterexample appended to the partition sentence (EQUIV)
    rel: 'EQUIV', ...CLEAN_OBS,
    partSentence: 'Every cold frame is either Cedar or Birch despite a logged instance of the shutter opening on its own.',
  },
  { // concessive leading frame on a likelihood sentence (MUTEX)
    rel: 'MUTEX', ...CLEAN_OBS,
    lik1Sentence: 'Despite a logged instance of both openings swinging wide at the spring trial, the vent opens for 50% of Cedar frames and 20% of Birch frames.',
  },
  { // contrary state appended to the observation, narrow span (IMPLIES)
    rel: 'IMPLIES',
    obsSentence: `At the morning check ${CORE} even as its vent stayed shut.`,
    obsSpanText: CORE,
  },
  { // contrary fact smuggled into the query itself (IMPLIES)
    rel: 'IMPLIES', ...CLEAN_OBS,
    query: 'Given that observation and the fact that its vent stayed shut all morning, what is the probability that its vent opened?',
  },
];
for (const [i, spec] of RATE_TRAPS.entries()) {
  test(`R7 material text beyond a rate span refuses (${i})`, async () => {
    const { X, bc } = build(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'REFUSED', `trap ${i}`);
    assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE', `trap ${i}`);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'REFUSED', `trap ${i}`);
  });
}

test('R7 concessive subordinators split a likelihood sentence', () => {
  const seg = sentences(
    'The vent opens for 20% of Birch frames even as the shutter and the vent swung open together.'
  )[0];
  const pieces = subClauses(seg, ['shutter', 'vent', 'Birch']);
  assert.ok(pieces.length >= 2, pieces.map((p) => p.text).join(' | '));
  assert.match(pieces[pieces.length - 1].text, /swung open together/);
});

// ---------------- controls: inert rate tails keep exact answers --------
test('R7 inert rate tails and adjuncts still answer exactly', async () => {
  const controls = [
    { // source adjunct ending the likelihood sentence (MUTEX -> 0/1)
      spec: { rel: 'MUTEX', ...CLEAN_OBS,
        lik1Sentence: 'The vent opens for 50% of Cedar frames and 20% of Birch frames in the spring census.' },
      want: '0/1',
    },
    { // record attribution kept, no co-activation reported (MUTEX -> 0/1)
      spec: { rel: 'MUTEX', ...CLEAN_OBS,
        lik1Sentence: "The vent opens for 50% of Cedar frames and 20% of Birch frames, according to the keeper's logbook." },
      want: '0/1',
    },
    { // population noun phrase ending the priors sentence (EQUIV -> 1/1)
      spec: { rel: 'EQUIV', ...CLEAN_OBS,
        priorSentence: 'Cedar holds 25% and Birch holds 75% in the Cedar and Birch population.',
        p0: '25%', p1: '75%' },
      want: '1/1',
    },
    { // the prior clause's own copula predicate after its literal (EQUIV)
      spec: { rel: 'EQUIV', ...CLEAN_OBS,
        priorSentence: 'Cedar accounts for 25% of frames; 75% of frames are Birch.',
        p0: '25% of frames', p1: '75%' },
      want: '1/1',
    },
    { spec: { rel: 'IMPLIES', obsSentence: `Beside the stone shed, ${CORE}.`, obsSpanText: CORE }, want: '1/1' },
    { spec: { rel: 'MUTEX', obsSentence: `Under the willow arch, ${CORE}.`, obsSpanText: CORE }, want: '0/1' },
    { spec: { rel: 'EQUIV', obsSentence: `During the afternoon inventory ${CORE}.`, obsSpanText: CORE }, want: '1/1' },
    { // licensed independence with a temporal observation adjunct
      spec: { rel: 'COND_INDEP', numbers: 'B',
        obsSentence: `After lunch ${CORE}.`, obsSpanText: CORE },
      want: '1/4',
    },
    { spec: { rel: 'IMPLIES', ...CLEAN_OBS }, want: '1/1' },
  ];
  for (const { spec, want } of controls) {
    const { X, bc } = build(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'ANSWER', JSON.stringify(spec));
    assert.equal(`${d.value.n}/${d.value.d}`, want, JSON.stringify(spec));
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'PASS', JSON.stringify(spec));
  }
});

// ---------------- whole-sentence OBS spans: kernel + witness -------------
// R9 contract: the deterministic kernel itself refuses contradictory
// material inside a whole-sentence OBS span with
// UNACCOUNTED_MATERIAL_SENTENCE, and the public solver/gate surface
// that early code. The entailment witness independently still refuses
// the smuggled clause, and cannot pardon a reported counterexample
// even when the endpoint approves every check.
function approveAll(t, capture) {
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    capture.prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...capture.prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({
      message: { content: JSON.stringify({ entailed: true, checks: ids.map((id) => ({ id, entailed: true })) }) },
    });
  });
}

test('R7 witness sweep refuses a whole-span co-activation report', async (t) => {
  const spec = {
    rel: 'MUTEX', wholeObsSpan: true,
    obsSentence: `At the morning check ${CORE} even as both openings swung wide together.`,
    obsSpanText: CORE,
  };
  const { X, bc } = build(spec);
  const prog = toProgram(bc, X.length);
  assert.equal(solve(prog, X).code, 'UNACCOUNTED_MATERIAL_SENTENCE', 'kernel whole-span refusal');
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(prog, X);
  assert.match(capture.prompt, /^C1 \[COUNTEREXAMPLE SWEEP\]/m);
  assert.match(capture.prompt, /ANY clause anywhere in the passage/);
  assert.equal(witness.entailed, false);
  assert.equal(verifyEntailmentWitness(witness, prog, X).code, 'SEMANTIC_NOT_ENTAILED');
  assert.equal(
    (await solveWithWitness(prog, X, { entailmentWitness: witness })).code,
    'UNACCOUNTED_MATERIAL_SENTENCE'
  );
});

// A reporting verb ("documented") preceding a strictly locative
// relative between the two likelihoods: the vent and shutter are only
// located side by side, so this is not a co-activation report and must
// not defeat the MUTEX license.
const LOCATIVE_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent was beside its shutter and 20% of Birch frames.';
// The matched actual occurrence: same attributed shape, but each part
// governs its own past occurrence verb under "while".
const JOINT_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent opened while its shutter opened and 20% of Birch frames.';
// A locative relative followed by extra trailing joint-occurrence
// material: stripping the locative phrase must not discard the rest.
const LOCATIVE_PLUS_JOINT_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent was beside its shutter, whose vent opened ' +
  'while its shutter opened, and 20% of Birch frames.';
const MOUNTED_LOCATIVE_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent was mounted beside its shutter and 20% of Birch frames.';
const LOCATIVE_WHEN_JOINT_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent was beside its shutter when both activated and 20% of Birch frames.';
const LOCATIVE_WHICH_FIRED_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent was beside its shutter which then fired together and 20% of Birch frames.';
const PAIR_LOCATIVE_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'recorded both indicators beside the door and 20% of Birch frames.';
const PAIR_FIRING_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'recorded both indicators firing at once beside the door and 20% of Birch frames.';
// A locative pair object followed by any further predicate or
// modifier is not a proved complete non-event description, even when
// its words are outside any activation list.
const PAIR_LOCATIVE_PULSING_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'recorded both indicators beside the door pulsing in unison and 20% of Birch frames.';
const PAIR_LOCATIVE_AGLOW_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'recorded both indicators beside the door aglow and 20% of Birch frames.';
// An action predicate governing the pair is not a reporting object.
const PAIR_POWERED_LOCATIVE_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'powered both indicators beside the door and 20% of Birch frames.';
const AMBIGUOUS_TOGETHER_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'recorded its vent and shutter together and 20% of Birch frames.';
const PROGRESSIVE_JOINT_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent opened while its shutter was opening and 20% of Birch frames.';
// The legacy splitter cuts at "and", so the solo piece must carry its
// own explicitly named subject: a physically-located pair followed by
// a later explicit "its vent opened alone" in one unsplit piece.
const LOCATIVE_SOLO_LIK1 =
  'The vent opens for 50% of Cedar frames according to the recorder who ' +
  'documented one frame whose vent was beside its shutter, and its vent opened alone, and 20% of Birch frames.';
// One cardinal-direction premodifier is part of the compact proved
// location NP; any further modifier or predicate after it is not.
const PAIR_WEST_DOOR_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'reported both indicators beside the west door and 20% of Birch frames.';
const PAIR_WEST_DOOR_PULSING_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'reported both indicators beside the west door pulsing in unison and 20% of Birch frames.';
const PAIR_WEST_WOODEN_DOOR_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'reported both indicators beside the west wooden door and 20% of Birch frames.';
// A located report elsewhere in the sentence never licenses a
// separately reported powered pair, in the same piece ("before") or
// in a different piece split at "and".
const PAIR_POWERED_BEFORE_RECORDED_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who said ' +
  'the installer powered both indicators before the clerk recorded both indicators beside the door and 20% of Birch frames.';
const PAIR_POWERED_OTHER_PIECE_LIK1 =
  'The vent opens for 50% of Cedar frames according to the inspector who ' +
  'powered both indicators and the clerk recorded both indicators beside the door and 20% of Birch frames.';

test('R7 purely locative relative between likelihoods is no MUTEX counterexample', async (t) => {
  const { X, bc } = build({ rel: 'MUTEX', lik1Sentence: LOCATIVE_LIK1, ...CLEAN_OBS });
  const prog = toProgram(bc, X.length);
  assert.deepEqual(licenseCounterexampleFindings(X, prog), []);
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(prog, X);
  assert.equal(witness.entailed, true);
  assert.ok(witness.checks.length > 0);
  assert.ok(witness.checks.every((c) => c.entailed === true));
});

test('R7 joint occurrence inside the reported relative defeats MUTEX despite approval', async (t) => {
  const mixed = build({ rel: 'MUTEX', lik1Sentence: LOCATIVE_PLUS_JOINT_LIK1, ...CLEAN_OBS });
  assert.ok(
    licenseCounterexampleFindings(mixed.X, toProgram(mixed.bc, mixed.X.length))
      .some((f) => f.rel === 'MUTEX'),
    'trailing joint occurrence after a locative relative is preserved',
  );
  const { X, bc } = build({ rel: 'MUTEX', lik1Sentence: JOINT_LIK1, ...CLEAN_OBS });
  const prog = toProgram(bc, X.length);
  const findings = licenseCounterexampleFindings(X, prog);
  assert.ok(findings.some((f) => f.rel === 'MUTEX'));
  const relFinding = findings.find((f) => f.piece.text.startsWith('whose'));
  assert.ok(relFinding, 'direct finding on the whole whose-relative');
  assert.equal(X.slice(relFinding.piece.s, relFinding.piece.e), relFinding.piece.text);
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(prog, X);
  assert.equal(witness.entailed, false);
  assert.equal(witness.code, 'RESIDUAL_CLAUSE_NOT_ENTAILED');
  assert.ok(witness.checks.some((c) => c.id === 'R1' && c.entailed === false));
  assert.equal(verifyEntailmentWitness(witness, prog, X).code, 'SEMANTIC_NOT_ENTAILED');
});

test('R7 bounded stative locative is stripped only when its complement is proven pure', async (t) => {
  const mounted = build({ rel: 'MUTEX', lik1Sentence: MOUNTED_LOCATIVE_LIK1, ...CLEAN_OBS });
  assert.deepEqual(
    licenseCounterexampleFindings(mounted.X, toProgram(mounted.bc, mounted.X.length)), []);
  for (const lik1Sentence of [LOCATIVE_WHEN_JOINT_LIK1, LOCATIVE_WHICH_FIRED_LIK1]) {
    const built = build({ rel: 'MUTEX', lik1Sentence, ...CLEAN_OBS });
    assert.ok(
      licenseCounterexampleFindings(built.X, toProgram(built.bc, built.X.length))
        .some((f) => f.rel === 'MUTEX'),
      'trailing joint material after a locative phrase is judged in full',
    );
  }
  const progressive = build({ rel: 'MUTEX', lik1Sentence: PROGRESSIVE_JOINT_LIK1, ...CLEAN_OBS });
  assert.ok(
    licenseCounterexampleFindings(progressive.X, toProgram(progressive.bc, progressive.X.length))
      .some((f) => f.rel === 'MUTEX'));
  const solo = build({ rel: 'EQUIV', lik1Sentence: LOCATIVE_SOLO_LIK1, ...CLEAN_OBS });
  const soloProg = toProgram(solo.bc, solo.X.length);
  assert.ok(
    licenseCounterexampleFindings(solo.X, soloProg)
      .some((f) => f.rel === 'EQUIV'),
    'explicit named subject with solo verb in one piece is a solo counterexample',
  );
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(soloProg, solo.X);
  assert.equal(witness.entailed, false);
  assert.equal(witness.code, 'RESIDUAL_CLAUSE_NOT_ENTAILED');
  assert.ok(witness.checks.some((c) => c.id === 'R1' && c.entailed === false));
  assert.equal(verifyEntailmentWitness(witness, soloProg, solo.X).code, 'SEMANTIC_NOT_ENTAILED');
});

test('R7 attributed non-event pair description is no hard counterexample, firing pair is', async (t) => {
  const benign = build({ rel: 'MUTEX', lik1Sentence: PAIR_LOCATIVE_LIK1, ...CLEAN_OBS });
  const benignProg = toProgram(benign.bc, benign.X.length);
  assert.deepEqual(licenseCounterexampleFindings(benign.X, benignProg), []);
  for (const lik1Sentence of [PAIR_FIRING_LIK1, AMBIGUOUS_TOGETHER_LIK1,
    PAIR_LOCATIVE_PULSING_LIK1, PAIR_LOCATIVE_AGLOW_LIK1,
    PAIR_POWERED_LOCATIVE_LIK1]) {
    const built = build({ rel: 'MUTEX', lik1Sentence, ...CLEAN_OBS });
    assert.ok(
      licenseCounterexampleFindings(built.X, toProgram(built.bc, built.X.length))
        .some((f) => f.rel === 'MUTEX'),
      'unproved or action pair report keeps the hard MUTEX finding');
  }
  const firing = build({ rel: 'MUTEX', lik1Sentence: PAIR_FIRING_LIK1, ...CLEAN_OBS });
  const firingProg = toProgram(firing.bc, firing.X.length);
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(firingProg, firing.X);
  assert.equal(witness.entailed, false);
  assert.equal(
    verifyEntailmentWitness(witness, firingProg, firing.X).code, 'SEMANTIC_NOT_ENTAILED');
});

test('R7 cardinal-door locative passes and the same report with a trailing predicate fails', async (t) => {
  const clean = build({ rel: 'MUTEX', lik1Sentence: PAIR_WEST_DOOR_LIK1, ...CLEAN_OBS });
  const cleanProg = toProgram(clean.bc, clean.X.length);
  assert.deepEqual(licenseCounterexampleFindings(clean.X, cleanProg), []);
  for (const lik1Sentence of [PAIR_WEST_DOOR_PULSING_LIK1, PAIR_WEST_WOODEN_DOOR_LIK1]) {
    const built = build({ rel: 'MUTEX', lik1Sentence, ...CLEAN_OBS });
    assert.ok(
      licenseCounterexampleFindings(built.X, toProgram(built.bc, built.X.length))
        .some((f) => f.rel === 'MUTEX'),
      'extra word or predicate after the compact location NP keeps the hard MUTEX finding');
  }
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(cleanProg, clean.X);
  assert.equal(witness.entailed, true);
  assert.ok(witness.checks.length > 0);
  assert.ok(witness.checks.every((c) => c.entailed === true));
});

test('R7 a located report does not exempt a different powered pair in the same sentence', async (t) => {
  const before = build({ rel: 'MUTEX', lik1Sentence: PAIR_POWERED_BEFORE_RECORDED_LIK1, ...CLEAN_OBS });
  const beforeProg = toProgram(before.bc, before.X.length);
  const beforeFindings = licenseCounterexampleFindings(before.X, beforeProg);
  assert.ok(
    beforeFindings.some((f) => f.rel === 'MUTEX' && /powered both/i.test(f.piece.text)),
    'powered pair before an independently recorded located pair keeps the hard MUTEX finding');
  for (const f of beforeFindings) {
    assert.equal(before.X.slice(f.piece.s, f.piece.e), f.piece.text);
  }
  const other = build({ rel: 'MUTEX', lik1Sentence: PAIR_POWERED_OTHER_PIECE_LIK1, ...CLEAN_OBS });
  const otherFindings = licenseCounterexampleFindings(
    other.X, toProgram(other.bc, other.X.length));
  assert.ok(
    otherFindings.some((f) => f.rel === 'MUTEX' &&
      /powered both/i.test(f.piece.text) && !/recorded both/i.test(f.piece.text)),
    'powered pair in a different piece is not exempted by the located report in another piece');
  for (const f of otherFindings) {
    assert.equal(other.X.slice(f.piece.s, f.piece.e), f.piece.text);
  }
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(beforeProg, before.X);
  assert.equal(witness.entailed, false);
  assert.equal(
    verifyEntailmentWitness(witness, beforeProg, before.X).code, 'SEMANTIC_NOT_ENTAILED');
});

test('R7 attributed locative and joint relatives both solicit a whole-sentence C check', async (t) => {
  const prompts = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const prompt = JSON.parse(init.body).messages[0].content;
    prompts.push(prompt);
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({
      message: {
        content: JSON.stringify({
          entailed: true,
          checks: ids.map((id) => ({ id, entailed: true })),
        }),
      },
    });
  });
  for (const lik1Sentence of [LOCATIVE_LIK1, JOINT_LIK1, PAIR_LOCATIVE_LIK1]) {
    const { X, bc } = build({ rel: 'MUTEX', lik1Sentence, ...CLEAN_OBS });
    await checkSemanticEntailment(toProgram(bc, X.length), X);
  }
  assert.equal(prompts.length, 3);
  assert.ok(prompts.every((p) => /^C\d+ \[COUNTEREXAMPLE SWEEP\]/m.test(p)));
  assert.ok(prompts.every((p) => p.includes('TRUE only if NO')));
  assert.ok(prompts.every((p) => p.includes('grammatical subject')));
  assert.ok(prompts.every((p) => p.includes('mere location or')));
});

test('R7 rejected required model claim still vetoes the benign locative text', async (t) => {
  const { X, bc } = build({ rel: 'MUTEX', lik1Sentence: LOCATIVE_LIK1, ...CLEAN_OBS });
  const prog = toProgram(bc, X.length);
  assert.deepEqual(licenseCounterexampleFindings(X, prog), []);
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({
      message: {
        content: JSON.stringify({
          entailed: true,
          checks: ids.map((id, i) => ({ id, entailed: i !== 0 })),
        }),
      },
    });
  });
  const witness = await checkSemanticEntailment(prog, X);
  assert.equal(witness.entailed, false);
  assert.equal(verifyEntailmentWitness(witness, prog, X).code, 'SEMANTIC_NOT_ENTAILED');
});

test('R7 witness sweep refuses a whole-span contrary vent state', async (t) => {
  const spec = {
    rel: 'IMPLIES', wholeObsSpan: true,
    obsSentence: `At the morning check ${CORE} with its vent shut for the whole check.`,
    obsSpanText: CORE,
  };
  const { X, bc } = build(spec);
  const prog = toProgram(bc, X.length);
  assert.equal(solve(prog, X).code, 'UNACCOUNTED_MATERIAL_SENTENCE', 'kernel whole-span refusal');
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(prog, X);
  assert.equal(witness.entailed, false);
  assert.equal(
    (await solveWithWitness(prog, X, { entailmentWitness: witness })).code,
    'UNACCOUNTED_MATERIAL_SENTENCE'
  );
});

test('R7 witness certifies a clean whole-span observation', async (t) => {
  const spec = {
    rel: 'IMPLIES', wholeObsSpan: true,
    obsSentence: `${CORE.charAt(0).toUpperCase() + CORE.slice(1)}.`,
    obsSpanText: CORE,
  };
  const { X, bc } = build(spec);
  const prog = toProgram(bc, X.length);
  const d = solve(prog, X);
  assert.equal(d.status, 'ANSWER');
  assert.equal(`${d.value.n}/${d.value.d}`, '1/1');
  const capture = {};
  approveAll(t, capture);
  const witness = await checkSemanticEntailment(prog, X);
  assert.doesNotMatch(capture.prompt, /\[COUNTEREXAMPLE SWEEP\]/);
  assert.equal(witness.entailed, true);
  assert.deepEqual(verifyEntailmentWitness(witness, prog, X), { ok: true });
  assert.equal((await solveWithWitness(prog, X, { entailmentWitness: witness })).status, 'ANSWER');
});
