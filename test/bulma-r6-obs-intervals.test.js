// Bulma R6 final-gate regressions (Toph-authored, aquarium domain).
// Round 6 released on observation sentences whose OBS span covered
// only a proper sub-span of the sentence while the contrary fact rode
// in the uncovered prefix/suffix behind a connector no list knew, and
// separately regressed a shipped coordinated-subject positive control
// in the witness. Every sentence below is written fresh for this
// suite; no gate wording is reused. The fix is structural: exact
// span-interval subtraction on OBS sentences (no connector lists), a
// coordinated-subject guard plus a DET_LINK/COND_INDEP overlap
// exemption in the witness findings, clause-final irregular verbs,
// and a full-sentence OBS entailment check for whole-sentence spans.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toProgram } from '../src/kernel/parser.js';
import {
  adjacentClauseFindings, autoSentTags, sentences, subClauses,
} from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import {
  checkSemanticEntailment, solveWithWitness, verifyEntailmentWitness,
} from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

const CORE = 'This tank was checked with its beacon glowing';

function build({ obsSentence, obsSpanText, wholeObsSpan = false, rel = 'EQUIV' }) {
  const link = rel === 'IMPLIES'
    ? 'The beacon always triggers the pump.'
    : 'The beacon and pump share a sump.';
  const S1 = 'Every tank is either Reef or Kelp.';
  const S2 = 'Reef makes up 25% of tanks and Kelp makes up 75%.';
  const S3 = 'The beacon glows for 50% of Reef tanks and 20% of Kelp tanks.';
  const S4 = 'The pump runs for 50% of Reef tanks and 20% of Kelp tanks.';
  const Q = 'What is the probability its pump ran?';
  const X = [S1, S2, S3, S4, link, obsSentence, Q].join(' ');
  const at = (s, from = 0) => {
    const i = X.indexOf(s, from);
    if (i < 0) throw new Error('needle ' + s);
    return `[${i},${i + s.length}]`;
  };
  const s4 = X.indexOf(S4);
  const bc = `HYP h0 ${at('Reef')}
HYP h1 ${at('Kelp')}
PART_EXHAUSTIVE (h0 h1) ${at('Reef or Kelp')}
EVENT e0 ${at('beacon')}
EVENT e1 ${at('pump')}
PRIOR h0 1/4 ctx0 ${at('25% of tanks')}
PRIOR h1 3/4 ctx0 ${at('75%.')}
LIK e0 POS GIVEN h0 1/2 ctx0 ${at('50% of Reef')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Kelp')}
LIK e1 POS GIVEN h0 1/2 ctx0 ${at('50% of Reef', s4)}
LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Kelp', s4)}
DET_LINK e0 ${rel} e1 ctx0 ${at(link)}
OBS e0 POS ctx0 ${at(wholeObsSpan ? obsSentence.replace(/\.$/, '') : obsSpanText)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc);
  const tagLines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tagLines.join('\n')}\n$2`) };
}

// Uncovered suffix/prefix matter behind attachments no connector list
// carries in splitting position: each must refuse by interval alone.
const TRAPS = [
  { obsSentence: `${CORE} after a lone beacon display earlier that day.`, obsSpanText: CORE },
  { obsSentence: `${CORE} following a recorded beacon-only display at first light.`, obsSpanText: CORE },
  { obsSentence: `${CORE} even as its pump stayed idle.`, obsSpanText: CORE, rel: 'IMPLIES' },
  { obsSentence: `${CORE} at the very moment its pump was idle.`, obsSpanText: CORE, rel: 'IMPLIES' },
  { obsSentence: `${CORE} in spite of the pump staying idle.`, obsSpanText: CORE, rel: 'IMPLIES' },
  { obsSentence: `${CORE} even as both outputs were live together.`, obsSpanText: CORE },
  { obsSentence: `Only the pump sounded and ${CORE.charAt(0).toLowerCase() + CORE.slice(1)}.`,
    obsSpanText: CORE.charAt(0).toLowerCase() + CORE.slice(1) },
  { obsSentence: `${CORE} (a lone beacon display was logged at first light).`, obsSpanText: CORE },
];
for (const [i, spec] of TRAPS.entries()) {
  test(`R6 uncovered OBS interval refuses (${i})`, async () => {
    const { X, bc } = build(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'REFUSED', spec.obsSentence);
    assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE', spec.obsSentence);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'REFUSED', spec.obsSentence);
  });
}

test('R6 inert uncovered OBS adjuncts still answer 1/1', async () => {
  const lower = CORE.charAt(0).toLowerCase() + CORE.slice(1);
  const controls = [
    { obsSentence: `${CORE} after the midday tea round.`, obsSpanText: CORE },
    { obsSentence: `${CORE} beside the cedar gate.`, obsSpanText: CORE },
    { obsSentence: `${CORE} at the north doorway.`, obsSpanText: CORE },
    { obsSentence: `${CORE} in the west bay.`, obsSpanText: CORE },
    { obsSentence: `${CORE} (beside the cedar gate).`, obsSpanText: CORE },
    { obsSentence: `${CORE} (the lid is painted blue).`, obsSpanText: CORE },
    { obsSentence: `Before the afternoon round; ${lower}.`, obsSpanText: lower },
    { obsSentence: `${CORE}.`, obsSpanText: CORE },
  ];
  for (const spec of controls) {
    const { X, bc } = build(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'ANSWER', spec.obsSentence);
    assert.equal(`${d.value.n}/${d.value.d}`, '1/1', spec.obsSentence);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'PASS', spec.obsSentence);
  }
});

// ---------------- coordinated subject + irregular final verb -----------

test('R6 coordinated subject is never split at the coordinator', () => {
  const seg = sentences('On the reef line, the beacon and pump always activate together.')[0];
  const pieces = subClauses(seg, ['beacon', 'pump']);
  assert.equal(pieces.length, 1, pieces.map((p) => p.text).join(' | '));
});

test('R6 clause-final irregular verb still separates coordinated clauses', () => {
  const seg = sentences('Only the bell rang and this tank was checked with its beacon glowing.')[0];
  const pieces = subClauses(seg, ['bell', 'beacon']);
  assert.ok(pieces.length >= 2, pieces.map((p) => p.text).join(' | '));
  assert.match(pieces[0].text, /bell rang$/);
});

test('R6 a sub-piece overlapping a DET_LINK span is never a witness residual', () => {
  const linkSentence = 'The beacon and pump share a sump despite the pump staying idle.';
  const obsSentence = `${CORE}.`;
  const X = [
    'Every tank is either Reef or Kelp.',
    'Reef makes up 25% of tanks and Kelp makes up 75%.',
    linkSentence,
    obsSentence,
    'What is the probability its pump ran?',
  ].join(' ');
  const at = (s) => {
    const i = X.indexOf(s);
    return `[${i},${i + s.length}]`;
  };
  const bc = `HYP h0 ${at('Reef')}
HYP h1 ${at('Kelp')}
PART_EXHAUSTIVE (h0 h1) ${at('Reef or Kelp')}
EVENT e0 ${at('beacon')}
EVENT e1 ${at('pump')}
DET_LINK e0 EQUIV e1 ctx0 ${at(linkSentence)}
OBS e0 POS ctx0 ${at(CORE)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc, X.length);
  const linkIdx = sentences(X).find((s) => s.text.includes('share a sump')).idx;
  const findings = adjacentClauseFindings(X, prog).filter((f) => f.segIdx === linkIdx);
  assert.deepEqual(findings, []);
});

// ---------------- whole-sentence OBS span goes to the witness ----------

test('R6 whole-sentence OBS span is checked as a full observation sentence', async (t) => {
  const spec = {
    obsSentence: `${CORE} even as both outputs were live together.`,
    obsSpanText: CORE,
  };
  const { X, bc } = build({ ...spec, wholeObsSpan: true });
  const prog = toProgram(bc, X.length);
  // R9: the deterministic kernel itself refuses the contradictory
  // material inside the whole-sentence observation span
  assert.equal(solve(prog, X).code, 'UNACCOUNTED_MATERIAL_SENTENCE');
  let prompt = '';
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({
      message: {
        content: JSON.stringify({
          entailed: false,
          code: 'NOT_ENTAILED',
          checks: ids.map((id) => ({ id, entailed: !id.startsWith('O') })),
        }),
      },
    });
  });
  const witness = await checkSemanticEntailment(prog, X);
  assert.match(prompt, /^O1 \[OBSERVATION SENTENCE\]/m);
  assert.equal(witness.entailed, false);
  assert.equal(
    (await solveWithWitness(prog, X, { entailmentWitness: witness })).code,
    'UNACCOUNTED_MATERIAL_SENTENCE'
  );
});

test('R6 O1 alone vetoes an otherwise approved location-only whole OBS sentence', async (t) => {
  const { X, bc } = build({
    rel: 'IMPLIES',
    obsSentence: CORE + ' beside its second indicator.',
    obsSpanText: CORE,
    wholeObsSpan: true,
  });
  const prog = toProgram(bc, X.length);
  const plain = solve(prog, X);
  assert.equal(plain.status, 'ANSWER');
  assert.deepEqual(plain.value, { n: 1n, d: 1n });
  assert.deepEqual(plain.interval, { min: { n: 1n, d: 1n }, max: { n: 1n, d: 1n } });
  let mode = 'approve';
  let prompt = '';
  let seenIds = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    seenIds = ids;
    const checks = ids
      .filter((id) => mode !== 'omitO1' || id !== 'O1')
      .map((id) => ({ id, entailed: !(mode === 'denyO1' && id === 'O1') }));
    return Response.json({
      message: { content: JSON.stringify({ entailed: true, code: 'ENTAILED', checks }) },
    });
  });
  const approved = await checkSemanticEntailment(prog, X);
  assert.deepEqual([...seenIds].sort(), ['L1', 'M1', 'O1']);
  assert.deepEqual(verifyEntailmentWitness(approved, prog, X), { ok: true });
  const approvedResult = await solveWithWitness(prog, X, { entailmentWitness: approved });
  assert.equal(approvedResult.status, 'ANSWER');
  assert.deepEqual(approvedResult.value, { n: 1n, d: 1n });
  assert.deepEqual(approvedResult.interval, { min: { n: 1n, d: 1n }, max: { n: 1n, d: 1n } });
  assert.equal(approvedResult.entailmentWitness, approved);
  for (mode of ['denyO1', 'omitO1']) {
    const witness = await checkSemanticEntailment(prog, X);
    assert.deepEqual([...seenIds].sort(), ['L1', 'M1', 'O1']);
    const verification = verifyEntailmentWitness(witness, prog, X);
    assert.equal(verification.ok, false);
    assert.equal(verification.code, 'SEMANTIC_NOT_ENTAILED');
    assert.equal(typeof verification.detail, 'string');
    const result = await solveWithWitness(prog, X, { entailmentWitness: witness });
    assert.equal(result.status, 'REFUSED');
    assert.equal(result.code, 'SEMANTIC_NOT_ENTAILED');
    assert.equal(result.entailmentWitness, witness);
  }
});
