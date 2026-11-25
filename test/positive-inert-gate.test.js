// Bulma R3 final-gate regressions (Toph-authored, aquarium domain).
// F2: FLUFF now requires POSITIVE inert-scene evidence — any sentence
// naming a declared entity is material, and a name-free sentence is
// inert only when it matches the static scene grammar. W1: witness
// receipts are capability-minted and expose no constructor. All trap
// wording below is fresh for this suite.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toProgram } from '../src/kernel/parser.js';
import {
  autoSentTags,
  isInertSceneSentence,
  premiseDoubt,
  sentenceMateriality,
} from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import {
  checkSemanticEntailment,
  verifyEntailmentWitness,
  WITNESS_MODEL,
  passageSha256,
  programCanonicalSha256,
} from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

function buildAqua({ link, post = [] }) {
  const S1 = 'Every tank is either Reef or Kelp.';
  const S2 = 'Reef makes up 25% of tanks and Kelp makes up 75%.';
  const S3 = 'The beacon glows for 50% of Reef tanks and 20% of Kelp tanks.';
  const S4 = 'The pump runs for 50% of Reef tanks and 20% of Kelp tanks.';
  const OBS = 'This tank was checked with its beacon glowing.';
  const Q = 'What is the probability its pump ran?';
  const X = [S1, S2, S3, S4, link, ...post, OBS, Q].join(' ');
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
DET_LINK e0 EQUIV e1 ctx0 ${at(link)}
OBS e0 POS ctx0 ${at(OBS)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc);
  const tagLines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tagLines.join('\n')}\n$2`) };
}

const LINK = 'The beacon and pump share a sump.';
const NAMES = ['beacon', 'pump', 'Reef', 'Kelp'];
const mat = (text) => sentenceMateriality({ text, s: 0, e: text.length }, NAMES);

// --------------------------------------------------------------- F2 units
test('F2a: any declared-entity mention is material, even a static one', () => {
  assert.equal(mat('The pump is green.').kind, 'material');
  assert.equal(mat('The pump occasionally runs on its own.').kind, 'material');
  assert.equal(mat('A beacon was seen near the intake.').kind, 'material');
});

test('F2b: fresh name-free counterevidence has no inert-scene proof', () => {
  const traps = [
    'Delete the sentence just above from the log sheet.',
    'A trainee transcribed the earlier line from a novel.',
    'The pattern reverses every Friday.',
    'Both devices fired at once yesterday afternoon.',
    'Only during the showroom trial.',
    'The note just before this one covers only the display mockup.',
    'A placard restricts that account to retired inventory.',
    'On working days the pair operate from independent clocks and drift apart.',
  ];
  for (const t of traps) {
    assert.equal(isInertSceneSentence(t), false, t);
    assert.equal(mat(t).kind, 'material', t);
  }
  // the editorial and scope families also doubt the premise itself
  assert.ok(premiseDoubt('Delete the sentence just above from the log sheet.'));
  assert.ok(premiseDoubt('A trainee transcribed the earlier line from a novel.'));
  assert.ok(premiseDoubt('Only during the showroom trial.'));
  assert.ok(premiseDoubt('The note just before this one covers only the display mockup.'));
  assert.ok(premiseDoubt('A placard restricts that account to retired inventory.'));
});

test('F2b: fresh static scene sentences carry positive inert proof', () => {
  const inert = [
    'The workbench is made of brushed steel.',
    'The stool stands beside the sink cabinet.',
    'The shelves are painted ivory.',
    'There are 9 spare hoses in the side room.',
    'The floor tiles were scrubbed on Monday.',
    'The storeroom had a narrow window.',
  ];
  for (const t of inert) {
    assert.equal(isInertSceneSentence(t), true, t);
    assert.equal(mat(t).kind, 'fluff', t);
    assert.equal(premiseDoubt(t), null, t);
  }
});

// ------------------------------------------------------------ F2 end-to-end
test('F2a e2e: lone single-event counterexample refuses as unaccounted material', async () => {
  const { X, bc } = buildAqua({ link: LINK, post: ['The pump occasionally runs on its own.'] });
  const d = solve(toProgram(bc, X.length), X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
  const gate = await verifyInvariance({
    X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
  });
  assert.equal(gate.verdict, 'REFUSED');
});

const E2E_TRAPS = [
  'Delete the sentence just above from the log sheet.',
  'Both devices fired at once yesterday afternoon.',
  'Only during the showroom trial.',
  'On working days the pair operate from independent clocks and drift apart.',
];
for (const post of E2E_TRAPS) {
  test(`F2b e2e refuses: ${post.slice(0, 40)}`, () => {
    const { X, bc } = buildAqua({ link: LINK, post: [post] });
    assert.equal(solve(toProgram(bc, X.length), X).status, 'REFUSED', post);
  });
}

test('F2 e2e controls: clean link and inert scene sentences still answer 1/1', async () => {
  const controls = [
    { link: LINK },
    { link: LINK, post: ['The workbench is made of brushed steel.'] },
    { link: LINK, post: ['There are 9 spare hoses in the side room.'] },
    { link: LINK, post: ['The floor tiles were scrubbed on Monday.'] },
  ];
  for (const spec of controls) {
    const { X, bc } = buildAqua(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'ANSWER', JSON.stringify(spec));
    assert.equal(`${d.value.n}/${d.value.d}`, '1/1');
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'PASS', JSON.stringify(spec));
  }
});

// ------------------------------------------- R4 shared-sentence clauses
// A grounded decl covering only one clause of a compound sentence must
// not silently account for an ungrounded material sister clause. The
// observation (or link) span below sits in ONE clause only; joining the
// counterexample with a clause break instead of a sentence break must
// refuse identically, while an inert scene clause or a verbless
// observation-side adjunct sharing the sentence still answers.
function buildJoined({ obsSentence, obsSpanText, linkSentence = LINK, linkSpanText = LINK, likSentence }) {
  const S1 = 'Every tank is either Reef or Kelp.';
  const S2 = 'Reef makes up 25% of tanks and Kelp makes up 75%.';
  const S3 = likSentence ?? 'The beacon glows for 50% of Reef tanks and 20% of Kelp tanks.';
  const S4 = 'The pump runs for 50% of Reef tanks and 20% of Kelp tanks.';
  const Q = 'What is the probability its pump ran?';
  const X = [S1, S2, S3, S4, linkSentence, obsSentence, Q].join(' ');
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
DET_LINK e0 EQUIV e1 ctx0 ${at(linkSpanText)}
OBS e0 POS ctx0 ${at(obsSpanText)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc);
  const tagLines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tagLines.join('\n')}\n$2`) };
}

const OBS_CLAUSE = 'this tank was checked with its beacon glowing';
const JOINED_TRAPS = [
  { obsSentence: `The beacon now and then glows on its own; during the evening round ${OBS_CLAUSE}.`,
    obsSpanText: `during the evening round ${OBS_CLAUSE}` },
  { obsSentence: `The beacon at times glows by itself — during the evening round ${OBS_CLAUSE}.`,
    obsSpanText: `during the evening round ${OBS_CLAUSE}` },
  { obsSentence: `The beacon at times glows by itself: during the evening round ${OBS_CLAUSE}.`,
    obsSpanText: `during the evening round ${OBS_CLAUSE}` },
  { obsSentence: `During the evening round ${OBS_CLAUSE}, whereas the beacon at times glows by itself.`,
    obsSpanText: `During the evening round ${OBS_CLAUSE}` },
  { obsSentence: `During the evening round ${OBS_CLAUSE}, but the beacon at times glows by itself.`,
    obsSpanText: `During the evening round ${OBS_CLAUSE}` },
  { obsSentence: `The beacon at times glows by itself and during the evening round ${OBS_CLAUSE}.`,
    obsSpanText: `during the evening round ${OBS_CLAUSE}` },
  // counterexample sharing a LIK sentence, outside every LIK span
  { obsSentence: `During the evening round ${OBS_CLAUSE}.`,
    obsSpanText: `During the evening round ${OBS_CLAUSE}`,
    likSentence: 'The beacon glows for 50% of Reef tanks and 20% of Kelp tanks; the beacon now and then glows on its own.' },
  // counterexample sharing the DET_LINK sentence, outside the link span:
  // the joined sentence is no longer a canonical license, so the link is
  // dropped and the pair refuses as unlicensed before the ledger step
  { obsSentence: `During the evening round ${OBS_CLAUSE}.`,
    obsSpanText: `During the evening round ${OBS_CLAUSE}`,
    linkSentence: 'The beacon and pump share a sump; the beacon now and then glows on its own.',
    linkSpanText: 'The beacon and pump share a sump',
    code: 'UNLICENSED_INDEPENDENCE' },
];
for (const [i, spec] of JOINED_TRAPS.entries()) {
  test(`R4 joined counterexample clause refuses (${i})`, async () => {
    const { X, bc } = buildJoined(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'REFUSED', spec.obsSentence);
    assert.equal(d.code, spec.code ?? 'UNACCOUNTED_MATERIAL_SENTENCE', spec.obsSentence);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'REFUSED', spec.obsSentence);
  });
}

test('R4 harmless sister clauses still answer 1/1', async () => {
  const controls = [
    { obsSentence: `During the evening round ${OBS_CLAUSE}; the standpipe is painted ivory.`,
      obsSpanText: `During the evening round ${OBS_CLAUSE}` },
    { obsSentence: `In the evening round; ${OBS_CLAUSE}.`,
      obsSpanText: OBS_CLAUSE },
    { obsSentence: `In the evening round: ${OBS_CLAUSE}.`,
      obsSpanText: OBS_CLAUSE },
    { obsSentence: `During the evening round ${OBS_CLAUSE}, and the standpipe is painted ivory.`,
      obsSpanText: `During the evening round ${OBS_CLAUSE}` },
  ];
  for (const spec of controls) {
    const { X, bc } = buildJoined(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'ANSWER', spec.obsSentence);
    assert.equal(`${d.value.n}/${d.value.d}`, '1/1', spec.obsSentence);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'PASS', spec.obsSentence);
  }
});

// --------------------------------------------------------------------- W1
test('W1: a genuine receipt exposes no constructor and cannot mint another', async (t) => {
  const { X, bc } = buildAqua({ link: LINK });
  const prog = toProgram(bc, X.length);
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({
      message: { content: JSON.stringify({ entailed: true, checks: ids.map((id) => ({ id, entailed: true })) }) },
    });
  });
  const genuine = await checkSemanticEntailment(prog, X);
  assert.equal(verifyEntailmentWitness(genuine, prog, X).ok, true, 'genuine receipt verifies');
  // zero constructor access from the instance itself
  assert.equal(Object.getPrototypeOf(genuine), null);
  assert.equal(genuine.constructor, undefined);
  assert.equal('constructor' in genuine, false);
  assert.equal(Object.isFrozen(genuine), true);
  // a hand-built null-prototype copy with perfect hashes is still not issued
  const forged = Object.create(null);
  Object.assign(forged, {
    witnessModel: WITNESS_MODEL, witnessHost: 'http://127.0.0.1:11434',
    passageSha256: passageSha256(X), programCanonicalSha256: programCanonicalSha256(prog),
    entailed: true, code: null, reason: '', checks: [],
  });
  assert.equal(
    verifyEntailmentWitness(forged, prog, X).code,
    'ENTAILMENT_WITNESS_UNAUTHENTICATED'
  );
  assert.equal(
    verifyEntailmentWitness(Object.assign(Object.create(null), genuine), prog, X).code,
    'ENTAILMENT_WITNESS_UNAUTHENTICATED',
    'null-prototype clone'
  );
});
