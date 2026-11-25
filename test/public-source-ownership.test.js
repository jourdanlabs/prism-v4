import test from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags } from '../src/kernel/sentences.js';
import { solveWithWitness } from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

function buildCase(opts = {}) {
  const evA = opts.evA ?? 'lamp';
  const evB = opts.evB ?? 'buzzer';
  const violetASentence =
    opts.violetASentence ?? `For Violet, the ${evA} is positive with probability 20%.`;
  const violetBSentence = opts.violetBSentence ?? null;
  const combined = opts.combined ?? null;
  const amberBSentence =
    'amberBSentence' in opts
      ? opts.amberBSentence
      : `For Amber, the ${evB} is positive with probability 20%.`;
  const amberBTarget = opts.amberBTarget ?? amberBSentence;
  const obsSentence = `The ${evB} reads positive on the selected console.`;
  const sentences = [
    'Each console is either Amber or Violet.',
    'Amber accounts for 40% of consoles.',
    'Violet accounts for 60% of consoles.',
    `The record tracks the ${evA}.`,
    `The record tracks the ${evB}.`,
    `For Amber, the ${evA} is positive with probability 30%.`,
    ...(combined ? [combined] : [violetASentence]),
    ...(amberBSentence ? [amberBSentence] : []),
    ...(violetBSentence ? [violetBSentence] : []),
    obsSentence,
    'What is the probability that the selected console is Amber?',
  ];
  const X = sentences.join(' ');
  const span = (needle, from = 0) => {
    const s = X.indexOf(needle, from);
    assert.notEqual(s, -1, `missing span text: ${needle}`);
    return `[${s},${s + needle.length}]`;
  };
  const trackA = X.indexOf(`The record tracks the ${evA}.`);
  const trackB = X.indexOf(`The record tracks the ${evB}.`);
  const lines = [
    `HYP h0 ${span('Amber')}`,
    `HYP h1 ${span('Violet')}`,
    `PART_EXHAUSTIVE (h0 h1) ${span('Amber')}`,
    `EVENT e0 ${span(evA, trackA)}`,
    `EVENT e1 ${span(evB, trackB)}`,
    `PRIOR h0 2/5 ctx0 ${span(sentences[1])}`,
    `PRIOR h1 3/5 ctx0 ${span(sentences[2])}`,
    `LIK e0 POS GIVEN h0 3/10 ctx0 ${span(sentences[5])}`,
  ];
  if (combined) {
    lines.push(
      `LIK e0 POS GIVEN h1 1/5 ctx0 ${span(combined)}`,
      `LIK e1 POS GIVEN h1 ${opts.combinedBValue ?? '1/5'} ctx0 ${span(combined)}`,
    );
  } else {
    if (opts.includeVioletADecl !== false) {
      lines.push(`LIK e0 POS GIVEN h1 1/5 ctx0 ${span(violetASentence)}`);
    }
    const bTarget = opts.violetBTarget ?? violetBSentence ?? violetASentence;
    const bSpan = opts.violetBSpanNeedle
      ? span(opts.violetBSpanNeedle, X.indexOf(bTarget))
      : span(bTarget);
    lines.push(`LIK e1 POS GIVEN h1 1/5 ctx0 ${bSpan}`);
  }
  const amberBSpan = opts.amberBSpanNeedle
    ? span(opts.amberBSpanNeedle, X.indexOf(amberBTarget))
    : span(amberBTarget);
  lines.push(
    `LIK e1 POS GIVEN h0 ${opts.amberBValue ?? '1/5'} ctx0 ${amberBSpan}`,
    `OBS e1 POS ctx0 ${span(obsSentence)}`,
    'ASK_PROB h0 GIVEN (e1:POS) ctx0',
    'END',
  );
  const prog = toProgram(lines.join('\n'), X.length);
  const tags = autoSentTags(X, prog).map(
    (r) => `SENT_TAG ${r.idx} ${r.tag}${r.ctx ? ` ${r.ctx}` : ''} [${r.span.s},${r.span.e}]`,
  );
  const bytecode = [...lines.slice(0, -2), ...tags, lines[lines.length - 2], 'END'].join('\n');
  return { X, bytecode };
}

function assertExactTwoFifths(decision) {
  assert.equal(decision.status, 'ANSWER');
  assert.equal(R.toString(decision.value), '2/5');
  assert.equal(R.toString(decision.interval.min), '2/5');
  assert.equal(R.toString(decision.interval.max), '2/5');
}

function assertExactFiveEighths(decision) {
  assert.equal(decision.status, 'ANSWER');
  assert.equal(R.toString(decision.value), '5/8');
  assert.equal(R.toString(decision.interval.min), '5/8');
  assert.equal(R.toString(decision.interval.max), '5/8');
}

const VARIANTS = [
  { evA: 'lamp', evB: 'buzzer', relation: 'beside' },
  { evA: 'beacon', evB: 'siren', relation: 'next to' },
];

test('direct solver rejects buzzer likelihood bound to the lamp literal', async () => {
  const { X, bytecode } = buildCase();
  const res = await solveWithWitness(bytecode, X, { offlineSyntheticBench: true });
  assert.equal(res.status, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.equal(res.interval, undefined);
});

test('single incidental wrong-event claim is refused directly and by the gate', async () => {
  for (const v of VARIANTS) {
    const incidental = `For Violet, the ${v.evA} ${v.relation} the ${v.evB} is positive with probability 20%.`;
    const { X, bytecode } = buildCase({
      evA: v.evA,
      evB: v.evB,
      violetASentence: incidental,
      includeVioletADecl: false,
    });
    const direct = await solveWithWitness(bytecode, X, { offlineSyntheticBench: true });
    assert.equal(direct.status, 'REFUSED');
    assert.equal(direct.code, 'UNSTABLE_BINDING');
    const gate = await verifyInvariance({
      X,
      extract: makeSemanticExtractor(bytecode),
      offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'REFUSED');
    assert.equal(gate.code ?? gate.decision?.code, 'UNSTABLE_BINDING');
  }
});

test('correctly owned incidental-location rate is retained with a distinct second-event rate', async () => {
  for (const v of VARIANTS) {
    const incidental = `For Violet, the ${v.evA} ${v.relation} the ${v.evB} is positive with probability 20%.`;
    const distinct = `For Violet, the ${v.evB} is positive with probability 20%.`;
    const { X, bytecode } = buildCase({
      evA: v.evA,
      evB: v.evB,
      violetASentence: incidental,
      violetBSentence: distinct,
    });
    const direct = await solveWithWitness(bytecode, X, { offlineSyntheticBench: true });
    assertExactTwoFifths(direct);
    const gate = await verifyInvariance({
      X,
      extract: makeSemanticExtractor(bytecode),
      offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'PASS');
  }
});

test('distinct equal-value sites give exact 2/5 and preserve the offline control', async () => {
  const distinct = 'For Violet, the buzzer is positive with probability 20%.';
  const { X, bytecode } = buildCase({ violetBSentence: distinct });
  const direct = await solveWithWitness(bytecode, X, { offlineSyntheticBench: true });
  assertExactTwoFifths(direct);
  assert.equal(direct.entailmentWitness, null);
  const gate = await verifyInvariance({
    X,
    extract: makeSemanticExtractor(bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gate.verdict, 'PASS');
});

test('ownership refusal holds despite a supplied witness and relaxed witness config', async () => {
  const { X, bytecode } = buildCase();
  const res = await solveWithWitness(bytecode, X, {
    requireEntailmentWitness: false,
    entailmentWitness: { code: 'ENTAILS', ok: true },
  });
  assert.equal(res.status, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.equal(res.interval, undefined);
});

test('two events predicated separately in one broad span keep distinct equal sites', async () => {
  const combined =
    'For Violet, the lamp is positive with probability 20%, and the buzzer is positive with probability 20%.';
  const { X, bytecode } = buildCase({ combined });
  const direct = await solveWithWitness(bytecode, X, { offlineSyntheticBench: true });
  assertExactTwoFifths(direct);
});

test('adjective inside a locative modifier does not move the event head', async () => {
  const wrongHead =
    'For Violet, the lamp installed beside the tall buzzer is positive with probability 20%.';
  const wrong = buildCase({ violetASentence: wrongHead, includeVioletADecl: false });
  const refused = await solveWithWitness(wrong.bytecode, wrong.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(refused.status, 'REFUSED');
  assert.equal(refused.code, 'UNSTABLE_BINDING');
  const gateRefused = await verifyInvariance({
    X: wrong.X,
    extract: makeSemanticExtractor(wrong.bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gateRefused.verdict, 'REFUSED');
  assert.equal(gateRefused.code ?? gateRefused.decision?.code, 'UNSTABLE_BINDING');

  const reversedHead =
    'For Violet, the buzzer installed beside the tall lamp is positive with probability 20%.';
  const reversed = buildCase({
    violetASentence: reversedHead,
    includeVioletADecl: false,
  });
  const direct = await solveWithWitness(reversed.bytecode, reversed.X, {
    offlineSyntheticBench: true,
  });
  assertExactTwoFifths(direct);
  const gate = await verifyInvariance({
    X: reversed.X,
    extract: makeSemanticExtractor(reversed.bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gate.verdict, 'PASS');
});

test('coordinated locative objects do not coordinate the subject event', async () => {
  const wrongHead =
    'For Violet, the lamp between the panel and the buzzer is positive with probability 20%.';
  const wrong = buildCase({ violetASentence: wrongHead, includeVioletADecl: false });
  const refused = await solveWithWitness(wrong.bytecode, wrong.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(refused.status, 'REFUSED');
  assert.equal(refused.code, 'UNSTABLE_BINDING');
  const gateRefused = await verifyInvariance({
    X: wrong.X,
    extract: makeSemanticExtractor(wrong.bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gateRefused.verdict, 'REFUSED');
  assert.equal(gateRefused.code ?? gateRefused.decision?.code, 'UNSTABLE_BINDING');

  const reversedHead =
    'For Violet, the buzzer between the panel and the lamp is positive with probability 20%.';
  const reversed = buildCase({
    violetASentence: reversedHead,
    includeVioletADecl: false,
  });
  const direct = await solveWithWitness(reversed.bytecode, reversed.X, {
    offlineSyntheticBench: true,
  });
  assertExactTwoFifths(direct);
  const gate = await verifyInvariance({
    X: reversed.X,
    extract: makeSemanticExtractor(reversed.bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gate.verdict, 'PASS');
});

test('transitive object rates with coordinated ellipsis keep distinct owners', async () => {
  const correctCombined =
    'For Violet, the console triggers the lamp with probability 20% and the buzzer with probability 50%.';
  const correct = buildCase({ combined: correctCombined, combinedBValue: '1/2' });
  const direct = await solveWithWitness(correct.bytecode, correct.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(direct.status, 'ANSWER');
  assert.equal(R.toString(direct.value), '4/19');
  assert.equal(R.toString(direct.interval.min), '4/19');
  assert.equal(R.toString(direct.interval.max), '4/19');
  const gate = await verifyInvariance({
    X: correct.X,
    extract: makeSemanticExtractor(correct.bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gate.verdict, 'PASS');

  const swappedCombined =
    'For Violet, the console triggers the lamp with probability 50% and the buzzer with probability 20%.';
  const swapped = buildCase({ combined: swappedCombined, combinedBValue: '1/2' });
  const refused = await solveWithWitness(swapped.bytecode, swapped.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(refused.status, 'REFUSED');
  assert.equal(refused.code, 'UNSTABLE_BINDING');
  const gateRefused = await verifyInvariance({
    X: swapped.X,
    extract: makeSemanticExtractor(swapped.bytecode),
    offlineSyntheticBench: true,
  });
  assert.equal(gateRefused.verdict, 'REFUSED');
  assert.equal(gateRefused.code ?? gateRefused.decision?.code, 'UNSTABLE_BINDING');

  const prepObject =
    'For Violet, the console triggers the lamp beside the buzzer with probability 20%.';
  const wrong = buildCase({ violetASentence: prepObject, includeVioletADecl: false });
  const prepRefused = await solveWithWitness(wrong.bytecode, wrong.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(prepRefused.status, 'REFUSED');
  assert.equal(prepRefused.code, 'UNSTABLE_BINDING');
});

test('coordinated rate inherits appositive subject event, not modifier event', async () => {
  const coordinated =
    'The second indicator, the chime, rings for 50% of Amber consoles including one whose chime sits next to its lantern and 20% of Violet consoles.';
  const correct = buildCase({
    evA: 'lantern',
    evB: 'chime',
    violetBSentence: coordinated,
    amberBSentence: null,
    amberBTarget: coordinated,
    amberBValue: '1/2',
    amberBSpanNeedle: '50%',
    violetBSpanNeedle: '20%',
  });
  assertExactFiveEighths(
    await solveWithWitness(correct.bytecode, correct.X, {
      offlineSyntheticBench: true,
    })
  );

  const wrongEvent = buildCase({
    evA: 'lantern',
    evB: 'chime',
    violetASentence: coordinated,
    violetBSentence: 'For Violet, the chime is positive with probability 20%.',
  });
  const refused = await solveWithWitness(wrongEvent.bytecode, wrongEvent.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(refused.status, 'REFUSED');
  assert.equal(refused.code, 'UNSTABLE_BINDING');
});

test('coordinated rate does not inherit event from unbound leading rate', async () => {
  const coordinated =
    'The second indicator, the chime, rings for 50% of Amber consoles including one whose chime sits next to its lantern and 20% of Violet consoles.';
  // Previous contradictory program: Amber/chime is bound to an independent
  // 20% sentence, the coordinated leading 50% is unbound, and the trailing
  // 20% claims Violet/chime only by inheriting that unclaimed clause.
  const contradictory = buildCase({
    evA: 'lantern',
    evB: 'chime',
    violetBSentence: coordinated,
  });
  const refused = await solveWithWitness(contradictory.bytecode, contradictory.X, {
    offlineSyntheticBench: true,
  });
  assert.equal(refused.status, 'REFUSED');
  assert.equal(refused.code, 'UNSTABLE_BINDING');
});
