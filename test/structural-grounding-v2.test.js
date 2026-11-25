// Structural Extraction & Grounding Upgrade v2 — fresh synthetic tests for
// the four general mechanisms: (1) worded frequency ratios + singular/
// plural event morphology, (2) coordinated two-arm rate grounding +
// equal-conditional-rate independence licenses, (3) query-path-scoped
// premise doubt, (4) conditional-query givens + PRIOR/POSTERIOR_STATED
// shared-site mutual exclusion. Every positive mechanism is paired with
// its refusal counterpart.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { parseLiteral, scanLiterals } from '../src/kernel/literals.js';
import { toProgram } from '../src/kernel/parser.js';
import { solve } from '../src/kernel/solve.js';
import {
  adjacentClauseFindings, autoSentTags, assertsIndependence,
  assertsUniversalRelation, canonicalCtxId, checkSentenceAccounting,
  ctxLabels, expandCtxExclude, inflectedNameRe, inflectedNameVariants,
  irrelevantOtherStratumSeg, irrelevantUnrecordedPriorSeg,
  isInertUncoveredInterval, isInspectionMethodSentence,
  queryConditionalGivenSpan, sentenceMateriality, sentences, subClauses,
  validateLicenseDecls,
} from '../src/kernel/sentences.js';
import { checkSemanticEntailment } from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import {
  ownershipCheck, nameMaps, coordinatedArmForSite, hypNameFromSpan,
} from '../src/invariance/ownership.js';
import {
  makeSemanticExtractor, makeMisbindingExtractor,
} from '../src/invariance/reference-extractors.js';
import { jsonToProgram, resolveSpanAll } from '../src/live/extractor.js';

const fmt = (v) => (v ? `${v.n}/${v.d}` : null);
const show = (x) => JSON.stringify(x, (k, v) => (typeof v === 'bigint' ? v.toString() : v));

function withLedger(X, bytecode) {
  const prog = toProgram(bytecode);
  const lines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return bytecode.replace(/(^|\n)(ASK_PROB)/, `$1${lines.join('\n')}\n$2`);
}

const at = (X, s, from = 0) => {
  const i = X.indexOf(s, from);
  if (i < 0) throw new Error(`fixture needle missing: "${s.slice(0, 40)}"`);
  return `[${i},${i + s.length}]`;
};

// ---------------------------------------------------------------------------
// Chassis A: coordinated arms (equal 1/4 values on Alpha), worded
// frequencies, equal-rate independence, two observations.
// Post(Alpha) = (3/5 * 1/4 * 1/4) / (3/5 * 1/4 * 1/4 + 2/5 * 1/10 * 3/10) = 25/33
// ---------------------------------------------------------------------------
function buildA(opts = {}) {
  const coordA = opts.coordA ??
    'Among Alpha panels, one in four carry a notch and one in four a stripe.';
  const coordB = opts.coordB ??
    'On Beta panels a notch turns up on one in ten and a stripe on three in ten.';
  const indep = opts.indep === undefined
    ? 'On each panel separately, the stripe rate is the same for panels with a notch as for panels without a notch.'
    : opts.indep;
  const obs = opts.obs === undefined
    ? 'This panel was observed with its notch and stripe present.'
    : opts.obs;
  const X = [
    'Every panel is either Alpha or Beta.',
    'Alpha accounts for 3/5 of panels and Beta accounts for 2/5.',
    coordA, coordB,
    ...(indep ? [indep] : []),
    ...(obs ? [obs] : []),
    opts.query ?? 'What is the probability the panel is Alpha?',
  ].join(' ');
  const aS = X.indexOf(coordA);
  const bS = X.indexOf(coordB);
  const needles = {
    'A-notch': ['one in four carry a notch', aS],
    'A-stripe': ['one in four a stripe', aS],
    'B-notch': ['a notch turns up on one in ten', bS],
    'B-stripe': ['a stripe on three in ten', bS],
    ...(opts.likSpanNeedles ?? {}),
  };
  const likSpan = {
    ...Object.fromEntries(
      Object.entries(needles).map(([k, [n, from]]) => [k, at(X, n, from)])
    ),
    ...(opts.likSpans ?? {}),
  };
  const likHyp = { 'A-notch': 'h0', 'A-stripe': 'h0', 'B-notch': 'h1', 'B-stripe': 'h1', ...(opts.likHyps ?? {}) };
  const likVal = { 'A-notch': '1/4', 'A-stripe': '1/4', 'B-notch': '1/10', 'B-stripe': '3/10' };
  const likEv = { 'A-notch': 'e0', 'A-stripe': 'e1', 'B-notch': 'e0', 'B-stripe': 'e1', ...(opts.likEvs ?? {}) };
  const drop = new Set(opts.dropLiks ?? []);
  const likLines = Object.keys(likVal)
    .filter((k) => !drop.has(k))
    .map((k) => `LIK ${likEv[k]} ${opts.likPols?.[k] ?? 'POS'} GIVEN ${likHyp[k]} ${likVal[k]} ctx0 ${likSpan[k]}`);
  const lines = [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'notch', aS)}`,
    `EVENT e1 ${at(X, 'stripe', aS)}`,
    `PRIOR h0 3/5 ctx0 ${at(X, '3/5 of panels')}`,
    `PRIOR h1 2/5 ctx0 ${at(X, '2/5')}`,
    ...likLines,
    ...(indep && opts.condIndep !== false
      ? [`COND_INDEP (e0 e1) GIVEN * ctx0 ${at(X, indep)}`]
      : []),
    ...(obs ? [`OBS e0 POS ctx0 ${at(X, obs)}`, `OBS e1 POS ctx0 ${at(X, obs)}`] : []),
    'ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0',
    'END',
  ];
  return { X, bc: withLedger(X, lines.join('\n')) };
}

// ---------------------------------------------------------------------------
// Chassis B: single event "clipped corner", Ash/Fir.
// Post(Ash | corner) = (2/5*4/5) / (2/5*4/5 + 3/5*1/5) = 8/11
// ---------------------------------------------------------------------------
function buildB(opts = {}) {
  const rate = opts.rate ??
    'The clipped corner appears for 4/5 of Ash panels and 1/5 of Fir panels.';
  const extra = opts.extra ?? [];
  const obs = opts.obs === undefined
    ? 'This panel was observed with a clipped corner.'
    : opts.obs;
  const query = opts.query ?? 'What is the probability the panel is Ash?';
  const X = [
    'Every panel is either Ash or Fir.',
    'Ash accounts for 2/5 of panels and Fir accounts for 3/5.',
    rate, ...extra,
    ...(obs ? [obs] : []),
    query,
  ].join(' ');
  const evNeedle = opts.evNeedle ?? 'clipped corner';
  const lines = [
    `HYP h0 ${at(X, 'Ash')}`,
    `HYP h1 ${at(X, 'Fir')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Ash or Fir')}`,
    `EVENT e0 ${at(X, evNeedle)}`,
    `PRIOR h0 2/5 ctx0 ${at(X, '2/5 of panels')}`,
    `PRIOR h1 3/5 ctx0 ${at(X, '3/5')}`,
    `LIK e0 POS GIVEN h0 ${opts.likA ?? '4/5'} ctx0 ${at(X, opts.likASpan ?? '4/5 of Ash')}`,
    ...(opts.dropFirLik ? [] : [
      `LIK e0 POS GIVEN h1 ${opts.likB ?? '1/5'} ctx0 ${at(X, opts.likBSpan ?? '1/5 of Fir')}`,
    ]),
    ...(obs ? [`OBS e0 POS ctx0 ${at(X, obs)}`] : []),
    `ASK_PROB h0 GIVEN (e0:${opts.givenPol ?? 'POS'}) ctx0`,
    'END',
  ];
  return { X, bc: withLedger(X, lines.join('\n')) };
}

// ======================= 1. worded frequency ratios =======================
test('literals: worded frequency ratios parse to exact reduced rationals', () => {
  const cases = [
    ['one in ten', '1/10'], ['3 in 10', '3/10'], ['two in five', '2/5'],
    ['three slips in ten', '3/10'], ['two cards in five', '2/5'],
    ['one unit in four', '1/4'], ['three out of ten', '3/10'],
    ['7 out of every 20', '7/20'], ['zero in ten', '0/1'],
    ['six in twelve', '1/2'],
  ];
  for (const [text, want] of cases) {
    assert.equal(fmt(parseLiteral(text)), want, text);
    const sites = scanLiterals(`xx ${text} yy`);
    assert.equal(sites.length, 1, text);
    assert.equal(sites[0].text, text, text);
    assert.equal(fmt(sites[0].value), want, text);
    assert.equal(`xx ${text} yy`.slice(sites[0].s, sites[0].e), text, text);
  }
});

test('literals: frequency guards refuse dates, ordinals, zero denominators', () => {
  for (const bad of ['3 in 0', 'three in zero', '3 in 2024', 'first in ten', 'one in the morning']) {
    assert.equal(parseLiteral(bad), null, bad);
  }
  assert.deepEqual(scanLiterals('On 3 in 2024 the log closed.').map((l) => l.text), []);
  // sentence-initial capitalised cardinal still binds; capitalised noun does not
  assert.equal(fmt(scanLiterals('Three slips in ten were marked.')[0].value), '3/10');
  assert.equal(parseLiteral('three Paris in ten'), null);
  // a denominator word is a fraction word, never a counted noun
  assert.equal(fmt(parseLiteral('five eighths')), '5/8');
});

// ======================= 1. morphology ====================================
test('morphology: conservative inflections with strict token boundaries', () => {
  assert.ok(inflectedNameRe('box').test('the boxes'));
  assert.ok(inflectedNameRe('boxes').test('the box'));
  assert.ok(inflectedNameRe('tally').test('tallies'));
  assert.ok(inflectedNameRe('tallies').test('tally'));
  assert.ok(inflectedNameRe('notch').test('notches'));
  assert.ok(inflectedNameRe('clipped corner').test('clipped corners'));
  assert.ok(inflectedNameRe('clipped corners').test('a clipped corner'));
  assert.ok(!inflectedNameRe('corner').test('cornerstone'));
  assert.ok(!inflectedNameRe('box').test('boxwood'));
  assert.ok(!inflectedNameRe('notch').test('notched'));
});

// ======================= 2. coordinated arms ==============================
test('coordinated arms + equal-rate license: exact answer through solve and gate swaps', async () => {
  const { X, bc } = buildA();
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeSemanticExtractor(bc),
  });
  assert.equal(res.verdict, 'PASS', JSON.stringify(res.gateFindings ?? res.slots?.swaps?.filter((s) => !s.ok)));
  assert.equal(fmt(res.decision.value), '25/33');
  assert.ok(res.slots.swaps.every((s) => s.ok !== false));
});

test('coordinated arms: plural event morphology in arm mentions still grounds', () => {
  const { X, bc } = buildA({
    coordA: 'Among Alpha panels, one in four carry notches and one in four stripes.',
    likSpanNeedles: {
      'A-notch': ['one in four carry notches', 0],
      'A-stripe': ['one in four stripes', 0],
    },
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
  const prog = toProgram(bc, X.length);
  assert.deepEqual(
    ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []
  );
});

test('coordinated arms: swapped events at equal values refuse UNSTABLE_BINDING', async () => {
  const { X } = buildA();
  const aS = X.indexOf('Among Alpha');
  const swapped = buildA({
    likSpans: {
      'A-notch': at(X, 'one in four a stripe', aS),
      'A-stripe': at(X, 'one in four carry a notch', aS),
    },
  });
  const prog = toProgram(swapped.bc, X.length);
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(failures.some((f) => f.code === 'UNSTABLE_BINDING'), JSON.stringify(failures));
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeSemanticExtractor(swapped.bc),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
});

test('coordinated arms: both arms bound to the wrong hypothesis refuse', () => {
  const { X, bc } = buildA({ likHyps: { 'A-notch': 'h1', 'A-stripe': 'h1', 'B-notch': 'h0', 'B-stripe': 'h0' } });
  const prog = toProgram(bc, X.length);
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by/.test(f.detail)), JSON.stringify(failures));
});

test('coordinated arms: NEG polarity against a coordinated arm refuses', () => {
  const { X, bc } = buildA({ likPols: { 'A-notch': 'NEG' } });
  const prog = toProgram(bc, X.length);
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(failures.some((f) => f.code === 'UNSTABLE_BINDING'), JSON.stringify(failures));
});

test('coordinated arms: missing sibling arm refuses (ownership + kernel)', () => {
  const { X, bc } = buildA({ dropLiks: ['A-stripe'] });
  const prog = toProgram(bc, X.length);
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /sibling/.test(f.detail)),
    JSON.stringify(failures)
  );
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'INCOMPLETE_FACTORIZATION');
});

test('coordinated arms: persistent misbinding extractor caught by gate swaps', async () => {
  const { X, bc } = buildA();
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeMisbindingExtractor(bc),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
});

// ======================= 2. equal-rate independence =======================
test('equal-rate: whether-form licenses independence', () => {
  const { X, bc } = buildA({
    indep: 'Within each panel, the probability of a stripe is identical whether a notch occurs or not.',
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
});

test('equal-rate: missing independence still refuses UNLICENSED_INDEPENDENCE', () => {
  const { X, bc } = buildA({ indep: 'The cabinet stands beside a stool.', condIndep: false });
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNLICENSED_INDEPENDENCE');
});

test('equal-rate: negated, hedged, restricted and non-licensing forms refuse', () => {
  for (const indep of [
    'On each panel separately, the stripe rate is not the same for panels with a notch as for panels without a notch.',
    'The stripe rate differs for panels with a notch.',
    'It remains unknown whether the stripe rate is the same for panels with a notch as for panels without a notch.',
    'Only for Alpha panels, the stripe rate is the same for panels with a notch as for panels without a notch.',
    'The crew used separate notebooks and never matched them.',
  ]) {
    const { X, bc } = buildA({ indep });
    const d = solve(bc, X);
    assert.equal(d.status, 'REFUSED', indep);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `${indep} -> ${d.code}`);
    const prog = toProgram(bc, X.length);
    assert.equal(validateLicenseDecls(X, prog).length, 1, indep);
  }
});

test('assertsIndependence: equal-rate definition recognised, negations not', () => {
  assert.ok(assertsIndependence('the stripe rate is the same for panels with a notch as for panels without a notch'));
  assert.ok(assertsIndependence('the probability of a stripe is identical whether a notch occurs or not'));
  assert.ok(assertsIndependence('the notch and stripe are independent'));
  assert.ok(!assertsIndependence('the stripe rate is not the same for panels with a notch as for panels without a notch'));
  assert.ok(!assertsIndependence('the stripe rate differs by notch status'));
  assert.ok(!assertsIndependence('unknown whether the stripe rate is the same with a notch'));
  assert.ok(!assertsIndependence('the crew used separate notebooks and never matched them'));
  assert.ok(!assertsIndependence('the notch and stripe are not independent'));
});

// ======================= 3. scoped premise doubt ==========================
test('scoped doubt: unchecked unqueried attribute does not poison the query', async () => {
  for (const extra of [
    ["Its centre hasn't been checked."],
    ['The secondary dial was not inspected.'],
  ]) {
    const { X, bc } = buildB({ extra });
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${extra[0]} -> ${show(d)}`);
    assert.equal(fmt(d.value), '8/11');
    const res = await verifyInvariance({
      offlineSyntheticBench: true, X, extract: makeSemanticExtractor(bc),
    });
    assert.equal(res.verdict, 'PASS', extra[0]);
  }
});

test('scoped doubt: unchecked QUERIED event still refuses SCOPED_OR_RETRACTED_PREMISE', () => {
  const { X, bc } = buildB({ extra: ["Its clipped corner hasn't been checked."] });
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'SCOPED_OR_RETRACTED_PREMISE');
});

test('scoped doubt: retraction and mechanism doubt still poison', () => {
  for (const extra of [
    ['The earlier figures were retracted.'],
    ['The relay connection is still unchecked.'],
    ["That hasn't been checked."],
  ]) {
    const { X, bc } = buildB({ extra });
    const d = solve(bc, X);
    assert.equal(d.status, 'REFUSED', extra[0]);
    assert.equal(d.code, 'SCOPED_OR_RETRACTED_PREMISE', `${extra[0]} -> ${d.code}`);
  }
});

test('scoped doubt: lost required tally refuses INCOMPLETE_FACTORIZATION', () => {
  const { X, bc } = buildB({
    rate: 'The clipped corner appears for 4/5 of Ash panels.',
    extra: ['The crew lost the corner tally for Fir.'],
    dropFirLik: true,
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'INCOMPLETE_FACTORIZATION');
});

// ======================= 4. conditional givens ============================
test('conditional given: query-sentence conditional clause witnesses the given', async () => {
  const { X, bc } = buildB({
    obs: null,
    query: 'What is the probability the panel is Ash, given that it has a clipped corner?',
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '8/11');
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeSemanticExtractor(bc),
  });
  assert.equal(res.verdict, 'PASS');
});

test('conditional given: fronted clause and NEG polarity ground correctly', () => {
  const fronted = buildB({
    obs: null,
    query: 'Given it carries a clipped corner, what is the probability the panel is Ash?',
  });
  assert.equal(fmt(solve(fronted.bc, fronted.X).value), '8/11');
  const neg = buildB({
    obs: null, givenPol: 'NEG',
    query: 'What is the probability the panel is Ash, given that it has no clipped corner?',
  });
  const d = solve(neg.bc, neg.X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '1/7');
});

test('conditional given: ungrounded or polarity-mismatched givens still refuse', () => {
  const plain = buildB({ obs: null });
  assert.equal(solve(plain.bc, plain.X).code, 'MISSING_OBSERVATION_WITNESS');
  const mismatched = buildB({
    obs: null,
    query: 'What is the probability the panel is Ash, given that it has no clipped corner?',
  });
  assert.equal(solve(mismatched.bc, mismatched.X).code, 'MISSING_OBSERVATION_WITNESS');
  const otherEvent = buildB({
    obs: null,
    query: 'What is the probability the panel is Ash, given that it has a brass hinge?',
  });
  assert.equal(solve(otherEvent.bc, otherEvent.X).code, 'MISSING_OBSERVATION_WITNESS');
});

test('conditional given: zero-support stratum still refuses', () => {
  const { X, bc } = buildB({
    rate: 'The clipped corner appears for 0% of Ash panels and 0% of Fir panels.',
    likA: '0/1', likB: '0/1', likASpan: '0% of Ash', likBSpan: '0% of Fir',
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.ok(['DEGENERATE_PARTITION', 'ZERO_SUPPORT_POSSIBLE', 'CONFLICTING_FIGURES'].includes(d.code), d.code);
});

test('morphology: plural rate subject grounds a singular declared event', async () => {
  const { X, bc } = buildB({
    rate: 'Clipped corners appear for 4/5 of Ash panels and 1/5 of Fir panels.',
    evNeedle: 'Clipped corner',
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '8/11');
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeSemanticExtractor(bc),
  });
  assert.equal(res.verdict, 'PASS');
});

// ======================= 4/2. extractor compilation =======================
const COORD_A_SENT =
  'Among Alpha panels, one in four carry a notch and one in four a stripe.';
const INDEP_SENT =
  'On each panel separately, the stripe rate is the same for panels with a notch as for panels without a notch.';
const OBS_SENT = 'This panel was observed with its notch and stripe present.';

function extractorJsonA(likOverrides = {}) {
  return {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Every panel is either Alpha or Beta.' },
      { id: 'h1', name: 'Beta', span: 'Every panel is either Alpha or Beta.' },
    ],
    events: [
      { id: 'e0', name: 'notch', span: 'carry a notch' },
      { id: 'e1', name: 'stripe', span: 'a stripe.' },
    ],
    partition: ['h0', 'h1'],
    partitionSpan: 'Every panel is either Alpha or Beta.',
    priors: [
      { hyp: 'h0', value: '3/5', span: 'Alpha accounts for 3/5 of panels' },
      { hyp: 'h1', value: '2/5', span: 'Beta accounts for 2/5' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/4', span: 'one in four carry a notch' },
      { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/4', span: 'one in four a stripe' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/10', span: 'a notch turns up on one in ten' },
      { ev: 'e1', pol: 'POS', hyp: 'h1', value: '3/10', span: 'a stripe on three in ten' },
    ].map((l, i) => ({ ...l, ...(likOverrides[i] ?? {}) })),
    posteriors: [],
    obs: [
      { ev: 'e0', pol: 'POS', span: OBS_SENT },
      { ev: 'e1', pol: 'POS', span: OBS_SENT },
    ],
    nonAsserted: [],
    condIndep: [{ evs: ['e0', 'e1'], span: INDEP_SENT }],
    detLinks: [],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, { ev: 'e1', pol: 'POS' }] },
  };
}

test('extractor: coordinated arms compile and solve (single-literal quotes)', () => {
  const { X } = buildA();
  const { program, errors } = jsonToProgram(extractorJsonA(), X);
  assert.deepEqual(errors, []);
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
});

test('extractor: broad two-literal coordinated quotes attribute distinct equal sites', () => {
  const { X } = buildA();
  const j = extractorJsonA({ 0: { span: COORD_A_SENT }, 1: { span: COORD_A_SENT } });
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
});

test('extractor: duplicate literal-only quotes resolve to distinct arm sites', () => {
  const { X } = buildA();
  const j = extractorJsonA({ 0: { span: 'one in four' }, 1: { span: 'one in four' } });
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  const alphaLiks = program.decls.filter((d2) => d2.op === 'LIK' && d2.hyp === 'h0');
  assert.notEqual(alphaLiks[0].span.s, alphaLiks[1].span.s);
  assert.equal(fmt(solve(program, X).value), '25/33');
});

test('extractor: swapped coordinated arm is rejected at compile time', () => {
  const { X } = buildA();
  const j = extractorJsonA({
    0: { span: 'one in four a stripe' },
    1: { span: 'one in four carry a notch' },
  });
  const { errors } = jsonToProgram(j, X);
  assert.ok(errors.some((e) => /UNSTABLE_BINDING/.test(e)), JSON.stringify(errors));
});

test('extractor: unsafe event names and hypothesis detLinks are rejected', () => {
  const { X } = buildA();
  const badName = extractorJsonA();
  badName.events = [{ id: 'e0', name: 'lamp and buzzer', span: 'carry a notch' }, badName.events[1]];
  assert.ok(jsonToProgram(badName, X).errors.some((e) => /unsafe EVENT/.test(e)));
  const badLink = extractorJsonA();
  badLink.detLinks = [{ a: 'h0', rel: 'EQUIV', b: 'e1', span: INDEP_SENT }];
  assert.ok(jsonToProgram(badLink, X).errors.some((e) => /events only/.test(e)));
});

// ================== 4. PRIOR / POSTERIOR_STATED exclusion ==================
const BOOKS_X = [
  'Every book is either Linden or Rowan.',
  'Linden accounts for 1/2 of books and Rowan accounts for 1/2.',
  'Of the books with crimson thread, 7/11 were repaired by Linden.',
  'A book with crimson thread was observed.',
  'What is the probability it is Linden?',
].join(' ');

test('extractor: same literal bound as PRIOR and POSTERIOR_STATED conflicts', () => {
  const j = {
    hyps: [
      { id: 'h0', name: 'Linden', span: 'Every book is either Linden or Rowan.' },
      { id: 'h1', name: 'Rowan', span: 'Every book is either Linden or Rowan.' },
    ],
    events: [{ id: 'e0', name: 'crimson thread', span: 'with crimson thread' }],
    partition: ['h0', 'h1'],
    partitionSpan: 'Every book is either Linden or Rowan.',
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Linden accounts for 1/2 of books' },
      { hyp: 'h1', value: '1/2', span: 'Rowan accounts for 1/2' },
      { hyp: 'h0', value: '7/11', span: '7/11 were repaired by Linden' },
    ],
    liks: [],
    posteriors: [
      { hyp: 'h0', ev: 'e0', pol: 'POS', value: '7/11', span: '7/11 were repaired by Linden' },
    ],
    obs: [{ ev: 'e0', pol: 'POS', span: 'A book with crimson thread was observed.' }],
    nonAsserted: [], condIndep: [], detLinks: [],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
  const { errors } = jsonToProgram(j, BOOKS_X);
  assert.ok(errors.some((e) => /shared literal/i.test(e)), JSON.stringify(errors));
});

test('ownership: PRIOR bound in an event-conditioned share clause refuses', () => {
  const bc = withLedger(BOOKS_X, [
    `HYP h0 ${at(BOOKS_X, 'Linden')}`,
    `HYP h1 ${at(BOOKS_X, 'Rowan')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(BOOKS_X, 'Linden or Rowan')}`,
    `EVENT e0 ${at(BOOKS_X, 'crimson thread')}`,
    `PRIOR h0 7/11 ctx0 ${at(BOOKS_X, '7/11 were repaired by Linden')}`,
    `PRIOR h1 1/2 ctx0 ${at(BOOKS_X, 'Rowan accounts for 1/2')}`,
    `OBS e0 POS ctx0 ${at(BOOKS_X, 'A book with crimson thread was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const prog = toProgram(bc, BOOKS_X.length);
  const failures = ownershipCheck({ X: BOOKS_X, decls: prog.decls, ...nameMaps(BOOKS_X, prog), prog });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /subgroup share/.test(f.detail)),
    JSON.stringify(failures)
  );
});

// ================== combined end-to-end probes ============================
test('combined: noun-form frequency arm + conditional-on givens (no OBS sentence)', () => {
  const { X, bc } = buildA({
    coordB: 'On Beta panels a notch turns up on one in ten and a stripe on three slips in ten.',
    obs: null,
    likSpanNeedles: { 'B-stripe': ['three slips in ten', 0] },
    query: 'What is the probability the panel is Alpha, conditional on a notch and a stripe?',
  });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
  const direct = buildB({
    obs: null,
    query: 'What is the probability the panel is Ash, conditional on a clipped corner?',
  });
  assert.equal(fmt(solve(direct.bc, direct.X).value), '8/11');
});

test('coordinated arms: a negated arm withholds coordinated grounding', () => {
  const coordNeg = 'Among Alpha panels, one in four carry a notch and never a stripe on one in five.';
  const { X, bc } = buildA({
    coordA: coordNeg,
    likSpanNeedles: {
      'A-notch': ['one in four carry a notch', 0],
      'A-stripe': ['one in five', 0],
    },
  });
  const prog = toProgram(bc, X.length);
  // LIK values in this fixture still declare 1/4 for the stripe arm, whose
  // literal is 1/5: no compatible site exists and ownership refuses.
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(failures.some((f) => f.code === 'UNSTABLE_BINDING'), JSON.stringify(failures));
});

// ===========================================================================
// R2: clean scrub + 4 general grounding fixes
// ===========================================================================

// ---- R2-2: event-name word-count guard scoped to coordinated arms ---------
test('R2 extractor: >3-word single-event name compiles and solves', () => {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
    'The catch at the north stop appears for 4/5 of Alpha units and 1/5 of Beta units.',
    'This unit was observed with the catch at the north stop.',
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const j = {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Every unit is either Alpha or Beta.' },
      { id: 'h1', name: 'Beta', span: 'Every unit is either Alpha or Beta.' },
    ],
    events: [
      { id: 'e0', name: 'catch at the north stop', span: 'catch at the north stop appears' },
    ],
    partition: ['h0', 'h1'],
    partitionSpan: 'Every unit is either Alpha or Beta.',
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2 of units' },
      { hyp: 'h1', value: '1/2', span: 'Beta accounts for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '4/5', span: '4/5 of Alpha units' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/5', span: '1/5 of Beta units' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'This unit was observed with the catch at the north stop.' }],
    nonAsserted: [], condIndep: [], detLinks: [],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
});

test('R2 coordinated arms: >3-word event name withholds coordinated attribution', () => {
  const X =
    'Among Alpha panels, one in four show the catch at the north stop and one in five a stripe.';
  const site = scanLiterals(X).find((l) => fmt(l.value) === '1/4');
  const hyps = [{ id: 'h0', name: 'Alpha' }, { id: 'h1', name: 'Beta' }];
  assert.equal(
    coordinatedArmForSite(X, site, [
      { id: 'e0', name: 'catch at the north stop' },
      { id: 'e1', name: 'stripe' },
    ], hyps),
    null
  );
  const short = buildA();
  const aSite = scanLiterals(short.X.slice(short.X.indexOf('Among Alpha')))
    .map((l) => ({ ...l, s: l.s + short.X.indexOf('Among Alpha'), e: l.e + short.X.indexOf('Among Alpha') }))
    .find((l) => fmt(l.value) === '1/4');
  assert.ok(coordinatedArmForSite(short.X, aSite, [
    { id: 'e0', name: 'notch' }, { id: 'e1', name: 'stripe' },
  ], hyps));
});

// ---- R2-3: ownerOfLiteral production/possessive/prepositional forms -------
// Single-event (lamp) Alpha/Beta unit chassis with caller-supplied prior
// and likelihood sentences. Decl specs: {sent, needle, value, hyp?}.
function buildUnit({ sents, priors, liks }) {
  const X = [
    'Every unit is either Alpha or Beta.',
    ...sents,
    'This unit was observed with a lamp.',
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const spanOf = (d) => at(X, d.needle, X.indexOf(d.sent));
  const lines = [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'lamp')}`,
    ...priors.map((p, i) => `PRIOR ${p.hyp ?? `h${i}`} ${p.value} ctx0 ${spanOf(p)}`),
    ...liks.map((l, i) => `LIK e0 POS GIVEN ${l.hyp ?? `h${i}`} ${l.value} ctx0 ${spanOf(l)}`),
    `OBS e0 POS ctx0 ${at(X, 'This unit was observed with a lamp.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ];
  return { X, bc: withLedger(X, lines.join('\n')) };
}
const LAMP_RATE =
  'The lamp lights for 1/2 of Alpha units and 1/4 of Beta units.';
const lampLiks = (hyps = {}) => [
  { sent: LAMP_RATE, needle: '1/2 of Alpha', value: '1/2', hyp: hyps.a },
  { sent: LAMP_RATE, needle: '1/4 of Beta', value: '1/4', hyp: hyps.b },
];
const stdPriors = [
  { sent: 'Alpha accounts for 1/2 of units.', needle: 'Alpha accounts for 1/2', value: '1/2' },
  { sent: 'Beta accounts for 1/2 of units.', needle: 'Beta accounts for 1/2', value: '1/2' },
];

test('R2 ownership: active production-verb priors ground; swapped owners refuse', async () => {
  const priors = [
    { sent: 'Alpha made two thirds of them.', needle: 'Alpha made two thirds', value: '2/3' },
    { sent: 'Beta made the other third.', needle: 'Beta made the other third', value: '1/3' },
  ];
  const sents = [...priors.map((p) => p.sent), LAMP_RATE];
  const { X, bc } = buildUnit({ sents, priors, liks: lampLiks() });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
  const prog = toProgram(bc, X.length);
  assert.deepEqual(
    ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []
  );
  const swapped = buildUnit({
    sents,
    priors: [{ ...priors[0], hyp: 'h1' }, { ...priors[1], hyp: 'h0' }],
    liks: lampLiks(),
  });
  const sprog = toProgram(swapped.bc, X.length);
  const failures = ownershipCheck({
    X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog,
  });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by/.test(f.detail)),
    JSON.stringify(failures)
  );
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeSemanticExtractor(bc),
  });
  assert.equal(res.verdict, 'PASS');
  assert.equal(fmt(res.decision.value), '4/5');
});

test('R2 ownership: pre-nominal likelihoods ground; swapped owners refuse', () => {
  const likSents = [
    'Half the Alpha panels carry a lamp.',
    'A quarter of the Beta panels carry a lamp.',
  ];
  const liks = [
    { sent: likSents[0], needle: likSents[0], value: '1/2' },
    { sent: likSents[1], needle: likSents[1], value: '1/4' },
  ];
  const sents = [...stdPriors.map((p) => p.sent), ...likSents];
  const { X, bc } = buildUnit({ sents, priors: stdPriors, liks });
  // the unit chassis queries "unit"; panels are the rate sentences' noun
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '2/3');
  const swapped = buildUnit({
    sents, priors: stdPriors,
    liks: [{ ...liks[0], hyp: 'h1' }, { ...liks[1], hyp: 'h0' }],
  });
  const sprog = toProgram(swapped.bc, X.length);
  const failures = ownershipCheck({
    X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog,
  });
  assert.ok(failures.some((f) => f.code === 'UNSTABLE_BINDING'), JSON.stringify(failures));
});

test('R2 ownership: origin prepositional priors ground (came off / printed on / came from)', () => {
  for (const alphaSent of [
    'Half the units came off the Alpha line.',
    'Half the units were printed on the Alpha press.',
  ]) {
    const priors = [
      { sent: alphaSent, needle: alphaSent, value: '1/2' },
      { sent: 'The other half came from Beta.', needle: 'The other half came from Beta.', value: '1/2' },
    ];
    const likSents = ['The lamp lights for 4/5 of Alpha units and 1/5 of Beta units.'];
    const liks = [
      { sent: likSents[0], needle: '4/5 of Alpha', value: '4/5' },
      { sent: likSents[0], needle: '1/5 of Beta', value: '1/5' },
    ];
    const { X, bc } = buildUnit({
      sents: [...priors.map((p) => p.sent), ...likSents], priors, liks,
    });
    const prog = toProgram(bc, X.length);
    assert.deepEqual(
      ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), [],
      alphaSent
    );
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${alphaSent} -> ${show(d)}`);
    assert.equal(fmt(d.value), '4/5');
    const swapped = buildUnit({
      sents: [...priors.map((p) => p.sent), ...likSents],
      priors: [{ ...priors[0], hyp: 'h1' }, { ...priors[1], hyp: 'h0' }],
      liks,
    });
    const sprog = toProgram(swapped.bc, X.length);
    assert.ok(
      ownershipCheck({ X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog })
        .some((f) => f.code === 'UNSTABLE_BINDING'),
      alphaSent
    );
  }
});

// ---- R2-3: multi-word hyphenation ------------------------------------------
test('R2 morphology: space-or-hyphen matching for multi-word event names', () => {
  assert.ok(inflectedNameRe('amber tag').test('an amber-tag'));
  assert.ok(inflectedNameRe('amber tag').test('amber-tags'));
  assert.ok(inflectedNameRe('amber tags').test('amber-tag'));
  assert.ok(inflectedNameRe('amber-tag').test('amber tags'));
  assert.ok(inflectedNameRe('green mark').test('a green-mark'));
  assert.ok(!inflectedNameRe('amber tag').test('amber-tagged'));
  assert.ok(!inflectedNameRe('amber tag').test('ambertag'));
});

test('R2 hyphenated event: declared "amber tag" grounds "amber-tags" rate text', () => {
  const X = [
    'Every panel is either Alpha or Beta.',
    'Alpha accounts for 1/2 of panels and Beta accounts for 1/2.',
    'Amber-tags appear for 4/5 of Alpha panels and 1/5 of Beta panels.',
    'This panel was observed with an amber tag.',
    'What is the probability the panel is Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'amber tag')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of panels')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'This panel was observed with an amber tag.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const prog = toProgram(bc, X.length);
  assert.deepEqual(
    ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []
  );
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
});

// ---- R2-3: contrasting-adjective equal-rate independence ------------------
test('R2 equal-rate: contrasting-adjective arms license independence', () => {
  const indep =
    'On each panel separately, the stripe rate is the same for panels with a deep notch as for panels with a shallow notch.';
  const { X, bc } = buildA({ indep });
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
  assert.ok(assertsIndependence(indep));
});

test('R2 equal-rate: identical or negated contrasting arms refuse', () => {
  for (const indep of [
    'On each panel separately, the stripe rate is the same for panels with a deep notch as for panels with a deep notch.',
    'On each panel separately, the stripe rate is not the same for panels with a deep notch as for panels with a shallow notch.',
    'The stripe rate differs for panels with a deep notch and panels with a shallow notch.',
  ]) {
    assert.ok(!assertsIndependence(indep), indep);
    const { X, bc } = buildA({ indep });
    const d = solve(bc, X);
    assert.equal(d.status, 'REFUSED', indep);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', indep);
    const prog = toProgram(bc, X.length);
    assert.equal(validateLicenseDecls(X, prog).length, 1, indep);
  }
});

// ---- R2-4: hypothesis head tokens & number words excluded from deriveCtx --
test('R2 ctxLabels: multi-word hyp heads and number words are not populations', () => {
  const excl = new Set(['alpha press', 'beta line', 'green mark']);
  assert.deepEqual([...ctxLabels('What is the probability it came from Alpha?', excl)], []);
  assert.deepEqual([...ctxLabels('Half the units came off the Alpha line', excl)], []);
  assert.deepEqual([...ctxLabels('x Half the units were counted', excl)], []);
  assert.deepEqual([...ctxLabels('x Three panels were counted', excl)], []);
  // genuine stratum labels survive untouched
  assert.ok(ctxLabels('The North panels were counted', excl).has('north'));
  assert.ok(ctxLabels('A day shift panel was observed', excl).has('day shift'));
  assert.ok(ctxLabels('A night-shift panel was observed', excl).has('night shift'));
});

function multiWordHypJson() {
  return {
    hyps: [
      { id: 'h0', name: 'Alpha press', span: 'Every unit came from the Alpha press or the Beta line.' },
      { id: 'h1', name: 'Beta line', span: 'Every unit came from the Alpha press or the Beta line.' },
    ],
    events: [{ id: 'e0', name: 'lamp', span: 'lamp lights' }],
    partition: ['h0', 'h1'],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Half the units came off the Alpha line.' },
      { hyp: 'h1', value: '1/2', span: 'The other half came from Beta.' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '4/5', span: '4/5 of Alpha units' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/5', span: '1/5 of Beta units' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'This unit was observed with a lamp.' }],
    nonAsserted: [], condIndep: [], detLinks: [],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
}

test('R2 extractor: multi-word hyp names and "Half" do not pollute deriveCtx', () => {
  const X = [
    'Every unit came from the Alpha press or the Beta line.',
    'Half the units came off the Alpha line.',
    'The other half came from Beta.',
    'The lamp lights for 4/5 of Alpha units and 1/5 of Beta units.',
    'This unit was observed with a lamp.',
    'What is the probability it came from Alpha?',
  ].join(' ');
  const { program, errors } = jsonToProgram(multiWordHypJson(), X);
  assert.deepEqual(errors, []);
  assert.equal(program.query.ctx, 'ctx0', show(program.query));
  for (const d of program.decls) {
    if (d.ctx) assert.ok(!/alpha|beta|half/.test(d.ctx), `${d.op} ${d.ctx}`);
  }
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
});

test('R2 extractor: a genuine North stratum still refuses CONTEXT_MISMATCH', () => {
  const X = [
    'Every unit came from the Alpha press or the Beta line.',
    'Half the units came off the Alpha line.',
    'The other half came from Beta.',
    'In the North, the lamp lights for 4/5 of Alpha units and 1/5 of Beta units.',
    'This unit was observed with a lamp.',
    'What is the probability it came from Alpha?',
  ].join(' ');
  const { program, errors } = jsonToProgram(multiWordHypJson(), X);
  assert.deepEqual(errors, []);
  const d = solve(program, X);
  assert.equal(d.status, 'REFUSED', show(d));
  assert.equal(d.code, 'CONTEXT_MISMATCH');
});

// ---- R2-5: direct stated posterior without spurious zero support ----------
const POST_X = [
  'Every unit is either Alpha or Beta.',
  'Of the units with a green mark, 7/11 came from Alpha.',
  'A unit with a green mark was observed.',
  'What is the probability it came from Alpha?',
].join(' ');
function postBc(X, ask = 'ASK_PROB h0 GIVEN (e0:POS) ctx0') {
  return withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'green mark')}`,
    `POSTERIOR_STATED h0 GIVEN e0 POS 7/11 ctx0 ${at(X, '7/11 came from Alpha')}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a green mark was observed.')}`,
    ask,
    'END',
  ].join('\n'));
}

test('R2 direct posterior: stated P(H|E) certifies with no priors or likelihoods', () => {
  const d = solve(postBc(POST_X), POST_X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '7/11');
  assert.equal(d.via, 'POSTERIOR_STATED');
  const neg = solve(postBc(POST_X, 'ASK_PROB NOT h0 GIVEN (e0:POS) ctx0'), POST_X);
  assert.equal(neg.status, 'ANSWER', show(neg));
  assert.equal(fmt(neg.value), '4/11');
});

test('R2 direct posterior: priors present, zero liks, and contradictions still refuse', () => {
  // priors without likelihoods: the V2 replay contract keeps refusing
  const priorProg = `HYP h0 [0,10]\nHYP h1 [10,20]\nPART_EXHAUSTIVE (h0 h1) [0,20]\n` +
    `EVENT e0 [30,40]\nPRIOR h0 2/5 ctx0 [0,10]\nPRIOR h1 3/5 ctx0 [10,20]\n` +
    `POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 [30,40]\nOBS e0 POS ctx0 [50,60]\n` +
    `ASK_PROB h0 GIVEN (e0:POS) ctx0\nEND`;
  assert.equal(solve(priorProg, null).code, 'ZERO_SUPPORT_POSSIBLE');
  // every likelihood zero: evidence impossible under both hypotheses
  const zeroProg = priorProg.replace(
    'POSTERIOR_STATED',
    'LIK e0 POS GIVEN h0 0/1 ctx0 [30,40]\nLIK e0 POS GIVEN h1 0/1 ctx0 [40,50]\nPOSTERIOR_STATED'
  );
  assert.equal(solve(zeroProg, null).code, 'DEGENERATE_PARTITION');
  // full factorization computing 1/2 contradicts the stated 4/5
  const contraProg = priorProg
    .replace('PRIOR h0 2/5', 'PRIOR h0 1/2')
    .replace('PRIOR h1 3/5', 'PRIOR h1 1/2')
    .replace(
      'POSTERIOR_STATED',
      'LIK e0 POS GIVEN h0 1/2 ctx0 [30,40]\nLIK e0 POS GIVEN h1 1/2 ctx0 [40,50]\nPOSTERIOR_STATED'
    );
  assert.equal(solve(contraProg, null).code, 'CONFLICTING_FIGURES');
  // no matching stated posterior + no factors: no certification path at all
  const noPost = postBc(POST_X).replace(/POSTERIOR_STATED[^\n]*\n/, '');
  assert.equal(solve(noPost, POST_X).status, 'REFUSED');
});

// ===========================================================================
// R3 (v3) — six general structural fixes. Generic synthetic fixtures only.
// ===========================================================================

// ---- R3-1: exact-name fallback for non-numeric spans ----------------------
const R3_UNIT_X = [
  'Every unit is either Alpha or Beta.',
  'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
  'The lamp lights for 4/5 of Alpha units and 1/5 of Beta units.',
  'This unit was observed with a lamp.',
  'What is the probability it came from Alpha?',
].join(' ');
function r3UnitJson(over = {}) {
  return {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Every unit is either Alpha or Beta.' },
      { id: 'h1', name: 'Beta', span: 'Every unit is either Alpha or Beta.' },
    ],
    events: [{ id: 'e0', name: 'lamp', span: 'lamp lights' }],
    partition: ['h0', 'h1'],
    partitionSpan: 'Every unit is either Alpha or Beta.',
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2 of units' },
      { hyp: 'h1', value: '1/2', span: 'Beta accounts for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '4/5', span: '4/5 of Alpha units' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/5', span: '1/5 of Beta units' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'observed with a lamp' }],
    nonAsserted: [], condIndep: [], detLinks: [],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
    ...over,
  };
}

test('R3 extractor: paraphrased HYP/EVENT spans fall back to the exact name', () => {
  const j = r3UnitJson();
  // the optional partition proof keeps demanding each hyp's own verbatim
  // quote, so a paraphrasing model simply omits it
  delete j.partitionSpan;
  // model paraphrases: neither quote occurs in R3_UNIT_X, both names do
  j.hyps[1] = { id: 'h1', name: 'Beta', span: 'repaired by Beta' };
  j.events[0] = { id: 'e0', name: 'lamp', span: 'lamp glows brightly' };
  const { program, errors } = jsonToProgram(j, R3_UNIT_X);
  assert.deepEqual(errors, []);
  const hyp1 = program.decls.find((d) => d.op === 'HYP' && d.id === 'h1');
  assert.equal(R3_UNIT_X.slice(hyp1.span.s, hyp1.span.e), 'Beta');
  const ev0 = program.decls.find((d) => d.op === 'EVENT');
  assert.equal(R3_UNIT_X.slice(ev0.span.s, ev0.span.e), 'lamp');
  const d = solve(program, R3_UNIT_X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
});

test('R3 extractor: name fallback stays closed when the name never occurs', () => {
  const j = r3UnitJson();
  j.hyps[1] = { id: 'h1', name: 'Omega', span: 'repaired by Omega' };
  const { errors } = jsonToProgram(j, R3_UNIT_X);
  assert.ok(errors.some((e) => /unresolvable span for HYP h1/.test(e)), show(errors));
  // and a paraphrased NUMERIC span is never rescued by any fallback
  const j2 = r3UnitJson();
  j2.priors[0] = { hyp: 'h0', value: '1/2', span: 'Alpha holds one half of the stock' };
  const r2 = jsonToProgram(j2, R3_UNIT_X);
  assert.ok(r2.errors.some((e) => /unresolvable span for PRIOR/.test(e)), show(r2.errors));
});

test('R3 extractor: hyphenated synonym event span prefers the canonical name', () => {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
    'The amber notch appears for 4/5 of Alpha units and 1/5 of Beta units.',
    'This unit was observed with an amber-shaped notch.',
    'What is the probability it came from Alpha?',
  ].join(' ');
  const j = r3UnitJson();
  j.events[0] = { id: 'e0', name: 'amber notch', span: 'amber-shaped notch' };
  j.obs[0] = { ev: 'e0', pol: 'POS', span: 'observed with an amber-shaped notch' };
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  assert.equal(nameMaps(X, program).evNameById.get('e0'), 'amber notch');
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
});

test('R3 extractor: NON_ASSERTED leading-article typo snaps to the sentence', () => {
  const X = [
    'Every unit is either Alpha or Beta.',
    'The old tally sheet quoted an early estimate for the whole stock.',
    'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
    'The lamp lights for 4/5 of Alpha units and 1/5 of Beta units.',
    'This unit was observed with a lamp.',
    'What is the probability it came from Alpha?',
  ].join(' ');
  const j = r3UnitJson();
  j.nonAsserted = [
    { span: 'A old tally sheet quoted an early estimate for the whole stock.', reason: 'QUOTED' },
  ];
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  const na = program.decls.find((d) => d.op === 'NON_ASSERTED');
  assert.equal(
    X.slice(na.span.s, na.span.e),
    'The old tally sheet quoted an early estimate for the whole stock.'
  );
  assert.equal(solve(program, X).status, 'ANSWER', show(solve(program, X)));
  // a body that differs beyond the article never snaps
  const j2 = r3UnitJson();
  j2.nonAsserted = [
    { span: 'A old tally sheet quoted a later estimate for the whole stock.', reason: 'QUOTED' },
  ];
  const r2 = jsonToProgram(j2, X);
  assert.ok(r2.errors.some((e) => /unresolvable span for NON_ASSERTED/.test(e)), show(r2.errors));
});

// ---- R3-2: lowercase hypothesis names --------------------------------------
const R3_LOWER_X = [
  'In the upper bay, three quarters are copper and the remaining quarter are tin.',
  "The lamp lights for one quarter of the upper bay's copper units and for half of its tin units.",
  'A unit was observed with a lamp.',
  'What is the probability it is copper?',
].join(' ');
function lowerBc(swapLiks = false) {
  const X = R3_LOWER_X;
  const h = swapLiks ? ['h1', 'h0'] : ['h0', 'h1'];
  return withLedger(X, [
    `HYP h0 ${at(X, 'copper')}`,
    `HYP h1 ${at(X, 'tin', 40)}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'copper and the remaining quarter are tin')}`,
    `EVENT e0 ${at(X, 'lamp')}`,
    `PRIOR h0 3/4 ctx0 ${at(X, 'three quarters are copper')}`,
    `PRIOR h1 1/4 ctx0 ${at(X, 'the remaining quarter are tin')}`,
    `LIK e0 POS GIVEN ${h[0]} 1/4 ctx0 ${at(X, "one quarter of the upper bay's copper units")}`,
    `LIK e0 POS GIVEN ${h[1]} 1/2 ctx0 ${at(X, 'half of its tin units')}`,
    `OBS e0 POS ctx0 ${at(X, 'observed with a lamp')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
}

test('R3 ownership: lowercase copper/tin hypotheses ground and answer', () => {
  const X = R3_LOWER_X;
  const bc = lowerBc();
  const prog = toProgram(bc, X.length);
  assert.equal(nameMaps(X, prog).nameById.get('h0'), 'copper');
  assert.equal(nameMaps(X, prog).nameById.get('h1'), 'tin');
  // a narrowed lowercase word is a name; a bare stop-word never is
  const i = X.indexOf('copper');
  assert.equal(hypNameFromSpan(X, { s: i, e: i + 6 }), 'copper');
  assert.equal(hypNameFromSpan('xx the yy', { s: 3, e: 6 }), null);
  assert.deepEqual(ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '3/5');
});

test('R3 ownership: lowercase wrong-owner bindings still refuse', () => {
  const X = R3_LOWER_X;
  const sprog = toProgram(lowerBc(true), X.length);
  const failures = ownershipCheck({
    X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog,
  });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by "(?:copper|tin)"/.test(f.detail)),
    JSON.stringify(failures)
  );
});

// ---- R3-3a: leading-verb gerund inflection ----------------------------------
function catchBc({ query: q } = {}) {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
    'The catch at the north stop appears for 4/5 of Alpha units and 1/5 of Beta units.',
    q ?? 'What is the probability the unit is Alpha, conditional on its catching at the north stop?',
  ].join(' ');
  const lines = [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'catch at the north stop')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of units')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ];
  return { X, bc: withLedger(X, lines.join('\n')) };
}

test('R3 morphology: verb-phrase events inflect the leading verb', () => {
  assert.ok(inflectedNameVariants('catch at the north stop').includes('catching at the north stop'));
  assert.ok(inflectedNameVariants('catch at the north stop').includes('caught at the north stop'));
  assert.ok(inflectedNameRe('fire the relay').test('its firing the relay'));
  assert.ok(inflectedNameRe('fire the relay').test('it fires the relay'));
  assert.ok(inflectedNameRe('trip the main breaker').test('its tripping the main breaker'));
  assert.ok(inflectedNameRe('pass the pressure check').test('it passes the pressure check'));
  // a plain noun phrase gains no verb inflections
  assert.ok(!inflectedNameVariants('amber notch').some((v) => v.includes('ambering')));
});

test('R3 conditional given: gerund clause witnesses the given', () => {
  const { X, bc } = catchBc();
  const prog = toProgram(bc, X.length);
  const span = queryConditionalGivenSpan(X, prog, 'e0', 'POS');
  assert.ok(span, 'conditional clause span');
  assert.match(X.slice(span.s, span.e), /catching at the north stop/);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
  // no conditional clause at all: the given stays ungrounded
  const plain = catchBc({ query: 'What is the probability the unit is Alpha?' });
  assert.equal(solve(plain.bc, plain.X).code, 'MISSING_OBSERVATION_WITNESS');
  // a conditional clause over a DIFFERENT event never witnesses this given
  const other = catchBc({
    query: 'What is the probability the unit is Alpha, conditional on its passing the toll gate?',
  });
  assert.equal(solve(other.bc, other.X).code, 'MISSING_OBSERVATION_WITNESS');
});

// ---- R3-3b: bare head-noun event anaphora -----------------------------------
test('R3 ownership: bare head noun grounds its modified event when unambiguous', () => {
  const X = [
    'Every panel is either Alpha or Beta.',
    'Alpha accounts for 1/2 of panels and Beta accounts for 1/2.',
    'A small crack showed up often.',
    'The crack appeared on 4/5 of Alpha panels and 1/5 of Beta panels.',
    'This panel was observed with a small crack.',
    'What is the probability the panel is Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'small crack')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of panels')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'observed with a small crack')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const prog = toProgram(bc, X.length);
  assert.deepEqual(ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
  // singular/plural head anaphora: "tiny pit" introduced, "pits" rated
  const Xp = X.replace(/small crack/g, 'tiny pit')
    .replace('The crack appeared', 'Pits appeared');
  const bcp = withLedger(Xp, [
    `HYP h0 ${at(Xp, 'Alpha')}`,
    `HYP h1 ${at(Xp, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(Xp, 'Alpha or Beta')}`,
    `EVENT e0 ${at(Xp, 'tiny pit')}`,
    `PRIOR h0 1/2 ctx0 ${at(Xp, '1/2 of panels')}`,
    `PRIOR h1 1/2 ctx0 ${at(Xp, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(Xp, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(Xp, '1/5 of Beta')}`,
    `OBS e0 POS ctx0 ${at(Xp, 'observed with a tiny pit')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const progp = toProgram(bcp, Xp.length);
  assert.deepEqual(ownershipCheck({ X: Xp, decls: progp.decls, ...nameMaps(Xp, progp), prog: progp }), []);
});

test('R3 ownership: a shared head noun keeps the bare form ambiguous', () => {
  const X = [
    'Every panel is either Alpha or Beta.',
    'Alpha accounts for 1/2 of panels and Beta accounts for 1/2.',
    'A red lamp and a blue lamp hang over the bench.',
    'The lamp appeared on 4/5 of Alpha panels and 1/5 of Beta panels.',
    'This panel was observed with a red lamp.',
    'What is the probability the panel is Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'red lamp')}`,
    `EVENT e1 ${at(X, 'blue lamp')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of panels')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'observed with a red lamp')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const prog = toProgram(bc, X.length);
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /event witness/.test(f.detail)),
    JSON.stringify(failures)
  );
});

// ---- R3-4a: universal relative-clause IMPLIES -------------------------------
function impliesBuild(link, a = 'e0', b = 'e1') {
  const X = [
    'Every card is either Alpha or Beta.',
    'Alpha accounts for 1/2 of cards and Beta accounts for 1/2.',
    'A star appears on 4/5 of Alpha cards and 1/5 of Beta cards.',
    'A notch appears on 9/10 of Alpha cards and 2/5 of Beta cards.',
    link,
    'This card was observed with a star.',
    'What is the probability this card has a notch?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `EVENT e1 ${at(X, 'notch')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of cards')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    `LIK e1 POS GIVEN h0 9/10 ctx0 ${at(X, '9/10 of Alpha')}`,
    `LIK e1 POS GIVEN h1 2/5 ctx0 ${at(X, '2/5 of Beta')}`,
    `DET_LINK ${a} IMPLIES ${b} ctx0 ${at(X, link)}`,
    `OBS e0 POS ctx0 ${at(X, 'observed with a star')}`,
    'ASK_PROB e1 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}
const R3_LINK = 'Every card in this box that bears a star has a notch as well.';

test('R3 license: universal relative-clause IMPLIES is population-wide', () => {
  assert.ok(assertsUniversalRelation('IMPLIES', R3_LINK, 'star', 'notch', { popSing: 'card' }));
  assert.ok(assertsUniversalRelation(
    'IMPLIES', 'Every unit that has a star also has a notch.', 'star', 'notch', { popSing: 'unit' }
  ));
  const { X, bc } = impliesBuild(R3_LINK);
  const prog = toProgram(bc, X.length);
  assert.deepEqual(validateLicenseDecls(X, prog), []);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '1/1');
});

test('R3 license: relative-clause IMPLIES refuses scoped, negated, reversed forms', () => {
  const scoped = impliesBuild('Every Alpha card in this box that bears a star has a notch as well.');
  const scopedInvalid = validateLicenseDecls(scoped.X, toProgram(scoped.bc, scoped.X.length));
  assert.equal(scopedInvalid.length, 1, show(scopedInvalid));
  assert.match(scopedInvalid[0].reason, /HYPOTHESIS_SCOPED_LICENSE/);
  assert.equal(solve(scoped.bc, scoped.X).status, 'REFUSED');
  const negated = impliesBuild('Every card in this box that bears a star does not have a notch.');
  assert.equal(validateLicenseDecls(negated.X, toProgram(negated.bc, negated.X.length)).length, 1);
  const reversed = impliesBuild(R3_LINK, 'e1', 'e0');
  assert.equal(validateLicenseDecls(reversed.X, toProgram(reversed.bc, reversed.X.length)).length, 1);
});

// ---- R3-4b: equal-rate COND_INDEP 'separately' lead scope -------------------
// The population is named with two synonym heads ("card" in the partition,
// "panels" in the priors), so no single exact population noun exists; the
// per-item 'separately' scope over one passage-grounded bare noun must still
// license. Arithmetic matches buildA (25/33).
function separatelyBuild(indep) {
  const coordA = 'Among Alpha panels, one in four carry a notch and one in four a stripe.';
  const coordB = 'On Beta panels a notch turns up on one in ten and a stripe on three in ten.';
  const X = [
    'Every card is either Alpha or Beta.',
    'Alpha accounts for 3/5 of panels and Beta accounts for 2/5.',
    coordA, coordB, indep,
    'This panel was observed with its notch and stripe present.',
    'What is the probability the panel is Alpha?',
  ].join(' ');
  const aS = X.indexOf(coordA);
  const bS = X.indexOf(coordB);
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'notch', aS)}`,
    `EVENT e1 ${at(X, 'stripe', aS)}`,
    `PRIOR h0 3/5 ctx0 ${at(X, '3/5 of panels')}`,
    `PRIOR h1 2/5 ctx0 ${at(X, '2/5')}`,
    `LIK e0 POS GIVEN h0 1/4 ctx0 ${at(X, 'one in four carry a notch', aS)}`,
    `LIK e1 POS GIVEN h0 1/4 ctx0 ${at(X, 'one in four a stripe', aS)}`,
    `LIK e0 POS GIVEN h1 1/10 ctx0 ${at(X, 'a notch turns up on one in ten', bS)}`,
    `LIK e1 POS GIVEN h1 3/10 ctx0 ${at(X, 'a stripe on three in ten', bS)}`,
    `COND_INDEP (e0 e1) GIVEN * ctx0 ${at(X, indep)}`,
    `OBS e0 POS ctx0 ${at(X, 'This panel was observed with its notch and stripe present.')}`,
    `OBS e1 POS ctx0 ${at(X, 'This panel was observed with its notch and stripe present.')}`,
    'ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test("R3 license: 'On/Within/For each <pop> separately' contrasting arms validate", () => {
  for (const lead of ['On each card separately', 'Within each card separately', 'For each card separately']) {
    const indep =
      `${lead}, the stripe rate is the same for cards with a deep notch as for cards with a shallow notch.`;
    assert.ok(assertsIndependence(indep), indep);
    const { X, bc } = separatelyBuild(indep);
    assert.deepEqual(validateLicenseDecls(X, toProgram(bc, X.length)), [], indep);
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${indep} -> ${show(d)}`);
    assert.equal(fmt(d.value), '25/33');
  }
});

test("R3 license: 'separately' scope refuses hyp-scoped, identical, negated arms", () => {
  const cases = [
    'On Alpha only, the stripe rate is the same for cards with a deep notch as for cards with a shallow notch.',
    'On each card separately, the stripe rate is the same for cards with a deep notch as for cards with a deep notch.',
    'On each card separately, the stripe rate is not the same for cards with a deep notch as for cards with a shallow notch.',
    'On each card separately, the stripe rate differs for cards with a deep notch and cards with a shallow notch.',
  ];
  for (const indep of cases) {
    const { X, bc } = separatelyBuild(indep);
    assert.equal(validateLicenseDecls(X, toProgram(bc, X.length)).length, 1, indep);
    assert.equal(solve(bc, X).status, 'REFUSED', indep);
  }
  assert.ok(!assertsIndependence(cases[1]));
  assert.ok(!assertsIndependence(cases[2]));
});

// ---- R3-5: widened origin-preposition window --------------------------------
test('R3 ownership: six intervening words still bind the origin owner', () => {
  const shared = 'Half the cloth units in a bin came off the Alpha line and the other half came from Beta.';
  const rate = 'The lamp lights for 4/5 of Alpha units and 1/5 of Beta units.';
  const priors = (h0, h1) => [
    { sent: shared, needle: 'Half the cloth units in a bin came off the Alpha line', value: '1/2', hyp: h0 },
    { sent: shared, needle: 'the other half came from Beta', value: '1/2', hyp: h1 },
  ];
  const liks = [
    { sent: rate, needle: '4/5 of Alpha', value: '4/5' },
    { sent: rate, needle: '1/5 of Beta', value: '1/5' },
  ];
  const { X, bc } = buildUnit({ sents: [shared, rate], priors: priors(), liks });
  const prog = toProgram(bc, X.length);
  assert.deepEqual(ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
  const swapped = buildUnit({ sents: [shared, rate], priors: priors('h1', 'h0'), liks });
  const sprog = toProgram(swapped.bc, X.length);
  const failures = ownershipCheck({ X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by/.test(f.detail)),
    JSON.stringify(failures)
  );
});

// ---- R3-6: introductory scene nouns are not strata ---------------------------
test('R3 ctx: a framing office is no population; genuine strata still label', () => {
  const excl = new Set(['alpha', 'beta']);
  assert.deepEqual(
    [...ctxLabels('A shipping office has a box of cards, half printed on the Alpha press.', excl)],
    []
  );
  assert.deepEqual([...ctxLabels('A repair shop tossed its old cards.', excl)], []);
  assert.ok(ctxLabels('A day shift panel was observed.', excl).has('day shift'));
  assert.ok(ctxLabels('On the night shift, the tallies were kept.', excl).has('night shift'));
  assert.ok(ctxLabels('The North panels were counted.', excl).has('north'));
});

test('R3 extractor: intro scene stays one population; a North stratum refuses', () => {
  const mk = (likSent) => {
    const X = [
      'A shipping office has a box of cards, half printed on the Alpha press and half printed on the Beta press.',
      likSent,
      'This card was observed with a lamp.',
      'What is the probability it came from Alpha?',
    ].join(' ');
    const j = {
      hyps: [
        { id: 'h0', name: 'Alpha', span: 'Alpha press' },
        { id: 'h1', name: 'Beta', span: 'Beta press' },
      ],
      events: [{ id: 'e0', name: 'lamp', span: 'lamp lights' }],
      partition: ['h0', 'h1'],
      priors: [
        { hyp: 'h0', value: '1/2', span: 'A shipping office has a box of cards, half printed on the Alpha press' },
        { hyp: 'h1', value: '1/2', span: 'half printed on the Beta press' },
      ],
      liks: [
        { ev: 'e0', pol: 'POS', hyp: 'h0', value: '4/5', span: '4/5 of Alpha cards' },
        { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/5', span: '1/5 of Beta cards' },
      ],
      posteriors: [],
      obs: [{ ev: 'e0', pol: 'POS', span: 'observed with a lamp' }],
      nonAsserted: [], condIndep: [], detLinks: [],
      query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
    };
    return { X, ...jsonToProgram(j, X) };
  };
  const one = mk('The lamp lights for 4/5 of Alpha cards and 1/5 of Beta cards.');
  assert.deepEqual(one.errors, []);
  for (const d of one.program.decls) {
    if (d.ctx) assert.ok(!/shipping/.test(d.ctx), `${d.op} ${d.ctx}`);
  }
  const d1 = solve(one.program, one.X);
  assert.equal(d1.status, 'ANSWER', show(d1));
  assert.equal(fmt(d1.value), '4/5');
  const two = mk('In the North, the lamp lights for 4/5 of Alpha cards and 1/5 of Beta cards.');
  assert.deepEqual(two.errors, []);
  const d2 = solve(two.program, two.X);
  assert.equal(d2.status, 'REFUSED', show(d2));
  assert.equal(d2.code, 'CONTEXT_MISMATCH');
});

// ===========================================================================
// R4 (v4) — five general polish fixes. Generic synthetic fixtures only.
// ===========================================================================

// ---- R4-1: neutral random-sampling OBS subject prefix -----------------------
test('R4 sampling prefix: neutral drawn/random prefixes are inert shards', () => {
  const names = ['Alpha', 'Beta', 'star', 'amber notch'];
  const opts = { sentencePrefix: true, observedNames: ['star', 'amber notch'] };
  for (const t of [
    'A unit drawn at random from the bin',
    'A unit picked at random from the box',
    'One card chosen at random',
    'A random unit from the bin',
    'A random card',
  ]) {
    assert.ok(isInertUncoveredInterval(t, names, opts), t);
  }
  // the sampling frame is a sentence-prefix shard only, never a tail
  assert.ok(!isInertUncoveredInterval(
    'A unit drawn at random from the bin', names, { sentencePrefix: false }
  ));
});

test('R4 sampling prefix: restricted sampling prefixes stay material', () => {
  const names = ['Alpha', 'Beta', 'star', 'amber notch'];
  const opts = { sentencePrefix: true, observedNames: ['star', 'amber notch'] };
  for (const t of [
    'An Alpha unit drawn at random from the bin',
    'A damaged unit drawn at random from the bin',
    'A unit from the night shift drawn at random',
    'A star drawn at random from the bin',
  ]) {
    assert.ok(!isInertUncoveredInterval(t, names, opts), t);
  }
});

function randomObsBuild(obsSent) {
  const indep =
    'On each unit separately, the amber notch rate is the same for units with a star as for units without a star.';
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
    'A star appears on 4/5 of Alpha units and 1/5 of Beta units.',
    'An amber notch appears on 1/2 of Alpha units and 1/4 of Beta units.',
    indep, obsSent,
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const oS = X.indexOf(obsSent);
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `EVENT e1 ${at(X, 'amber notch')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of units')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    `LIK e1 POS GIVEN h0 1/2 ctx0 ${at(X, '1/2 of Alpha')}`,
    `LIK e1 POS GIVEN h1 1/4 ctx0 ${at(X, '1/4 of Beta')}`,
    `COND_INDEP (e0 e1) GIVEN * ctx0 ${at(X, indep)}`,
    `OBS e0 POS ctx0 ${at(X, 'has both a star', oS)}`,
    `OBS e1 POS ctx0 ${at(X, 'and an amber notch', oS)}`,
    'ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test('R4 sampling prefix: narrow OBS spans answer past a neutral subject', () => {
  for (const obs of [
    'A unit drawn at random from the bin has both a star and an amber notch.',
    'A random unit from the bin has both a star and an amber notch.',
  ]) {
    const { X, bc } = randomObsBuild(obs);
    const prog = toProgram(bc, X.length);
    assert.deepEqual(
      ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), [], obs
    );
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${obs} -> ${show(d)}`);
    assert.equal(fmt(d.value), '8/9', obs);
  }
});

test('R4 sampling prefix: restricted subjects refuse UNACCOUNTED_MATERIAL_SENTENCE', () => {
  for (const obs of [
    'An Alpha unit drawn at random from the bin has both a star and an amber notch.',
    'A damaged unit drawn at random from the bin has both a star and an amber notch.',
    'A unit from the night shift drawn at random has both a star and an amber notch.',
  ]) {
    const { X, bc } = randomObsBuild(obs);
    const d = solve(bc, X);
    assert.equal(d.status, 'REFUSED', obs);
    assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE', `${obs} -> ${d.code}`);
  }
});

// ---- R4-2: universal relative-clause IMPLIES verbs + population grounding ---
test('R4 license: every listed present verb licenses the relative core', () => {
  const scope = { popSing: 'card' };
  for (const s of [
    'Every card in this box that bears a star has a notch as well.',
    'Every card in this box that has a star has a notch as well.',
    'Every card that carries a star shows a notch.',
    'Every card that displays a star exhibits a notch.',
    'Every card that contains a star has a notch.',
    'All cards in this box that bear a star have a notch.',
    'All the cards that have a star have a notch as well.',
  ]) {
    assert.ok(assertsUniversalRelation('IMPLIES', s, 'star', 'notch', scope), s);
  }
  // a compound population phrase grounds only when the passage
  // established that exact phrase; the bare head alone never does
  const fare = 'Every fare card in this box that bears a star has a notch as well.';
  assert.ok(!assertsUniversalRelation('IMPLIES', fare, 'star', 'notch', scope));
  assert.ok(assertsUniversalRelation('IMPLIES', fare, 'star', 'notch', {
    popSing: 'card',
    popPhrases: new Set(['fare card']),
    hypLc: new Set(['alpha', 'beta']),
    evLc: new Set(['star', 'notch']),
  }));
});

function fareImpliesBuild(link) {
  const X = [
    'Every fare card is either Alpha or Beta.',
    'Alpha accounts for 1/2 of fare cards and Beta accounts for 1/2.',
    'A star appears on 4/5 of Alpha fare cards and 1/5 of Beta fare cards.',
    'A notch appears on 9/10 of Alpha fare cards and 2/5 of Beta fare cards.',
    link,
    'This fare card was observed with a star.',
    'What is the probability this fare card has a notch?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `EVENT e1 ${at(X, 'notch')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of fare cards')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    `LIK e1 POS GIVEN h0 9/10 ctx0 ${at(X, '9/10 of Alpha')}`,
    `LIK e1 POS GIVEN h1 2/5 ctx0 ${at(X, '2/5 of Beta')}`,
    `DET_LINK e0 IMPLIES e1 ctx0 ${at(X, link)}`,
    `OBS e0 POS ctx0 ${at(X, 'observed with a star')}`,
    'ASK_PROB e1 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test('R4 license: fare-card compound and All-plural heads validate and answer', () => {
  for (const link of [
    'Every fare card in this box that bears a star has a notch as well.',
    'Every card in this box that bears a star has a notch as well.',
    'All fare cards in this box that bear a star have a notch.',
  ]) {
    const { X, bc } = fareImpliesBuild(link);
    assert.deepEqual(validateLicenseDecls(X, toProgram(bc, X.length)), [], link);
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${link} -> ${show(d)}`);
    assert.equal(fmt(d.value), '1/1', link);
  }
});

test('R4 license: hyp-scoped and foreign relative heads still refuse', () => {
  const scoped = fareImpliesBuild(
    'Every Alpha fare card in this box that bears a star has a notch as well.'
  );
  const scopedInvalid = validateLicenseDecls(scoped.X, toProgram(scoped.bc, scoped.X.length));
  assert.equal(scopedInvalid.length, 1, show(scopedInvalid));
  assert.match(scopedInvalid[0].reason, /HYPOTHESIS_SCOPED_LICENSE/);
  assert.equal(solve(scoped.bc, scoped.X).status, 'REFUSED');
  const foreign = fareImpliesBuild(
    'Every valve in this box that bears a star has a notch as well.'
  );
  assert.equal(
    validateLicenseDecls(foreign.X, toProgram(foreign.bc, foreign.X.length)).length, 1
  );
  assert.equal(solve(foreign.bc, foreign.X).status, 'REFUSED');
  const allScoped = impliesBuild('All Alpha cards that bear a star have a notch.');
  assert.equal(
    validateLicenseDecls(allScoped.X, toProgram(allScoped.bc, allScoped.X.length)).length, 1
  );
});

// ---- R4-3: scoped unqueried unchecked-attribute witness handling ------------
test('R4 entailment: proven unqueried unchecked remark skips the FLUFF check', async (t) => {
  let prompt = null;
  t.mock.method(globalThis, 'fetch', async (url, opts) => {
    prompt = JSON.parse(opts.body).messages[0].content;
    const ids = [...new Set(
      [...prompt.matchAll(/^(L\d+|M1|F\d+|O\d+|C\d+|R\d+) \[/gm)].map((m) => m[1])
    )];
    return Response.json({ message: { content: JSON.stringify({
      entailed: true, code: null, reason: 'ok',
      checks: ids.map((id) => ({ id, entailed: true, reason: 'ok' })),
    }) } });
  });
  const { X, bc } = buildB({
    extra: ["The secondary dial hasn't been checked.", 'The cabinet stands beside a stool.'],
  });
  const prog = toProgram(bc, X.length);
  const tags = new Map(
    prog.decls.filter((d) => d.op === 'SENT_TAG').map((d) => [d.idx, d.tag])
  );
  const dialIdx = sentences(X).find((s) => s.text.includes('secondary dial')).idx;
  const cabIdx = sentences(X).find((s) => s.text.includes('cabinet stands')).idx;
  assert.equal(tags.get(dialIdx), 'FLUFF');
  assert.equal(tags.get(cabIdx), 'FLUFF');
  const w = await checkSemanticEntailment(prog, X);
  assert.equal(w.entailed, true, show(w));
  assert.ok(!new RegExp(`\\bF${dialIdx} \\[`).test(prompt), 'dial F-check excluded');
  assert.ok(new RegExp(`\\bF${cabIdx} \\[`).test(prompt), 'cabinet F-check retained');
});

test('R4 entailment: queried unchecked and FLUFF restrictions still fail', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => {
    throw new Error('unexpected witness request');
  });
  const queried = buildB({ extra: ["Its clipped corner hasn't been checked."] });
  const wq = await checkSemanticEntailment(toProgram(queried.bc, queried.X.length), queried.X);
  assert.equal(wq.entailed, false);
  assert.equal(wq.code, 'SCOPED_OR_RETRACTED_PREMISE');
  assert.equal(fetch.mock.callCount(), 0);
  fetch.mock.restore();
  // a normal FLUFF sentence the witness judges a restriction still
  // fails the strict AND with FLUFF_RESTRICTION
  t.mock.method(globalThis, 'fetch', async (url, opts) => {
    const p = JSON.parse(opts.body).messages[0].content;
    const ids = [...new Set(
      [...p.matchAll(/^(L\d+|M1|F\d+|O\d+|C\d+|R\d+) \[/gm)].map((m) => m[1])
    )];
    return Response.json({ message: { content: JSON.stringify({
      entailed: false, code: 'FLUFF_RESTRICTION', reason: 'restricts evidence',
      checks: ids.map((id) => ({
        id, entailed: !id.startsWith('F'), reason: id.startsWith('F') ? 'restricts' : 'ok',
      })),
    }) } });
  });
  const plain = buildB({ extra: ['The cabinet stands beside a stool.'] });
  const wp = await checkSemanticEntailment(toProgram(plain.bc, plain.X.length), plain.X);
  assert.equal(wp.entailed, false);
  assert.equal(wp.code, 'FLUFF_RESTRICTION');
});

// ---- R4-4: passive-agent posterior owner + multi-posterior lookup -----------
const R4_POST_X = [
  'Every unit is either Alpha or Beta.',
  'Of the units with a star, seven elevenths were repaired by Alpha and four elevenths by Beta.',
  'A unit with a star was observed.',
  'What is the probability it came from Alpha?',
].join(' ');
function r4PostBc(hA = 'h0', hB = 'h1') {
  const X = R4_POST_X;
  return withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `POSTERIOR_STATED ${hA} GIVEN e0 POS 7/11 ctx0 ${at(X, 'seven elevenths were repaired by Alpha')}`,
    `POSTERIOR_STATED ${hB} GIVEN e0 POS 4/11 ctx0 ${at(X, 'four elevenths by Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a star was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
}

test('R4 posterior: passive-agent owners ground both stated shares', () => {
  const bc = r4PostBc();
  const prog = toProgram(bc, R4_POST_X.length);
  assert.deepEqual(
    ownershipCheck({ X: R4_POST_X, decls: prog.decls, ...nameMaps(R4_POST_X, prog), prog }), []
  );
  const d = solve(bc, R4_POST_X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '7/11');
  assert.equal(d.via, 'POSTERIOR_STATED');
});

test('R4 posterior: wrong-owner passive bindings refuse UNSTABLE_BINDING', () => {
  const prog = toProgram(r4PostBc('h1', 'h0'), R4_POST_X.length);
  const failures = ownershipCheck({
    X: R4_POST_X, decls: prog.decls, ...nameMaps(R4_POST_X, prog), prog,
  });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by "Alpha"/.test(f.detail)),
    JSON.stringify(failures)
  );
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by "Beta"/.test(f.detail)),
    JSON.stringify(failures)
  );
});

test('R4 posterior: a PRIOR in the passive-agent subgroup share still refuses', () => {
  const X = R4_POST_X;
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `PRIOR h0 7/11 ctx0 ${at(X, 'seven elevenths were repaired by Alpha')}`,
    `PRIOR h1 4/11 ctx0 ${at(X, 'four elevenths by Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a star was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const prog = toProgram(bc, X.length);
  const failures = ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /subgroup share/.test(f.detail)),
    JSON.stringify(failures)
  );
});

test('R4 posterior: multi-posterior group consistency gates the direct answer', () => {
  const base = `HYP h0 [0,10]\nHYP h1 [10,20]\nPART_EXHAUSTIVE (h0 h1) [0,20]\n` +
    `EVENT e0 [30,40]\nPOSTERIOR_STATED h0 GIVEN e0 POS 7/11 ctx0 [30,40]\n` +
    `POSTERIOR_STATED h1 GIVEN e0 POS 4/11 ctx0 [40,50]\nOBS e0 POS ctx0 [50,60]\n` +
    `ASK_PROB h0 GIVEN (e0:POS) ctx0\nEND`;
  const d = solve(base, null);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '7/11');
  // over-sum across distinct hyps contradicts
  assert.equal(solve(base.replace('4/11', '5/11'), null).code, 'CONFLICTING_FIGURES');
  // every partition member stated but the shares do not exhaust 1/1
  assert.equal(solve(base.replace('4/11', '3/11'), null).code, 'CONFLICTING_FIGURES');
  // duplicate conflicting values for one hyp (wildcard ctx twin)
  const dup = base.replace(
    'OBS e0',
    'POSTERIOR_STATED h0 GIVEN e0 POS 6/11 * [60,70]\nOBS e0'
  );
  assert.equal(solve(dup, null).code, 'CONFLICTING_FIGURES', show(solve(dup, null)));
  // no stated posterior for the target: the reversed refusal stands
  const onlyOther = base.replace('POSTERIOR_STATED h0 GIVEN e0 POS 7/11 ctx0 [30,40]\n', '');
  assert.equal(solve(onlyOther, null).code, 'REVERSED_CONDITIONAL');
});

// ---- R4-5: pure structural inventory-overview sentence -----------------------
test('R4 inventory: name-free overview frames are FLUFF', () => {
  const names = ['Alpha', 'Beta', 'lamp', 'star'];
  for (const txt of [
    'The depot keeps separate inventories of the valves on its upper and lower bays.',
    'The depot maintains separate records of the valves.',
    'The depot keeps separate logs of the panels by bay.',
    'The depot maintains separate tallies of the units.',
  ]) {
    assert.equal(sentenceMateriality(sentences(txt)[0], names).kind, 'fluff', txt);
  }
});

test('R4 inventory: named, numbered, mechanism, or retracted frames stay material', () => {
  const names = ['Alpha', 'Beta', 'lamp', 'star'];
  for (const txt of [
    'The depot keeps separate inventories of the lamp.',
    'The depot keeps separate inventories of the Alpha valves.',
    'The depot keeps separate inventories of the valves at a 40% share.',
    'The depot keeps separate inventories of the relays.',
    'The depot keeps separate inventories and the figures were retracted.',
    'The depot keeps separate inventories, and stores the valves by bay.',
  ]) {
    assert.equal(sentenceMateriality(sentences(txt)[0], names).kind, 'material', txt);
  }
});

test('R4 inventory: overview passage answers; real unaccounted premises refuse', () => {
  const ok = buildB({
    extra: ['The depot keeps separate inventories of the valves on its upper and lower bays.'],
  });
  const okProg = toProgram(ok.bc, ok.X.length);
  const invIdx = sentences(ok.X).find((s) => s.text.includes('separate inventories')).idx;
  assert.equal(
    okProg.decls.find((d) => d.op === 'SENT_TAG' && d.idx === invIdx).tag, 'FLUFF'
  );
  const d = solve(ok.bc, ok.X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '8/11');
  for (const extra of [
    ['The depot keeps separate inventories of the clipped corner.'],
    ['The clipped corner appeared on this panel yesterday.'],
  ]) {
    const bad = buildB({ extra });
    const bd = solve(bad.bc, bad.X);
    assert.equal(bd.status, 'REFUSED', extra[0]);
    assert.equal(bd.code, 'UNACCOUNTED_MATERIAL_SENTENCE', `${extra[0]} -> ${bd.code}`);
  }
});

// ===========================================================================
// R5 (v5) — four general structural fixes. Generic synthetic fixtures only
// (Alpha/Beta, star, cracked seal, stamped lid, upper/lower deck, clerk).
// ===========================================================================

// ---- R5-1: participial event names + leading-conjunction OBS anchoring -----
function r5F1Build(obs, obsSpans) {
  const coordA = 'Among Alpha units, one in four carry a star and one in four a cracked seal.';
  const coordB = 'On Beta units a star turns up on one in ten and a cracked seal on three in ten.';
  const indep = 'On each unit separately, the cracked seal rate is the same for units with a star as for units without a star.';
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 3/5 of units and Beta accounts for 2/5.',
    coordA, coordB, indep, obs,
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const aS = X.indexOf(coordA);
  const bS = X.indexOf(coordB);
  const oS = X.indexOf(obs);
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star', aS)}`,
    `EVENT e1 ${at(X, 'cracked seal', aS)}`,
    `PRIOR h0 3/5 ctx0 ${at(X, '3/5 of units')}`,
    `PRIOR h1 2/5 ctx0 ${at(X, '2/5')}`,
    `LIK e0 POS GIVEN h0 1/4 ctx0 ${at(X, 'one in four carry a star', aS)}`,
    `LIK e1 POS GIVEN h0 1/4 ctx0 ${at(X, 'one in four a cracked seal', aS)}`,
    `LIK e0 POS GIVEN h1 1/10 ctx0 ${at(X, 'a star turns up on one in ten', bS)}`,
    `LIK e1 POS GIVEN h1 3/10 ctx0 ${at(X, 'a cracked seal on three in ten', bS)}`,
    `COND_INDEP (e0 e1) GIVEN * ctx0 ${at(X, indep)}`,
    `OBS e0 POS ctx0 ${at(X, obsSpans[0], oS)}`,
    `OBS e1 POS ctx0 ${at(X, obsSpans[1], oS)}`,
    'ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test('R5 participial: declared-name participles and gerunds never split a coordinated OBS', () => {
  const names = ['Alpha', 'Beta', 'star', 'cracked seal', 'flashing beacon', 'buzzer'];
  for (const txt of [
    'A random unit from this tray has both a star and a cracked seal.',
    'A unit has a star and a flashing beacon.',
    'A unit has a star and a stamped lid.',
  ]) {
    assert.equal(subClauses(sentences(txt)[0], names).length, 1, txt);
  }
  // an undeclared prenominal participle is a modifier, not a verb either
  assert.equal(
    subClauses(sentences('A unit has a star and a sealed box.')[0], names).length, 1
  );
});

test('R5 participial: real sister clauses still split', () => {
  const names = ['Alpha', 'Beta', 'star', 'cracked seal', 'flashing beacon', 'buzzer'];
  for (const txt of [
    'A unit had a star and the seal cracked.',
    'A unit had a star and the buzzer fired alone.',
    'A unit had a star, its buzzer silent.',
    'A unit had a star despite a cracked seal remaining shut.',
  ]) {
    assert.ok(subClauses(sentences(txt)[0], names).length >= 2, txt);
  }
});

test('R5 participial: coordinated cracked-seal OBS with leading-and span is clean', () => {
  const { X, bc } = r5F1Build(
    'A random unit from this tray has both a star and a cracked seal.',
    ['has both a star', 'and a cracked seal']
  );
  const prog = toProgram(bc, X.length);
  assert.deepEqual(adjacentClauseFindings(X, prog), []);
  assert.equal(checkSentenceAccounting(X, prog).ok, true);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
});

test('R5 anchoring: comma-coordinated OBS piece anchors through its leading conjunction', () => {
  // the strong comma coordinator DOES split this sentence in two; the
  // second OBS quote starts at "and", just before the split piece —
  // only the leading-coordinator anchoring keeps the piece grounded
  const { X, bc } = r5F1Build(
    'A unit has a star, and a cracked seal present.',
    ['has a star', 'and a cracked seal present']
  );
  const prog = toProgram(bc, X.length);
  const seg = sentences(X).find((s) => s.text.includes('cracked seal present'));
  assert.equal(
    subClauses(seg, ['Alpha', 'Beta', 'star', 'cracked seal']).length, 2
  );
  assert.deepEqual(adjacentClauseFindings(X, prog), []);
  assert.equal(checkSentenceAccounting(X, prog).ok, true);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '25/33');
});

function r5F1Refusal(obs) {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta for 1/2.',
    'The star appears for 1/2 of Alpha units and 1/4 of Beta units.',
    obs,
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of units')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta for 1/2')}`,
    `LIK e0 POS GIVEN h0 1/2 ctx0 ${at(X, '1/2 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/4 ctx0 ${at(X, '1/4 of Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'had a star')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test('R5 participial: unanchored counterexamples still produce findings and refuse', () => {
  for (const obs of [
    'A unit had a star and the seal cracked.',
    'A unit had a star and the lamp fired alone.',
    'A unit had a star, its buzzer silent.',
    'A unit had a star despite a cracked seal remaining shut.',
  ]) {
    const { X, bc } = r5F1Refusal(obs);
    const prog = toProgram(bc, X.length);
    assert.ok(adjacentClauseFindings(X, prog).length > 0, obs);
    const d = solve(bc, X);
    assert.equal(d.status, 'REFUSED', obs);
    assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE', `${obs} -> ${d.code}`);
  }
  const ok = r5F1Refusal('A unit had a star.');
  const okD = solve(ok.bc, ok.X);
  assert.equal(okD.status, 'ANSWER', show(okD));
  assert.equal(fmt(okD.value), '2/3');
});

// ---- R5-2: duplicate OBS span disambiguation --------------------------------
const R5_DUP_X = [
  'Every unit is either Alpha or Beta.',
  'Alpha accounts for 1/2 of units and Beta for 1/2.',
  'On Alpha units a star appears on one half and a stamped lid on one third.',
  'On Beta units a star appears on one quarter and a stamped lid on one half.',
  'The star rate is the same for units with a stamped lid as for units without a stamped lid.',
  'A random unit from the bin has both a star and a stamped lid.',
  'What is the probability the unit is Alpha?',
].join(' ');
function r5DupJson(obs) {
  return {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Alpha' },
      { id: 'h1', name: 'Beta', span: 'Beta' },
    ],
    events: [
      { id: 'e0', name: 'star', span: 'star' },
      { id: 'e1', name: 'stamped lid', span: 'stamped lid' },
    ],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'Beta for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: 'a star appears on one half' },
      { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/3', span: 'a stamped lid on one third' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: 'a star appears on one quarter' },
      { ev: 'e1', pol: 'POS', hyp: 'h1', value: '1/2', span: 'a stamped lid on one half' },
    ],
    posteriors: [],
    obs,
    nonAsserted: [],
    condIndep: [{
      evs: ['e0', 'e1'],
      span: 'The star rate is the same for units with a stamped lid as for units without a stamped lid',
    }],
    detLinks: [],
    partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, { ev: 'e1', pol: 'POS' }] },
  };
}

test('R5 duplicate OBS: verbatim rate-sentence copy never captures the observation', () => {
  const X = R5_DUP_X;
  // the fixture genuinely carries the quote in two rate sentences first
  assert.ok(resolveSpanAll(X, 'and a stamped lid').length >= 3);
  const { program, errors } = jsonToProgram(
    r5DupJson([
      { ev: 'e0', pol: 'POS', span: 'has both a star' },
      { ev: 'e1', pol: 'POS', span: 'and a stamped lid' },
    ]), X
  );
  assert.deepEqual(errors, []);
  const obsSeg = sentences(X).find((s) => s.text.includes('random unit from the bin'));
  for (const d of program.decls.filter((x) => x.op === 'OBS')) {
    assert.ok(d.span.s >= obsSeg.s && d.span.e <= obsSeg.e,
      `${d.ev} bound outside the observation sentence: ${show(d.span)}`);
  }
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/7');
});

test('R5 duplicate OBS: unresolvable observation quote still fails closed', () => {
  const { errors } = jsonToProgram(
    r5DupJson([
      { ev: 'e0', pol: 'POS', span: 'has both a star' },
      { ev: 'e1', pol: 'POS', span: 'and a polished lid' },
    ]), R5_DUP_X
  );
  assert.ok(errors.some((e) => /unresolvable span for OBS e1/.test(e)), show(errors));
});

// ---- R5-3a: inspection-method background remarks -----------------------------
test('R5 inspection: pure method remarks are FLUFF', () => {
  const names = ['Alpha', 'Beta', 'star', 'stamped lid'];
  for (const txt of [
    'You can tell which coating they applied by inspecting the seam.',
    'One can tell which dye was used by checking the lining.',
    'You can tell what dye was used by checking the lining.',
    'The clerk can identify which finish was used by feeling the lining.',
  ]) {
    assert.equal(isInspectionMethodSentence(txt, names), true, txt);
    assert.equal(sentenceMateriality(sentences(txt)[0], names).kind, 'fluff', txt);
  }
});

test('R5 inspection: named, numbered, doubted, or negated frames stay material', () => {
  const names = ['Alpha', 'Beta', 'star', 'stamped lid'];
  for (const txt of [
    'You can tell which star they applied by inspecting the seam.',
    'You can tell which coating they applied at a 40% share by inspecting the seam.',
    'You can tell which coating they applied by inspecting the seam nobody checked.',
    'You cannot tell which coating they applied by inspecting the seam.',
    'You can tell which coating they applied, by inspecting the seam.',
  ]) {
    assert.equal(isInspectionMethodSentence(txt, names), false, txt);
    assert.equal(sentenceMateriality(sentences(txt)[0], names).kind, 'material', txt);
  }
});

test('R5 inspection: method remark in a full passage answers as FLUFF', () => {
  const ok = buildB({
    extra: ['You can tell which coating they applied by inspecting the seam.'],
  });
  const prog = toProgram(ok.bc, ok.X.length);
  const idx = sentences(ok.X).find((s) => s.text.includes('inspecting the seam')).idx;
  assert.equal(prog.decls.find((d) => d.op === 'SENT_TAG' && d.idx === idx).tag, 'FLUFF');
  const d = solve(ok.bc, ok.X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '8/11');
});

// ---- R5-3b: zero-prior unrecorded-overall-tally accounting -------------------
const R5_TALLY = 'The clerk kept no overall tally of how many items each workshop handled.';
function r5PostBuild(tally = R5_TALLY) {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Of the units with a star, seven elevenths were repaired by Alpha and four elevenths by Beta.',
    tally,
    'A unit with a star was observed.',
    'What is the probability it came from Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `POSTERIOR_STATED h0 GIVEN e0 POS 7/11 ctx0 ${at(X, 'seven elevenths were repaired by Alpha')}`,
    `POSTERIOR_STATED h1 GIVEN e0 POS 4/11 ctx0 ${at(X, 'four elevenths by Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a star was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test('R5 tally: zero-prior direct-posterior program excuses the unrecorded tally', () => {
  const { X, bc } = r5PostBuild();
  const prog = toProgram(bc, X.length);
  const seg = sentences(X).find((s) => s.text.includes('overall tally'));
  assert.equal(irrelevantUnrecordedPriorSeg(X, prog, seg), true);
  assert.equal(
    prog.decls.find((d) => d.op === 'SENT_TAG' && d.idx === seg.idx).tag, 'FLUFF'
  );
  assert.equal(checkSentenceAccounting(X, prog).ok, true);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '7/11');
  // the passive frame proves the same way
  const passive = r5PostBuild("No overall count of each supplier's output was recorded.");
  const pProg = toProgram(passive.bc, passive.X.length);
  const pSeg = sentences(passive.X).find((s) => s.text.includes('overall count'));
  assert.equal(irrelevantUnrecordedPriorSeg(passive.X, pProg, pSeg), true);
  assert.equal(solve(passive.bc, passive.X).status, 'ANSWER');
});

test('R5 tally: LIK-only zero-prior program still refuses MISSING_BASE_RATE', () => {
  const X = [
    'Every unit is either Alpha or Beta.',
    'The star appears for 1/2 of Alpha units and 1/4 of Beta units.',
    R5_TALLY,
    'A unit with a star was observed.',
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star')}`,
    `LIK e0 POS GIVEN h0 1/2 ctx0 ${at(X, '1/2 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/4 ctx0 ${at(X, '1/4 of Beta')}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a star was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const prog = toProgram(bc, X.length);
  const seg = sentences(X).find((s) => s.text.includes('overall tally'));
  assert.equal(irrelevantUnrecordedPriorSeg(X, prog, seg), true);
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED', show(d));
  assert.equal(d.code, 'MISSING_BASE_RATE');
});

test('R5 tally: any declared PRIOR keeps the unrecorded tally material', () => {
  const bad = buildB({ extra: [R5_TALLY] });
  const prog = toProgram(bad.bc, bad.X.length);
  const seg = sentences(bad.X).find((s) => s.text.includes('overall tally'));
  assert.equal(irrelevantUnrecordedPriorSeg(bad.X, prog, seg), false);
  const d = solve(bad.bc, bad.X);
  assert.equal(d.status, 'REFUSED', show(d));
  assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
});

// ---- R5-4: paired spatial strata + disjoint-stratum accounting ---------------
test('R5 strata: paired spatial phrases and possessives derive distinct ctx ids', () => {
  const excl = expandCtxExclude(['Alpha', 'Beta', 'star']);
  assert.equal(canonicalCtxId('On the upper deck every item is checked.', excl), 'ctx_upper_deck');
  assert.equal(canonicalCtxId("Half of the upper deck's panels are Alpha.", excl), 'ctx_upper_deck');
  assert.equal(canonicalCtxId('A panel from the lower deck was drawn.', excl), 'ctx_lower_deck');
  assert.equal(canonicalCtxId('On the upper decks the crew works.', excl), 'ctx_upper_deck');
  assert.equal(canonicalCtxId('At the north pier the bins stand.', excl), 'ctx_north_pier');
  assert.equal(canonicalCtxId('In the inner vault the seals rest.', excl), 'ctx_inner_vault');
  // possessive population nouns normalize the same way
  assert.equal(canonicalCtxId("Half of the night shift's panels are Alpha.", excl), 'ctx_night_shift');
  assert.equal(canonicalCtxId("On the upper ward's roster the clerk signs.", excl), 'ctx_upper_ward');
  // abstract/part heads and declared-name words never label a stratum
  assert.equal(canonicalCtxId('On the upper side of the panel a mark sits.', excl), null);
  assert.equal(canonicalCtxId('In the north case the rate holds.', excl), null);
  assert.equal(
    canonicalCtxId('At the north pier the bins stand.', expandCtxExclude(['north beacon'])),
    null
  );
});

function r5DeckBuild({ lowerSentence = null, deck = 'upper' } = {}) {
  const X = [
    'Every panel is either Alpha or Beta.',
    'On the upper deck, Alpha accounts for 1/2 of panels and Beta for 1/2.',
    'On the upper deck, a star appears on 1/2 of Alpha panels and 1/4 of Beta panels.',
    ...(lowerSentence ? [lowerSentence] : []),
    `A panel from the ${deck} deck was observed with a star.`,
    `What is the probability a panel from the ${deck} deck with a star is Alpha?`,
  ].join(' ');
  const qCtx = `ctx_${deck}_deck`;
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star', X.indexOf('a star appears'))}`,
    `PRIOR h0 1/2 ctx_upper_deck ${at(X, '1/2 of panels')}`,
    `PRIOR h1 1/2 ctx_upper_deck ${at(X, 'Beta for 1/2')}`,
    `LIK e0 POS GIVEN h0 1/2 ctx_upper_deck ${at(X, '1/2 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/4 ctx_upper_deck ${at(X, '1/4 of Beta')}`,
    `OBS e0 POS ${qCtx} ${at(X, `from the ${deck} deck was observed with a star`)}`,
    `ASK_PROB h0 GIVEN (e0:POS) ${qCtx}`,
    'END',
  ].join('\n'));
  return { X, bc };
}
const R5_LOWER_QUAL = 'On the lower deck every item, of either alloy, passes the load test.';

test('R5 strata: querying the upper deck excuses a qualitative lower-deck sentence', () => {
  const { X, bc } = r5DeckBuild({ lowerSentence: R5_LOWER_QUAL });
  const prog = toProgram(bc, X.length);
  const seg = sentences(X).find((s) => s.text.includes('lower deck'));
  assert.equal(irrelevantOtherStratumSeg(X, prog, seg), true);
  assert.equal(
    prog.decls.find((d) => d.op === 'SENT_TAG' && d.idx === seg.idx).tag, 'FLUFF'
  );
  assert.equal(checkSentenceAccounting(X, prog).ok, true);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '2/3');
});

test('R5 strata: querying the lower deck on upper-deck premises refuses cleanly', () => {
  const { X, bc } = r5DeckBuild({ deck: 'lower' });
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED', show(d));
  assert.equal(d.code, 'CONTEXT_MISMATCH');
});

test('R5 strata: queried-stratum mention, literal, or doubt keeps the other stratum material', () => {
  for (const variant of [
    'On the lower deck every item, unlike an upper-deck item, passes the load test.',
    'On the lower deck an item is not an upper-deck item and passes the load test.',
    'On the lower deck 1/2 of the items pass the load test.',
    'On the lower deck nobody has checked the load test results.',
  ]) {
    const { X, bc } = r5DeckBuild({ lowerSentence: variant });
    const prog = toProgram(bc, X.length);
    const seg = sentences(X).find((s) => s.text.includes('lower deck'));
    assert.equal(irrelevantOtherStratumSeg(X, prog, seg), false, variant);
    assert.equal(solve(bc, X).status, 'REFUSED', variant);
  }
});

test('R5 strata: extractor derives paired deck ctxs from the text itself', () => {
  const X = [
    'Every panel is either Alpha or Beta.',
    'On the upper deck, Alpha accounts for 1/2 of panels and Beta for 1/2.',
    'On the upper deck, a star appears on 1/2 of Alpha panels and 1/4 of Beta panels.',
    'A panel from the upper deck was observed with a star.',
    'What is the probability a panel from the upper deck with a star is Alpha?',
  ].join(' ');
  const j = {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Alpha' },
      { id: 'h1', name: 'Beta', span: 'Beta' },
    ],
    events: [{ id: 'e0', name: 'star', span: 'star' }],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'Beta for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: '1/2 of Alpha panels' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: '1/4 of Beta panels' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'from the upper deck was observed with a star' }],
    nonAsserted: [], condIndep: [], detLinks: [], partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  assert.equal(program.query.ctx, 'ctx_upper_deck');
  for (const d of program.decls) {
    if (d.ctx) assert.equal(d.ctx, 'ctx_upper_deck', `${d.op} ${d.ctx}`);
  }
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '2/3');
});

// ---- R5 entailment: scoped FLUFF checks stay witness-checked -----------------
function r5MockWitness(t, captured) {
  t.mock.method(globalThis, 'fetch', async (url, opts) => {
    const prompt = JSON.parse(opts.body).messages[0].content;
    captured.push(prompt);
    const ids = [...new Set(
      [...prompt.matchAll(/^(L\d+|M1|F\d+|O\d+|C\d+|R\d+) \[/gm)].map((m) => m[1])
    )];
    return Response.json({ message: { content: JSON.stringify({
      entailed: true, code: null, reason: 'ok',
      checks: ids.map((id) => ({ id, entailed: true, reason: 'ok' })),
    }) } });
  });
}

test('R5 entailment: unrecorded-count FLUFF gets a scoped zero-prior check', async (t) => {
  const captured = [];
  r5MockWitness(t, captured);
  const { X, bc } = r5PostBuild();
  const prog = toProgram(bc, X.length);
  const idx = sentences(X).find((s) => s.text.includes('overall tally')).idx;
  const w = await checkSemanticEntailment(prog, X);
  assert.equal(w.entailed, true, show(w));
  assert.equal(captured.length, 1);
  assert.ok(captured[0].includes(`F${idx} [INERT SENTENCE — UNRECORDED OVERALL COUNT]`), captured[0]);
  assert.ok(captured[0].includes('NO marginal priors'), captured[0]);
});

test('R5 entailment: other-stratum FLUFF gets a scoped stratum check', async (t) => {
  const captured = [];
  r5MockWitness(t, captured);
  const { X, bc } = r5DeckBuild({ lowerSentence: R5_LOWER_QUAL });
  const prog = toProgram(bc, X.length);
  const idx = sentences(X).find((s) => s.text.includes('lower deck')).idx;
  const w = await checkSemanticEntailment(prog, X);
  assert.equal(w.entailed, true, show(w));
  assert.equal(captured.length, 1);
  assert.ok(captured[0].includes(`F${idx} [INERT SENTENCE — OTHER STRATUM]`), captured[0]);
  assert.ok(captured[0].includes('"upper deck"'), captured[0]);
});

test('R5 entailment: inventory-overview FLUFF gets a scoped overview check', async (t) => {
  const captured = [];
  r5MockWitness(t, captured);
  const { X, bc } = buildB({
    extra: ['The depot keeps separate inventories of the valves on its upper and lower bays.'],
  });
  const prog = toProgram(bc, X.length);
  const idx = sentences(X).find((s) => s.text.includes('separate inventories')).idx;
  const w = await checkSemanticEntailment(prog, X);
  assert.equal(w.entailed, true, show(w));
  assert.equal(captured.length, 1);
  assert.ok(captured[0].includes(`F${idx} [INERT SENTENCE — INVENTORY OVERVIEW]`), captured[0]);
});

// ===========================================================================
// R6 (v6) — three general structural fixes. Generic synthetic fixtures only
// (Alpha/Beta, brass/zinc, star/cracked seal, upper/lower deck, clerk-free).
// ===========================================================================

// ---- R6-1: leading-coordinator anchoring for numeric factor decls ----------
function r6PostBuild(rate, spanH1, valH1 = '4/11') {
  const X = [
    'Every unit is either Alpha or Beta.',
    rate,
    'A unit with a blue mark was observed.',
    'What is the probability it came from Alpha?',
  ].join(' ');
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'blue mark')}`,
    `POSTERIOR_STATED h0 GIVEN e0 POS 7/11 ctx0 ${at(X, '7/11 were handled by Alpha')}`,
    `POSTERIOR_STATED h1 GIVEN e0 POS ${valH1} ctx0 ${at(X, spanH1)}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a blue mark was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  return { X, bc };
}

test('R6 coord rate: leading-and POSTERIOR piece anchors and solves', () => {
  for (const [rate, spanH1] of [
    // comma-coordinated elided arm (the brief's shape verbatim)
    ['Of the items with a blue mark, 7/11 were handled by Alpha, and 4/11 by Beta.',
     'and 4/11 by Beta'],
    // bare-coordinated full arm: the rate verb lives inside the span
    ['Of the items with a blue mark, 7/11 were handled by Alpha and 4/11 were handled by Beta.',
     'and 4/11 were handled by Beta'],
  ]) {
    const { X, bc } = r6PostBuild(rate, spanH1);
    const prog = toProgram(bc, X.length);
    const seg = sentences(X).find((s) => s.text.includes('7/11'));
    assert.ok(
      subClauses(seg, ['Alpha', 'Beta', 'blue mark']).length >= 2, rate
    );
    assert.deepEqual(adjacentClauseFindings(X, prog), [], rate);
    assert.equal(checkSentenceAccounting(X, prog).ok, true, rate);
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${rate} -> ${show(d)}`);
    assert.equal(fmt(d.value), '7/11', rate);
  }
});

test('R6 coord rate: second rate, wrong value/owner, doubt, co-occurrence stay residual', () => {
  // Each sister piece defeats one numeric anchoring guard, so it stays
  // an adjacent (witness-refusing) residual; doubt and a misbound
  // value additionally refuse in the kernel itself.
  const cases = [
    ['Of the items with a blue mark, 7/11 were handled by Alpha, and 4/11 by Beta plus 1/11 held back.',
     'and 4/11 by Beta plus 1/11 held back', '4/11'],
    ['Of the items with a blue mark, 7/11 were handled by Alpha and 4/11 were handled by Beta.',
     'and 4/11 were handled by Beta', '3/11'],
    ['Of the items with a blue mark, 7/11 were handled by Alpha and 4/11 were handled by Alpha.',
     'and 4/11 were handled by Alpha', '4/11'],
    ['Of the items with a blue mark, 7/11 were handled by Alpha, and 4/11 by Beta was retracted.',
     'and 4/11 by Beta was retracted', '4/11'],
    ['Of the items with a blue mark, 7/11 were handled by Alpha and 4/11 were handled by Beta together.',
     'and 4/11 were handled by Beta together', '4/11'],
    ['Of the items with a blue mark, 7/11 were handled by Alpha and 4/11 were handled by Beta despite the log.',
     'and 4/11 were handled by Beta despite the log', '4/11'],
  ];
  for (const [rate, spanH1, valH1] of cases) {
    const { X, bc } = r6PostBuild(rate, spanH1, valH1);
    const prog = toProgram(bc, X.length);
    assert.ok(adjacentClauseFindings(X, prog).length > 0, rate);
  }
  const doubt = r6PostBuild(
    'Of the items with a blue mark, 7/11 were handled by Alpha, and 4/11 by Beta was retracted.',
    'and 4/11 by Beta was retracted'
  );
  assert.equal(solve(doubt.bc, doubt.X).status, 'REFUSED');
  const wrongVal = r6PostBuild(
    'Of the items with a blue mark, 7/11 were handled by Alpha and 4/11 were handled by Beta.',
    'and 4/11 were handled by Beta', '3/11'
  );
  assert.equal(solve(wrongVal.bc, wrongVal.X).status, 'REFUSED');
});

// ---- R6-2: contradictory COND_INDEP duplicate of a DET_LINK -----------------
const R6_LINK_SENT =
  'Every unit in this tray that carries a star has a cracked seal as well.';
const R6_LINK_X = [
  'Every unit is either Alpha or Beta.',
  'Alpha accounts for 1/2 of units and Beta for 1/2.',
  'A star appears on 1/2 of Alpha units and 1/4 of Beta units.',
  R6_LINK_SENT,
  'A unit with a star was observed.',
  'What is the probability the unit is Alpha?',
].join(' ');
function r6LinkJson({ det = true, cond = true } = {}) {
  const span = R6_LINK_SENT.replace(/\.$/, '');
  return {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Alpha' },
      { id: 'h1', name: 'Beta', span: 'Beta' },
    ],
    events: [
      { id: 'e0', name: 'star', span: 'star' },
      { id: 'e1', name: 'cracked seal', span: 'cracked seal' },
    ],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'Beta for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: '1/2 of Alpha units' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: '1/4 of Beta units' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'A unit with a star was observed' }],
    nonAsserted: [],
    condIndep: cond ? [{ evs: ['e0', 'e1'], span }] : [],
    detLinks: det ? [{ a: 'e0', rel: 'IMPLIES', b: 'e1', span }] : [],
    partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
}

test('R6 dedup: duplicate COND_INDEP on the DET_LINK sentence compiles away', () => {
  const { program, errors } = jsonToProgram(r6LinkJson(), R6_LINK_X);
  assert.deepEqual(errors, []);
  assert.equal(program.decls.filter((d) => d.op === 'COND_INDEP').length, 0);
  assert.equal(program.decls.filter((d) => d.op === 'DET_LINK').length, 1);
  assert.deepEqual(validateLicenseDecls(R6_LINK_X, program), []);
  const d = solve(program, R6_LINK_X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '2/3');
});

test('R6 dedup: standalone COND_INDEP is kept and refuses UNLICENSED_INDEPENDENCE', async () => {
  const { program } = jsonToProgram(r6LinkJson({ det: false }), R6_LINK_X);
  assert.equal(program.decls.filter((d) => d.op === 'COND_INDEP').length, 1);
  assert.equal(assertsIndependence(R6_LINK_SENT), false);
  assert.ok(validateLicenseDecls(R6_LINK_X, program)
    .some((b) => /does not affirmatively assert independence/.test(b.reason)));
  const X = R6_LINK_X;
  const bc = withLedger(X, [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'star', X.indexOf('A star appears'))}`,
    `EVENT e1 ${at(X, 'cracked seal')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of units')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta for 1/2')}`,
    `LIK e0 POS GIVEN h0 1/2 ctx0 ${at(X, '1/2 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/4 ctx0 ${at(X, '1/4 of Beta')}`,
    `COND_INDEP (e0 e1) GIVEN * ctx0 ${at(X, R6_LINK_SENT)}`,
    `OBS e0 POS ctx0 ${at(X, 'A unit with a star was observed.')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ].join('\n'));
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X, extract: makeSemanticExtractor(bc),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNLICENSED_INDEPENDENCE');
});

test('R6 dedup: an independence-asserting sentence keeps its COND_INDEP', () => {
  const indepSent = 'The star and cracked seal are independent.';
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta for 1/2.',
    'A star appears on 1/2 of Alpha units and 1/4 of Beta units.',
    indepSent,
    'A unit with a star was observed.',
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const span = indepSent.replace(/\.$/, '');
  const j = r6LinkJson();
  j.condIndep = [{ evs: ['e0', 'e1'], span }];
  j.detLinks = [{ a: 'e0', rel: 'EQUIV', b: 'e1', span }];
  const { program } = jsonToProgram(j, X);
  assert.equal(assertsIndependence(indepSent), true);
  assert.equal(program.decls.filter((d) => d.op === 'COND_INDEP').length, 1);
  assert.equal(program.decls.filter((d) => d.op === 'DET_LINK').length, 1);
});

// ---- R6-3a: explicit binary partition sentence auto-anchoring ----------------
function r6BrassJson() {
  return {
    hyps: [
      { id: 'h0', name: 'brass', span: 'brass valves' },
      { id: 'h1', name: 'zinc', span: 'zinc valves' },
    ],
    events: [{ id: 'e0', name: 'star', span: 'star' }],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Brass accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'zinc for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: '1/2 of brass valves' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: '1/4 of zinc valves' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'A valve with a star was observed' }],
    nonAsserted: [], condIndep: [], detLinks: [], partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
}
function r6BrassX(first) {
  return [
    first,
    'Brass accounts for 1/2 of valves and zinc for 1/2.',
    'A star appears on 1/2 of brass valves and 1/4 of zinc valves.',
    'A valve with a star was observed.',
    'What is the probability the valve is brass?',
  ].join(' ');
}

test('R6 partition: All/Every binary sentences auto-anchor PART_EXHAUSTIVE', () => {
  for (const first of [
    'All its valves are brass or zinc.',
    'Every valve is either brass or zinc.',
  ]) {
    const X = r6BrassX(first);
    const { program, errors } = jsonToProgram(r6BrassJson(), X);
    assert.deepEqual(errors, [], first);
    const part = program.decls.find((d) => d.op === 'PART_EXHAUSTIVE');
    assert.equal(X.slice(part.span.s, part.span.e), first, first);
    assert.equal(checkSentenceAccounting(X, program).ok, true, first);
    const d = solve(program, X);
    assert.equal(d.status, 'ANSWER', `${first} -> ${show(d)}`);
    assert.equal(fmt(d.value), '2/3', first);
  }
});

test('R6 partition: only-workshops sentence auto-anchors PART_EXHAUSTIVE', () => {
  const first = 'Alpha and Beta are the only workshops that serviced the cart.';
  const X = [
    first,
    'Alpha accounts for 1/2 of carts and Beta accounts for 1/2.',
    'A star appears on 1/2 of Alpha carts and 1/4 of Beta carts.',
    'A cart with a star was observed.',
    'What is the probability the cart is Alpha?',
  ].join(' ');
  const j = {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Alpha carts' },
      { id: 'h1', name: 'Beta', span: 'Beta carts' },
    ],
    events: [{ id: 'e0', name: 'star', span: 'star' }],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'Beta accounts for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: '1/2 of Alpha carts' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: '1/4 of Beta carts' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'A cart with a star was observed' }],
    nonAsserted: [], condIndep: [], detLinks: [], partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  const part = program.decls.find((d) => d.op === 'PART_EXHAUSTIVE');
  assert.equal(X.slice(part.span.s, part.span.e), first);
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '2/3');
});

test('R6 partition: non-exhaustive or negated sentences never anchor', () => {
  for (const first of [
    'Some valves are brass or zinc.',
    'All its valves are not brass or zinc.',
  ]) {
    const X = r6BrassX(first);
    const { program } = jsonToProgram(r6BrassJson(), X);
    const part = program.decls.find((d) => d.op === 'PART_EXHAUSTIVE');
    assert.notEqual(X.slice(part.span.s, part.span.e), first, first);
    const d = solve(program, X);
    assert.equal(d.status, 'REFUSED', first);
    assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE', first);
  }
});

// ---- R6-3b: unqueried other-stratum qualitative endpoint LIK filter ---------
const R6_LOWER_QUAL =
  'On the lower deck every item, of either alloy, passes the load test.';
function r6DeckJson({ deck = 'upper', lower = R6_LOWER_QUAL,
                       extraObs = [], extraGiven = [] } = {}) {
  const X = [
    'Every panel is either Alpha or Beta.',
    'On the upper deck, Alpha accounts for 1/2 of panels and Beta for 1/2.',
    'On the upper deck, a star appears on 1/2 of Alpha panels and 1/4 of Beta panels.',
    lower,
    `A panel from the ${deck} deck was observed with a star.`,
    `What is the probability a panel from the ${deck} deck with a star is Alpha?`,
  ].join(' ');
  const lowerSpan = lower.replace(/\.$/, '');
  const j = {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Alpha' },
      { id: 'h1', name: 'Beta', span: 'Beta' },
    ],
    events: [
      { id: 'e0', name: 'star', span: 'star' },
      { id: 'e1', name: 'load test', span: 'load test' },
    ],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'Beta for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: '1/2 of Alpha panels' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: '1/4 of Beta panels' },
      { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/1', span: lowerSpan },
      { ev: 'e1', pol: 'POS', hyp: 'h1', value: '1/1', span: lowerSpan },
    ],
    posteriors: [],
    obs: [
      { ev: 'e0', pol: 'POS', span: `from the ${deck} deck was observed with a star` },
      ...extraObs,
    ],
    nonAsserted: [], condIndep: [], detLinks: [], partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, ...extraGiven] },
  };
  return { X, j };
}
const r6LikE1 = (p) => p.decls.filter((d) => d.op === 'LIK' && d.ev === 'e1').length;

test('R6 strata lik: unqueried other-stratum qualitative endpoint LIKs filter cleanly', () => {
  const { X, j } = r6DeckJson();
  const { program, errors } = jsonToProgram(j, X);
  assert.deepEqual(errors, []);
  assert.equal(program.query.ctx, 'ctx_upper_deck');
  assert.equal(r6LikE1(program), 0);
  assert.equal(program.decls.filter((d) => d.op === 'EVENT' && d.id === 'e1').length, 0);
  const d = solve(program, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '2/3');
});

test('R6 strata lik: same-stratum, queried-event, and numeric variants fail closed', () => {
  const variants = [
    r6DeckJson({ deck: 'lower' }),
    r6DeckJson({
      extraObs: [{ ev: 'e1', pol: 'POS', span: 'passes the load test' }],
      extraGiven: [{ ev: 'e1', pol: 'POS' }],
    }),
    r6DeckJson({ lower: 'On the lower deck 1/2 of the items pass the load test.' }),
  ];
  for (const { X, j } of variants) {
    const { program, errors } = jsonToProgram(j, X);
    assert.equal(r6LikE1(program), 2, X);
    assert.ok(errors.some((e) => /no literal equal to the declared value/.test(e)),
      `${X} -> ${show(errors)}`);
  }
});

test('R6 strata lik: single-population qualitative endpoint LIK never filters', () => {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta for 1/2.',
    'A star appears on 1/2 of Alpha units and 1/4 of Beta units.',
    'Every item passes the load test.',
    'A unit with a star was observed.',
    'What is the probability the unit is Alpha?',
  ].join(' ');
  const j = {
    hyps: [
      { id: 'h0', name: 'Alpha', span: 'Alpha' },
      { id: 'h1', name: 'Beta', span: 'Beta' },
    ],
    events: [
      { id: 'e0', name: 'star', span: 'star' },
      { id: 'e1', name: 'load test', span: 'load test' },
    ],
    priors: [
      { hyp: 'h0', value: '1/2', span: 'Alpha accounts for 1/2' },
      { hyp: 'h1', value: '1/2', span: 'Beta for 1/2' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: '1/2 of Alpha units' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: '1/4 of Beta units' },
      { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/1', span: 'Every item passes the load test' },
      { ev: 'e1', pol: 'POS', hyp: 'h1', value: '1/1', span: 'Every item passes the load test' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'A unit with a star was observed' }],
    nonAsserted: [], condIndep: [], detLinks: [], partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
  };
  const { program, errors } = jsonToProgram(j, X);
  assert.equal(program.query.ctx, 'ctx0');
  assert.equal(r6LikE1(program), 2);
  assert.ok(errors.some((e) => /no literal equal to the declared value/.test(e)),
    show(errors));
});

// ---- R7 Fix 1: complete preposition set in VERB_PHRASE_LINK ----------------
test('R7 morphology: temporal/spatial prepositions license leading-verb inflection', () => {
  const open = inflectedNameVariants('open before dawn');
  assert.ok(open.includes('opening before dawn'));
  assert.ok(open.includes('opens before dawn'));
  assert.ok(open.includes('opened before dawn'));
  assert.ok(!open.some((v) => v.includes('openning')));
  assert.ok(inflectedNameRe('pass through inspection').test('its passing through inspection'));
  assert.ok(inflectedNameRe('pass through inspection').test('it passes through inspection'));
  assert.ok(inflectedNameRe('turn past zero').test('its turning past zero'));
  assert.ok(inflectedNameRe('close after dusk').test('its closing after dusk'));
  assert.ok(inflectedNameRe('fall below threshold').test('its falling below threshold'));
  // a noun phrase / link-less phrase still gains no verb inflection
  assert.ok(!inflectedNameVariants('amber notch').some((v) => v.includes('ambering')));
  assert.ok(!inflectedNameVariants('open gate').includes('opening gate'));
});

function openBc({ query: q } = {}) {
  const X = [
    'Every unit is either Alpha or Beta.',
    'Alpha accounts for 1/2 of units and Beta accounts for 1/2.',
    'The open before dawn appears for 4/5 of Alpha units and 1/5 of Beta units.',
    q ?? 'What is the probability the unit is Alpha, conditional on its opening before dawn?',
  ].join(' ');
  const lines = [
    `HYP h0 ${at(X, 'Alpha')}`,
    `HYP h1 ${at(X, 'Beta')}`,
    `PART_EXHAUSTIVE (h0 h1) ${at(X, 'Alpha or Beta')}`,
    `EVENT e0 ${at(X, 'open before dawn')}`,
    `PRIOR h0 1/2 ctx0 ${at(X, '1/2 of units')}`,
    `PRIOR h1 1/2 ctx0 ${at(X, 'Beta accounts for 1/2')}`,
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at(X, '4/5 of Alpha')}`,
    `LIK e0 POS GIVEN h1 1/5 ctx0 ${at(X, '1/5 of Beta')}`,
    'ASK_PROB h0 GIVEN (e0:POS) ctx0',
    'END',
  ];
  return { X, bc: withLedger(X, lines.join('\n')) };
}

test('R7 conditional given: "opening before dawn" witnesses the given', () => {
  const { X, bc } = openBc();
  const prog = toProgram(bc, X.length);
  const span = queryConditionalGivenSpan(X, prog, 'e0', 'POS');
  assert.ok(span, 'conditional clause span');
  assert.match(X.slice(span.s, span.e), /opening before dawn/);
  const d = solve(bc, X);
  assert.equal(d.status, 'ANSWER', show(d));
  assert.equal(fmt(d.value), '4/5');
  // a negated conditional never witnesses a POS given
  const neg = openBc({
    query: 'What is the probability the unit is Alpha, conditional on its not opening before dawn?',
  });
  assert.equal(solve(neg.bc, neg.X).code, 'MISSING_OBSERVATION_WITNESS');
  // a conditional clause over a different event never witnesses this given
  const other = openBc({
    query: 'What is the probability the unit is Alpha, conditional on its passing through inspection?',
  });
  assert.equal(solve(other.bc, other.X).code, 'MISSING_OBSERVATION_WITNESS');
});

// ---- R7 Fix 2: R4c prepositional origin includes in / at -------------------
test('R7 ownership: "fired in" / "cast at" origin priors ground; swapped owners refuse', () => {
  const cases = [
    {
      sent: 'Half of the vessels in a crate were fired in the Alpha kiln and half in the Beta kiln.',
      priors: [
        { needle: 'Half of the vessels in a crate were fired in the Alpha kiln', value: '1/2' },
        { needle: 'half in the Beta kiln', value: '1/2' },
      ],
      expect: '2/3',
    },
    {
      sent: 'Two thirds of the lots were cast at the Alpha foundry and one third at the Beta foundry.',
      priors: [
        { needle: 'Two thirds of the lots were cast at the Alpha foundry', value: '2/3' },
        { needle: 'one third at the Beta foundry', value: '1/3' },
      ],
      expect: '4/5',
    },
  ];
  for (const { sent, priors: p, expect } of cases) {
    const priors = p.map((q) => ({ sent, ...q }));
    const sents = [sent, LAMP_RATE];
    const { X, bc } = buildUnit({ sents, priors, liks: lampLiks() });
    const prog = toProgram(bc, X.length);
    assert.deepEqual(
      ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), [], sent);
    const d = solve(bc, X);
    assert.equal(d.status, 'ANSWER', `${sent} -> ${show(d)}`);
    assert.equal(fmt(d.value), expect);
    const swapped = buildUnit({
      sents,
      priors: [{ ...priors[0], hyp: 'h1' }, { ...priors[1], hyp: 'h0' }],
      liks: lampLiks(),
    });
    const sprog = toProgram(swapped.bc, X.length);
    assert.ok(
      ownershipCheck({ X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog })
        .some((f) => f.code === 'UNSTABLE_BINDING' && /owned by/.test(f.detail)),
      sent);
  }
});

test('R7 ownership: an in/at phrase never overrides the nearer grammatical owner', () => {
  const priors = [
    { sent: 'Half of Alpha lots were cast at the Beta foundry.', needle: 'Half of Alpha lots', value: '1/2' },
    { sent: 'Half of Beta lots were cast at the Alpha foundry.', needle: 'Half of Beta lots', value: '1/2' },
  ];
  const sents = [...priors.map((p) => p.sent), LAMP_RATE];
  const { X, bc } = buildUnit({ sents, priors, liks: lampLiks() });
  const prog = toProgram(bc, X.length);
  assert.deepEqual(
    ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog), prog }), []);
  const swapped = buildUnit({
    sents,
    priors: [{ ...priors[0], hyp: 'h1' }, { ...priors[1], hyp: 'h0' }],
    liks: lampLiks(),
  });
  const sprog = toProgram(swapped.bc, X.length);
  const failures = ownershipCheck({
    X, decls: sprog.decls, ...nameMaps(X, sprog), prog: sprog,
  });
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by "Alpha"/.test(f.detail)),
    JSON.stringify(failures));
  assert.ok(
    failures.some((f) => f.code === 'UNSTABLE_BINDING' && /owned by "Beta"/.test(f.detail)),
    JSON.stringify(failures));
});
