import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { parseBytecode, toProgram } from '../src/kernel/parser.js';
import { solve } from '../src/kernel/solve.js';
import { autoSentTags } from '../src/kernel/sentences.js';
import { scanLiterals } from '../src/kernel/literals.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { applyPermutation, pairwiseSwaps } from '../src/invariance/swap.js';
import { spanScoped, heuristicAssertedHead } from '../src/invariance/ablation.js';
import {
  makeSemanticExtractor,
  makePositionalExtractor,
  makeMisbindingExtractor,
} from '../src/invariance/reference-extractors.js';

// V2: solve(program, X) requires a complete sentence-partition ledger — the
// fixtures simulate compiler-side extraction output, so append the derived
// SENT_TAG ledger the reference extractor would emit.
function withLedger(progStr, X) {
  const tags = autoSentTags(X, toProgram(progStr))
    .map((t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`)
    .join('\n');
  return progStr.replace('ASK_PROB', `${tags}\nASK_PROB`);
}

// ---------- swap engine ----------
test('swap: literal permutation remaps substrings + spans', () => {
  const X = 'Alpha, compared to the 20% share for Beta, accounted for 80%.';
  const sites = scanLiterals(X); // [20% @24-27, 80% @54-57]
  assert.equal(sites.length, 2);
  const { X: Xp, remappedSites } = applyPermutation(X, sites, [1, 0]);
  assert.equal(
    Xp,
    'Alpha, compared to the 80% share for Beta, accounted for 20%.'
  );
  assert.equal(remappedSites[0].text, '80%');
  assert.equal(remappedSites[1].text, '20%');
});

test('swap: word-form literals of different lengths', () => {
  const X = 'the first test is right five eighths of the time; the second, seven twelfths';
  const sites = scanLiterals(X);
  assert.equal(sites.length, 2);
  const { X: Xp } = applyPermutation(X, sites, [1, 0]);
  assert.ok(Xp.includes('seven twelfths of the time'));
  assert.ok(Xp.includes('the second, five eighths'));
});

test('swap: pairwiseSwaps skips identical literals', () => {
  const sites = [
    { s: 0, e: 3, text: '20%', value: R.make(1n, 5n) },
    { s: 5, e: 8, text: '20%', value: R.make(1n, 5n) },
    { s: 10, e: 13, text: '80%', value: R.make(4n, 5n) },
  ];
  const swaps = pairwiseSwaps(sites);
  // only (20%_a,80%) and (20%_b,80%) -> 2 swaps; (20%,20%) skipped
  assert.equal(swaps.length, 2);
});

// ---------- the Bulma positional-bug probe ----------
// "Alpha, compared to the 20% share for Beta, accounted for 80%."
// Semantic ground truth: PRIOR alpha=4/5, PRIOR beta=1/5.
// The October A1 positional extractor emits a plausible P1 but re-emits the
// memorized binding on counterfactuals -> gate must fire UNSTABLE_BINDING.
const BULMA_X =
  'In the last audit, Alpha, compared to the 20% share for Beta, accounted for 80% of flagged cases. ' +
  'Among Alpha cases, nine tenths get flagged; among Beta cases, one tenth. ' +
  'A case has been flagged. What is the probability it is Alpha?';

function bulmaProgram() {
  const i80 = BULMA_X.indexOf('80%');
  const i20 = BULMA_X.indexOf('20%');
  const i910 = BULMA_X.indexOf('nine tenths');
  const i110 = BULMA_X.indexOf('one tenth');
  const flagSeg = BULMA_X.indexOf('flagged');
  const obsSeg = BULMA_X.indexOf('has been flagged');
  const iAlpha = BULMA_X.indexOf('Alpha');
  const iBeta = BULMA_X.indexOf('Beta');
  return `HYP h_alpha [${iAlpha},${iAlpha + 5}]
HYP h_beta [${iBeta},${iBeta + 4}]
PART_EXHAUSTIVE (h_alpha h_beta) [${iAlpha},${iBeta + 4}]
EVENT e_flag [${flagSeg},${flagSeg + 7}]
PRIOR h_alpha 4/5 ctx0 [${i80},${i80 + 3}]
PRIOR h_beta 1/5 ctx0 [${i20},${i20 + 3}]
LIK e_flag POS GIVEN h_alpha 9/10 ctx0 [${i910},${i910 + 11}]
LIK e_flag POS GIVEN h_beta 1/10 ctx0 [${i110},${i110 + 9}]
OBS e_flag POS ctx0 [${obsSeg},${obsSeg + 16}]
ASK_PROB h_alpha GIVEN (e_flag:POS) ctx0
END`;
}

test('gate: semantic extractor passes literal-swap invariance', async () => {
  const prog = bulmaProgram();
  const d = solve(withLedger(prog, BULMA_X), BULMA_X);
  assert.equal(d.status, 'ANSWER');
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: BULMA_X,
    extract: makeSemanticExtractor(prog),
  });
  assert.equal(res.verdict, 'PASS');
  assert.ok(res.slots.swaps.every((s) => s.ok));
});

test('gate: Bulma positional bug deterministically caught as UNSTABLE_BINDING', async () => {
  const prog = bulmaProgram();
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: BULMA_X,
    extract: makePositionalExtractor(prog),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.ok(res.slots.swaps.some((s) => s.ok === false));
});

test('gate: kernel refusals pass through (MISSING_BASE_RATE)', async () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
OBS e0 POS ctx0 [60,70]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: 'x'.repeat(80),
    extract: () => prog,
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'MISSING_BASE_RATE');
});

// ---------- context ablation ----------
test('ablation: retracted/obsolete scope suppresses assertion', () => {
  const X =
    'The obsolete 1998 manual states that four fifths of flagged cases are Alpha. ' +
    'The 2026 audit supersedes it.';
  const i = X.indexOf('four fifths');
  const probe = spanScoped(X, { s: i, e: i + 11 });
  assert.equal(probe.scoped, true);
  const aIso = heuristicAssertedHead('four fifths of flagged cases are Alpha', 'four fifths of flagged cases are Alpha');
  const aFull = heuristicAssertedHead('four fifths of flagged cases are Alpha', X);
  assert.ok(aIso - aFull > 0.4);
});

test('gate: context-suppressed premise -> SCOPED_OR_RETRACTED_PREMISE', async () => {
  const X =
    'The obsolete 1998 manual states that the prior for Alpha is 4/5 and Beta is 1/5. ' +
    'When flagged, Alpha shows 9/10 of the time and Beta 1/10. A case was flagged.';
  const i45 = X.indexOf('4/5');
  const i15 = X.indexOf('1/5');
  const i910 = X.indexOf('9/10');
  const i110 = X.lastIndexOf('1/10');
  const flag = X.indexOf('flagged.');
  const iAlpha = X.indexOf('Alpha');
  const iBeta = X.indexOf('Beta');
  const iEv = X.indexOf('flagged,');
  const prog = `HYP h0 [${iAlpha},${iAlpha + 5}]
HYP h1 [${iBeta},${iBeta + 4}]
PART_EXHAUSTIVE (h0 h1) [${iAlpha},${iBeta + 4}]
EVENT e0 [${iEv},${iEv + 7}]
PRIOR h0 4/5 ctx0 [${i45},${i45 + 3}]
PRIOR h1 1/5 ctx0 [${i15},${i15 + 3}]
LIK e0 POS GIVEN h0 9/10 ctx0 [${i910},${i910 + 4}]
LIK e0 POS GIVEN h1 1/10 ctx0 [${i110},${i110 + 4}]
OBS e0 POS ctx0 [${flag},${flag + 8}]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X,
    extract: makeSemanticExtractor(prog),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'SCOPED_OR_RETRACTED_PREMISE');
});

test('gate: clean asserted context passes ablation', async () => {
  const X =
    'The current 2026 audit states the prior for Alpha is 4/5 and Beta is 1/5. ' +
    'When flagged, Alpha shows 9/10 of the time and Beta 1/10. A case was flagged.';
  const i45 = X.indexOf('4/5');
  const i15 = X.indexOf('1/5');
  const i910 = X.indexOf('9/10');
  const i110 = X.lastIndexOf('1/10');
  const flag = X.indexOf('flagged.');
  const iAlpha = X.indexOf('Alpha');
  const iBeta = X.indexOf('Beta');
  const iEv = X.indexOf('flagged,');
  const prog = `HYP h0 [${iAlpha},${iAlpha + 5}]
HYP h1 [${iBeta},${iBeta + 4}]
PART_EXHAUSTIVE (h0 h1) [${iAlpha},${iBeta + 4}]
EVENT e0 [${iEv},${iEv + 7}]
PRIOR h0 4/5 ctx0 [${i45},${i45 + 3}]
PRIOR h1 1/5 ctx0 [${i15},${i15 + 3}]
LIK e0 POS GIVEN h0 9/10 ctx0 [${i910},${i910 + 4}]
LIK e0 POS GIVEN h1 1/10 ctx0 [${i110},${i110 + 4}]
OBS e0 POS ctx0 [${flag},${flag + 8}]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X,
    extract: makeSemanticExtractor(prog),
  });
  assert.equal(res.verdict, 'PASS');
});

// ---------- complement ----------
test('gate: complement check is exact on factor heads', async () => {
  const prog = bulmaProgram();
  const res = await verifyInvariance({
    offlineSyntheticBench: true, X: BULMA_X, extract: makeSemanticExtractor(prog) });
  assert.equal(res.slots.complement.ok, true);
  assert.ok(!res.slots.complement.skipped);
});

test('gate: qualScore complement violation caught', async () => {
  const prog = bulmaProgram();
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: BULMA_X,
    extract: makeSemanticExtractor(prog),
    qualScore: (_X, _q, negated) => (negated ? 0.3 : 0.6), // sums to 0.9
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'COMPLEMENT_VIOLATION');
});

// ---------- source-grounded ownership witness ----------
// "Cedar (the 40% share belongs to Quartz) made up 60% of pallets." — the
// correct prior binding is Cedar->60%, Quartz->40%. An extractor that
// systematically binds by literal position (first % to Cedar) releases the
// wrong answer on EVERY text — numeric invariance alone cannot see it.
const CEDAR_X =
  'Every pallet was either Cedar or Quartz. ' +
  'Cedar (the 40% share belongs to Quartz) made up 60% of pallets. ' +
  'The beacon lit for nine tenths of Cedar pallets and eight twentieths of Quartz pallets. ' +
  "This pallet's beacon lit. What is the probability that it was Cedar?";

function cedarProgram() {
  const i40 = CEDAR_X.indexOf('40%');
  const i60 = CEDAR_X.indexOf('60%');
  const i910 = CEDAR_X.indexOf('nine tenths');
  const i820 = CEDAR_X.indexOf('eight twentieths');
  const iCedar = CEDAR_X.indexOf('Cedar');
  const iQuartz = CEDAR_X.indexOf('Quartz');
  const iBeacon = CEDAR_X.indexOf('beacon lit');
  const iObs = CEDAR_X.indexOf("pallet's beacon lit");
  return `HYP h_cedar [${iCedar},${iCedar + 5}]
HYP h_quartz [${iQuartz},${iQuartz + 6}]
PART_EXHAUSTIVE (h_cedar h_quartz) [${iCedar},${iQuartz + 6}]
EVENT e_beacon [${iBeacon},${iBeacon + 6}]
PRIOR h_cedar 3/5 ctx0 [${i60},${i60 + 3}]
PRIOR h_quartz 2/5 ctx0 [${i40},${i40 + 3}]
LIK e_beacon POS GIVEN h_cedar 9/10 ctx0 [${i910},${i910 + 11}]
LIK e_beacon POS GIVEN h_quartz 2/5 ctx0 [${i820},${i820 + 16}]
OBS e_beacon POS ctx0 [${iObs},${iObs + 19}]
ASK_PROB h_cedar GIVEN (e_beacon:POS) ctx0
END`;
}

test('gate: source-grounded ownership — correct owner binding passes', async () => {
  const prog = cedarProgram();
  const d = solve(withLedger(prog, CEDAR_X), CEDAR_X);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '27/35'); // (3/5*9/10)/(3/5*9/10+2/5*2/5)
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: CEDAR_X,
    extract: makeSemanticExtractor(prog),
  });
  assert.equal(res.verdict, 'PASS', `unexpected ${res.code}: ${res.details}`);
});

test('gate: persistent owner misbinding caught as UNSTABLE_BINDING', async () => {
  const prog = cedarProgram();
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: CEDAR_X,
    extract: makeMisbindingExtractor(prog),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
});

// ---------- likelihood-direction strictness ----------
// "Filter the lot to beads whose tag flashed: 80% of those beads are Ash" —
// the clause asserts P(Ash|tag), not P(tag|Ash). A LIK bound there is the
// reversed-conditional smuggle even when the numeric value is faithful.
const ASH_X =
  'Every bead is Ash or Fir. The lot is 40% Ash and 60% Fir. ' +
  'Filter the lot to beads whose tag flashed: 80% of those beads are Ash and 20% are Fir. ' +
  "What is the chance a randomly selected bead's tag flashes?";

function ashProgram(op /* 'LIK' | 'POSTERIOR_STATED' */) {
  const iAsh = ASH_X.indexOf('Ash');
  const iFir = ASH_X.indexOf('Fir');
  const i40 = ASH_X.indexOf('40%');
  const i60 = ASH_X.indexOf('60%');
  const i80 = ASH_X.indexOf('80%');
  const i20 = ASH_X.indexOf('20%');
  const iTag = ASH_X.indexOf('tag flashed');
  const postLines =
    op === 'LIK'
      ? `LIK e_tag POS GIVEN h_ash 4/5 ctx0 [${i80},${i80 + 3}]\nLIK e_tag POS GIVEN h_fir 1/5 ctx0 [${i20},${i20 + 3}]`
      : `POSTERIOR_STATED h_ash GIVEN e_tag POS 4/5 ctx0 [${i80},${i80 + 3}]\nPOSTERIOR_STATED h_fir GIVEN e_tag POS 1/5 ctx0 [${i20},${i20 + 3}]`;
  return `HYP h_ash [${iAsh},${iAsh + 3}]
HYP h_fir [${iFir},${iFir + 3}]
PART_EXHAUSTIVE (h_ash h_fir) [${iAsh},${iFir + 3}]
EVENT e_tag [${iTag},${iTag + 3}]
PRIOR h_ash 2/5 ctx0 [${i40},${i40 + 3}]
PRIOR h_fir 3/5 ctx0 [${i60},${i60 + 3}]
${postLines}
ASK_PROB e_tag GIVEN () ctx0
END`;
}

test('gate: LIK bound inside a posterior-direction clause refused', async () => {
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: ASH_X,
    extract: makeSemanticExtractor(ashProgram('LIK')),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
});

test('gate: correctly-typed POSTERIOR_STATED still refused by kernel (tag marginal underdetermined)', async () => {
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: ASH_X,
    extract: makeSemanticExtractor(ashProgram('POSTERIOR_STATED')),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'INCOMPLETE_FACTORIZATION');
});

// ---------- affirmative COND_INDEP license ----------
// "The tone and blink are NOT conditionally independent" must not serve as
// an independence license: a license is only valid over text that AFFIRMS
// independence.
const SAFFRON_X =
  'Every crate belongs to Saffron or Teal. Saffron makes up 25% of crates and Teal 75%. ' +
  'A Saffron crate triggers the tone with probability 80% and the blink with probability 60%. ' +
  'A Teal crate triggers the tone with probability 20% and the blink with probability 40%. ' +
  'The tone and blink are not conditionally independent within either cohort. ' +
  'This crate triggered both the tone and blink. What is the probability it is Saffron?';

function saffronProgram(X = SAFFRON_X) {
  const iSaf = X.indexOf('Saffron');
  const iTeal = X.indexOf('Teal');
  const i25 = X.indexOf('25%');
  const i75 = X.indexOf('75%');
  const i80 = X.indexOf('80%');
  const i60 = X.indexOf('60%');
  const i20 = X.indexOf('20%');
  const i40 = X.indexOf('40%');
  const iTone = X.indexOf('tone');
  const iBlink = X.indexOf('blink');
  const iLic = X.indexOf('The tone and blink');
  const iLicEnd = X.indexOf('cohort') + 6;
  const iObs = X.indexOf('This crate triggered');
  return `HYP h_saffron [${iSaf},${iSaf + 7}]
HYP h_teal [${iTeal},${iTeal + 4}]
PART_EXHAUSTIVE (h_saffron h_teal) [${iSaf},${iTeal + 4}]
EVENT e_tone [${iTone},${iTone + 4}]
EVENT e_blink [${iBlink},${iBlink + 5}]
PRIOR h_saffron 1/4 ctx0 [${i25},${i25 + 3}]
PRIOR h_teal 3/4 ctx0 [${i75},${i75 + 3}]
LIK e_tone POS GIVEN h_saffron 4/5 ctx0 [${i80},${i80 + 3}]
LIK e_blink POS GIVEN h_saffron 3/5 ctx0 [${i60},${i60 + 3}]
LIK e_tone POS GIVEN h_teal 1/5 ctx0 [${i20},${i20 + 3}]
LIK e_blink POS GIVEN h_teal 2/5 ctx0 [${i40},${i40 + 3}]
OBS e_tone POS ctx0 [${iObs},${iObs + 20}]
OBS e_blink POS ctx0 [${iObs},${iObs + 20}]
COND_INDEP (e_tone e_blink) GIVEN * ctx0 [${iLic},${iLicEnd}]
ASK_PROB h_saffron GIVEN (e_tone:POS e_blink:POS) ctx0
END`;
}

test('gate: negated independence clause is not an affirmative license', async () => {
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: SAFFRON_X,
    extract: makeSemanticExtractor(saffronProgram()),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNLICENSED_INDEPENDENCE');
});

test('gate: affirmative independence clause licenses the joint', async () => {
  const X2 = SAFFRON_X.replace(
    'are not conditionally independent',
    'are conditionally independent'
  );
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: X2,
    extract: makeSemanticExtractor(saffronProgram(X2)),
  });
  assert.equal(res.verdict, 'PASS', `unexpected ${res.code}: ${res.details}`);
});

// ---------- backward-governing scope ----------
// "The counts ... were struck from the record" suppresses the premise even
// though the scoping sentence FOLLOWS it — and the figures sit inside a
// quoted string besides.
const ELM_X =
  'Every vial is Elm or Flax. The log read, "Elm makes up 75% of vials and Flax 25%." ' +
  'The counts came from a tabletop exercise and were struck from the record. ' +
  'The lamp lights for 80% of Elm vials and 20% of Flax vials. ' +
  "This vial's lamp lit. What is the probability it is Elm?";

function elmProgram(withNonAsserted) {
  const iElm = ELM_X.indexOf('Elm');
  const iFlax = ELM_X.indexOf('Flax');
  const i75 = ELM_X.indexOf('75%');
  const i25 = ELM_X.indexOf('25%');
  const i80 = ELM_X.indexOf('80%');
  const i20 = ELM_X.indexOf('20%');
  const iLamp = ELM_X.indexOf('lamp');
  const iObs = ELM_X.indexOf("vial's lamp lit");
  const iLog = ELM_X.indexOf('Elm makes up');
  const iLogEnd = ELM_X.indexOf('Flax 25%') + 8;
  const na = withNonAsserted
    ? `NON_ASSERTED [${iLog},${iLogEnd}] RETRACTED\n`
    : '';
  return `HYP h_elm [${iElm},${iElm + 3}]
HYP h_flax [${iFlax},${iFlax + 4}]
PART_EXHAUSTIVE (h_elm h_flax) [${iElm},${iFlax + 4}]
EVENT e_lamp [${iLamp},${iLamp + 4}]
PRIOR h_elm 3/4 ctx0 [${i75},${i75 + 3}]
PRIOR h_flax 1/4 ctx0 [${i25},${i25 + 3}]
LIK e_lamp POS GIVEN h_elm 4/5 ctx0 [${i80},${i80 + 3}]
LIK e_lamp POS GIVEN h_flax 1/5 ctx0 [${i20},${i20 + 3}]
${na}OBS e_lamp POS ctx0 [${iObs},${iObs + 15}]
ASK_PROB h_elm GIVEN (e_lamp:POS) ctx0
END`;
}

test('gate: retracted priors typed NON_ASSERTED -> SCOPED_OR_RETRACTED_PREMISE', async () => {
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: ELM_X,
    extract: makeSemanticExtractor(elmProgram(true)),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'SCOPED_OR_RETRACTED_PREMISE');
});

test('gate: retracted priors asserted verbatim -> SCOPED via backward scope + quotes', async () => {
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: ELM_X,
    extract: makeSemanticExtractor(elmProgram(false)),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'SCOPED_OR_RETRACTED_PREMISE');
});

// ---------- distinct-population contexts ----------
// Priors from South; likelihoods exist only for North. A faithful program
// ledgering the figures under a North CtxId refuses CONTEXT_MISMATCH in the
// kernel; a program collapsing everything into ctx0 must be refused by the
// gate's source-grounded population witness.
const OPAL_X =
  'Every South intake is Opal or Jade. In South, Opal accounts for 20% and Jade 80%. ' +
  'The only alarm figures come from North: there, 90% of Opal intakes and 30% of Jade intakes trigger the alarm. ' +
  'A South intake triggered the alarm. What is the probability that the South intake is Opal?';

function opalProgram(likCtx) {
  const iOpal = OPAL_X.indexOf('Opal');
  const iJade = OPAL_X.indexOf('Jade');
  const i20 = OPAL_X.indexOf('20%');
  const i80 = OPAL_X.indexOf('80%');
  const i90 = OPAL_X.indexOf('90%');
  const i30 = OPAL_X.indexOf('30%');
  const iAlarm = OPAL_X.indexOf('alarm');
  const iObs = OPAL_X.indexOf('A South intake triggered');
  return `HYP h_opal [${iOpal},${iOpal + 4}]
HYP h_jade [${iJade},${iJade + 4}]
PART_EXHAUSTIVE (h_opal h_jade) [${iOpal},${iJade + 4}]
EVENT e_alarm [${iAlarm},${iAlarm + 5}]
PRIOR h_opal 1/5 ctx0 [${i20},${i20 + 3}]
PRIOR h_jade 4/5 ctx0 [${i80},${i80 + 3}]
LIK e_alarm POS GIVEN h_opal 9/10 ${likCtx} [${i90},${i90 + 3}]
LIK e_alarm POS GIVEN h_jade 3/10 ${likCtx} [${i30},${i30 + 3}]
OBS e_alarm POS ctx0 [${iObs},${iObs + 24}]
ASK_PROB h_opal GIVEN (e_alarm:POS) ctx0
END`;
}

test('gate: distinct population CtxIds refuse CONTEXT_MISMATCH in kernel', async () => {
  const d = solve(withLedger(opalProgram('ctx_north'), OPAL_X), OPAL_X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'CONTEXT_MISMATCH');
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: OPAL_X,
    extract: makeSemanticExtractor(opalProgram('ctx_north')),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'CONTEXT_MISMATCH');
});

test('gate: collapsed-population CtxIds refused by source ctx witness', async () => {
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: OPAL_X,
    extract: makeSemanticExtractor(opalProgram('ctx0')),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'CONTEXT_MISMATCH');
});

// ---------- same-event query conditioning ----------
test('gate: query conditioned on its own event resolves exactly to 1/1', async () => {
  const prog = cedarProgram().replace(
    'ASK_PROB h_cedar GIVEN',
    'ASK_PROB e_beacon GIVEN'
  );
  const res = await verifyInvariance({
    offlineSyntheticBench: true,
    X: CEDAR_X,
    extract: makeSemanticExtractor(prog),
  });
  assert.equal(res.verdict, 'PASS', `unexpected ${res.code}: ${res.details}`);
  assert.equal(R.toString(res.decision.value), '1/1');
});
