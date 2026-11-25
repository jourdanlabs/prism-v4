import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { parseLiteral, scanLiterals } from '../src/kernel/literals.js';
import { parseBytecode, toBytecode, toJSON, fromJSON, ParseError } from '../src/kernel/parser.js';
import { solve } from '../src/kernel/solve.js';
import { buildReceipt, verifyReceipt } from '../src/kernel/receipt.js';

// ---------- rationals ----------
test('rational: gcd normalization, zero denominator', () => {
  const r = R.make(4n, 8n);
  assert.equal(r.n, 1n);
  assert.equal(r.d, 2n);
  assert.throws(() => R.make(1n, 0n));
  assert.equal(R.toString(R.make(-6n, -9n)), '2/3');
});

test('rational: exact arithmetic p*q/(p*q+(1-p)*r)', () => {
  // classic: prior 1/5, sens 4/5, fpr 1/10 -> post = (1/5*4/5)/(1/5*4/5 + 4/5*1/10) = 2/3
  const p = R.make(1n, 5n);
  const q = R.make(4n, 5n);
  const r = R.make(1n, 10n);
  const num = R.mul(p, q);
  const den = R.add(num, R.mul(R.sub(R.one(), p), r));
  const post = R.div(num, den);
  assert.equal(R.toString(post), '2/3');
});

test('literals: parse digits, %, decimals, fraction words', () => {
  assert.equal(R.toString(parseLiteral('62.5%')), '5/8');
  assert.equal(R.toString(parseLiteral('5/8')), '5/8');
  assert.equal(R.toString(parseLiteral('0.625')), '5/8');
  assert.equal(R.toString(parseLiteral('five eighths')), '5/8');
  assert.equal(R.toString(parseLiteral('seven twelfths')), '7/12');
  assert.equal(R.toString(parseLiteral('a third')), '1/3');
  assert.equal(R.toString(parseLiteral('half')), '1/2');
  assert.equal(R.toString(parseLiteral('two fifths')), '2/5');
  assert.equal(R.toString(parseLiteral('3 in 10')), '3/10');
});

test('literals: scan finds sites in document order', () => {
  const x = 'Alpha, compared to the 20% share for Beta, accounted for 80%.';
  const sites = scanLiterals(x);
  assert.equal(sites.length, 2);
  assert.equal(sites[0].text, '20%');
  assert.equal(R.toString(sites[0].value), '1/5');
  assert.equal(R.toString(sites[1].value), '4/5');
});

// ---------- parser ----------
const GOOD = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
PRIOR h0 1/5 ctx0 [0,10]
PRIOR h1 4/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
OBS e0 POS ctx0 [60,70]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;

test('parser: full program round-trips', () => {
  const prog = parseBytecode(GOOD);
  assert.equal(prog.decls.length, 9);
  assert.equal(prog.query.target, 'h0');
  const ser = toBytecode(prog);
  const prog2 = parseBytecode(ser);
  assert.deepEqual(toJSON(prog2), toJSON(prog));
  const j = toJSON(prog);
  const prog3 = fromJSON(JSON.parse(JSON.stringify(j)));
  assert.deepEqual(toJSON(prog3), j);
});

test('parser: rejects malformed programs', () => {
  assert.throws(() => parseBytecode('PRIOR h0 1/2 ctx0 [0,5]\nASK_PROB h0 GIVEN () ctx0\nEND'), ParseError);
  assert.throws(() => parseBytecode(GOOD.replace('END', 'EXTRA END')), ParseError);
  assert.throws(
    () => parseBytecode(GOOD.replace('PRIOR h0 1/5', 'PRIOR h0 5/0')),
    ParseError
  );
  assert.throws(
    () => parseBytecode(GOOD.replace('PRIOR h0 1/5', 'PRIOR h0 6/5')),
    ParseError
  ); // probability out of [0,1]
  assert.throws(() => parseBytecode(GOOD.replace('LIK e0 POS GIVEN h0', 'LIK e0 POS GIVEN h9')), ParseError);
});

// ---------- solver: clean answer ----------
test('solve: exact posterior, witnesses bound', () => {
  const X = null;
  const d = solve(GOOD, null);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '2/3'); // (1/5*4/5) / (1/5*4/5 + 4/5*1/10)
  assert.ok(d.witnesses.length >= 5);
});

test('solve: complement via NOT target sums to 1', () => {
  const X = null;
  const neg = parseBytecode(GOOD.replace('ASK_PROB h0', 'ASK_PROB NOT h0'));
  const d = solve(neg, null);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '1/3');
});

test('solve: event target conditioned on itself is tautologically exact', () => {
  const prog = GOOD.replace('ASK_PROB h0 GIVEN', 'ASK_PROB e0 GIVEN');
  const d = solve(prog, null);
  assert.equal(d.status, 'ANSWER');
  // P(e0 | e0 observed POS) === 1/1 exactly — conditioning on the queried
  // event is a tautology, not a marginalization problem.
  assert.equal(R.toString(d.value), '1/1');
});

test('solve: event target conditioned on its own negation is exactly zero', () => {
  const prog = GOOD
    .replace('OBS e0 POS', 'OBS e0 NEG')
    .replace('ASK_PROB h0 GIVEN (e0:POS)', 'ASK_PROB e0 GIVEN (e0:NEG)');
  const d = solve(prog, null);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '0/1');
});

test('solve: NOT event target conditioned on itself is exactly zero', () => {
  const prog = GOOD.replace('ASK_PROB h0 GIVEN', 'ASK_PROB NOT e0 GIVEN');
  const d = solve(prog, null);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '0/1');
});

test('solve: event-target tautology still requires an OBS witness', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
ASK_PROB e0 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'MISSING_OBSERVATION_WITNESS');
});

test('solve: event target marginalizes over partition for a DISTINCT event', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
EVENT e1 [50,60]
PRIOR h0 1/5 ctx0 [0,10]
PRIOR h1 4/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
LIK e1 POS GIVEN h0 1/2 ctx0 [50,60]
LIK e1 POS GIVEN h1 9/10 ctx0 [60,70]
OBS e0 POS ctx0 [70,80]
COND_INDEP (e0 e1) GIVEN * ctx0 [80,90]
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'ANSWER');
  // P(e1|e0+) = post(h0|e0)*L(e1|h0) + post(h1|e0)*L(e1|h1)
  // = 2/3*1/2 + 1/3*9/10 = 1/3 + 3/10 = 19/30
  assert.equal(R.toString(d.value), '19/30');
});

test('solve: distinct event target without COND_INDEP refuses UNLICENSED_INDEPENDENCE', () => {
  // Marginalizing P(e1|e0) = sum_h post(h|e0)*L(e1|h) presumes e1 is
  // conditionally independent of e0 under each hypothesis. Deterministically
  // linked signals violate that; absent a license the kernel must refuse.
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
EVENT e1 [50,60]
PRIOR h0 1/5 ctx0 [0,10]
PRIOR h1 4/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
LIK e1 POS GIVEN h0 1/2 ctx0 [50,60]
LIK e1 POS GIVEN h1 9/10 ctx0 [60,70]
OBS e0 POS ctx0 [70,80]
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNLICENSED_INDEPENDENCE');
});

test('solve: zero-support self-conditioning refuses DEGENERATE_PARTITION', () => {
  // Every lamp likelihood is zero: P(lamp lit) = 0, so conditioning on the
  // lamp having lit is impossible. The tautology shortcut may not release
  // 1/1 on an empty event.
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
PRIOR h0 2/5 ctx0 [0,10]
PRIOR h1 3/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 0/1 ctx0 [30,40]
LIK e0 POS GIVEN h1 0/1 ctx0 [40,50]
OBS e0 POS ctx0 [50,60]
ASK_PROB e0 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'DEGENERATE_PARTITION');
});

test('solve: positive-support self-conditioning still releases the tautology', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
PRIOR h0 2/5 ctx0 [0,10]
PRIOR h1 3/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/5 ctx0 [40,50]
OBS e0 POS ctx0 [50,60]
ASK_PROB e0 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '1/1');
});

test('solve: stated posterior with possible zero support refuses ZERO_SUPPORT_POSSIBLE', () => {
  // V2 replay: priors + partition + stated posterior but no likelihoods.
  // There is no direct-posterior bypass — the query routes through
  // solveCredal, where the conditioning event's mass is not bounded below
  // by zero -> ZERO_SUPPORT_POSSIBLE, not a 4/5 release.
  const prog = `HYP h0 [24,27]
HYP h1 [31,34]
PART_EXHAUSTIVE (h0 h1) [24,34]
EVENT e0 [101,105]
PRIOR h0 2/5 ctx0 [53,56]
PRIOR h1 3/5 ctx0 [90,93]
POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 [120,123]
OBS e0 POS ctx0 [133,176]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'ZERO_SUPPORT_POSSIBLE');
});

test('solve: stated posterior on impossible evidence refuses DEGENERATE_PARTITION', () => {
  // V2 replay: full factorization declares the evidence impossible under
  // both hypotheses; the stated 4/5 posterior cannot release around the
  // zero-support check.
  const prog = `HYP h0 [24,27]
HYP h1 [31,34]
PART_EXHAUSTIVE (h0 h1) [24,34]
EVENT e0 [99,103]
PRIOR h0 2/5 ctx0 [53,56]
PRIOR h1 3/5 ctx0 [90,93]
LIK e0 POS GIVEN h0 0/1 ctx0 [115,117]
LIK e0 POS GIVEN h1 0/1 ctx0 [138,140]
POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 [183,186]
OBS e0 POS ctx0 [196,239]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'DEGENERATE_PARTITION');
});

// ---------- the 7 typed refusals ----------
test('refusal: MISSING_BASE_RATE (LIK+OBS, no priors)', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
OBS e0 POS ctx0 [60,70]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'MISSING_BASE_RATE');
});

test('refusal: MISSING_BASE_RATE (priors not summing to 1)', () => {
  const prog = GOOD.replace('PRIOR h1 4/5 ctx0', 'PRIOR h1 3/5 ctx0');
  const d = solve(prog, null);
  assert.equal(d.code, 'MISSING_BASE_RATE');
});

test('refusal: REVERSED_CONDITIONAL (stated posterior cannot substitute for LIK)', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
EVENT e1 [50,60]
PRIOR h0 1/5 ctx0 [0,10]
PRIOR h1 4/5 ctx0 [10,20]
POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 [30,40]
OBS e0 POS ctx0 [60,70]
OBS e1 POS ctx0 [70,80]
COND_INDEP (e0 e1) GIVEN * ctx0 [30,60]
ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'REVERSED_CONDITIONAL');
});

test('refusal: UNLICENSED_INDEPENDENCE (2 obs, no COND_INDEP)', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
EVENT e1 [50,60]
PRIOR h0 1/5 ctx0 [0,10]
PRIOR h1 4/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
LIK e1 POS GIVEN h0 1/2 ctx0 [50,60]
LIK e1 POS GIVEN h1 9/10 ctx0 [60,70]
OBS e0 POS ctx0 [70,80]
OBS e1 POS ctx0 [80,90]
ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.code, 'UNLICENSED_INDEPENDENCE');
});

test('solve: licensed independence with COND_INDEP computes', () => {
  const prog = `HYP h0 [0,10]
HYP h1 [10,20]
PART_EXHAUSTIVE (h0 h1) [0,20]
EVENT e0 [30,40]
EVENT e1 [50,60]
PRIOR h0 1/5 ctx0 [0,10]
PRIOR h1 4/5 ctx0 [10,20]
LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]
LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]
LIK e1 POS GIVEN h0 1/2 ctx0 [50,60]
LIK e1 POS GIVEN h1 9/10 ctx0 [60,70]
OBS e0 POS ctx0 [70,80]
OBS e1 POS ctx0 [80,90]
COND_INDEP (e0 e1) GIVEN * ctx0 [30,60]
ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0
END`;
  const d = solve(prog, null);
  assert.equal(d.status, 'ANSWER');
  // w0 = 1/5 * 4/5 * 1/2 = 2/25 ; w1 = 4/5 * 1/10 * 9/10 = 9/125
  // z = 10/125 + 9/125 = 19/125 -> post = (10/125)/(19/125) = 10/19
  assert.equal(R.toString(d.value), '10/19');
});

test('refusal: DUPLICATE_EVIDENCE (same EvId twice in givens)', () => {
  const prog = GOOD.replace('ASK_PROB h0 GIVEN (e0:POS)', 'ASK_PROB h0 GIVEN (e0:POS e0:POS)');
  const d = solve(prog, null);
  assert.equal(d.code, 'DUPLICATE_EVIDENCE');
});

test('refusal: DUPLICATE_EVIDENCE (two OBS same EvId)', () => {
  const prog = GOOD.replace(
    'OBS e0 POS ctx0 [60,70]',
    'OBS e0 POS ctx0 [60,70]\nOBS e0 POS ctx0 [60,70]'
  );
  const d = solve(prog, null);
  assert.equal(d.code, 'DUPLICATE_EVIDENCE');
});

test('refusal: CONFLICTING_FIGURES (same key, two rationals)', () => {
  const prog = GOOD.replace(
    'LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]',
    'LIK e0 POS GIVEN h0 4/5 ctx0 [30,40]\nLIK e0 POS GIVEN h0 3/5 ctx0 [40,50]'
  );
  const d = solve(prog, null);
  assert.equal(d.code, 'CONFLICTING_FIGURES');
});

test('refusal: CONFLICTING_FIGURES (stated posterior vs computed)', () => {
  const prog = GOOD.replace(
    'OBS e0 POS ctx0 [60,70]',
    'POSTERIOR_STATED h0 GIVEN e0 POS 1/2 ctx0 [55,60]\nOBS e0 POS ctx0 [60,70]'
  );
  const d = solve(prog, null);
  // stated 1/2, computed 2/3 -> conflict
  assert.equal(d.code, 'CONFLICTING_FIGURES');
});

test('refusal: CONTEXT_MISMATCH (premise only at other ctx)', () => {
  const prog = GOOD.replace('PRIOR h1 4/5 ctx0', 'PRIOR h1 4/5 ctx_pre');
  const d = solve(prog, null);
  assert.equal(d.code, 'CONTEXT_MISMATCH');
});

test('refusal: SCOPED_OR_RETRACTED_PREMISE (needed prior retracted)', () => {
  const prog = GOOD.replace(
    'OBS e0 POS ctx0',
    'NON_ASSERTED [10,20] RETRACTED\nOBS e0 POS ctx0'
  );
  const d = solve(prog, null);
  assert.equal(d.code, 'SCOPED_OR_RETRACTED_PREMISE');
  assert.equal(d.details.reason, 'RETRACTED');
});

test('refusal: INCOMPLETE_FACTORIZATION (LIK missing, no substitute)', () => {
  const prog = GOOD.replace('LIK e0 POS GIVEN h1 1/10 ctx0 [40,50]\n', '');
  const d = solve(prog, null);
  assert.equal(d.code, 'INCOMPLETE_FACTORIZATION');
});

test('receipt: deterministic, self-verifying, span-bound', () => {
  const X = 'Alpha account: ' + '0123456789'.repeat(8);
  const prog = parseBytecode(GOOD);
  const d = solve(prog, X);
  const r1 = buildReceipt({ X, program: prog, decision: d });
  const r2 = buildReceipt({ X, program: prog, decision: d });
  assert.equal(r1.receipt_sha256, r2.receipt_sha256);
  assert.ok(verifyReceipt(r1, { X, program: prog, decision: d }));
  // tampered text changes the seal
  const X2 = X.slice(0, 30) + 'X' + X.slice(31);
  assert.ok(!verifyReceipt(r1, { X: X2, program: prog, decision: d }));
  assert.equal(r1.schema, 'prism.decision_receipt.v1');
});
