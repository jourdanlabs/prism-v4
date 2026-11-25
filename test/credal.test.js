import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { simplexLP, vertexEnumerate, solveCredal, compilePolytope, farkasHolds } from '../src/kernel/credal.js';
import { generateCases } from '../scripts/crosscheck.mjs';
import { sentences, autoSentTags } from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import { toProgram, parseBytecode } from '../src/kernel/parser.js';
import { makeRng } from '../src/synth/rng.js';

// ---------- exact simplex vs naive vertex enumeration (ARCH-SPEC §2) ----------
// 100 random bounded polytopes: A entries in [1,4] (all-positive rows bound
// the feasible set), x0 entries in [1,3] -> b = A·x0 feasible by construction.
// Independent check: enumerate every basic feasible solution exactly and
// compare min/max of c·x against the simplex's optimum.
// Exact fraction-free row rank (Bareiss). vertexEnumerate needs full row rank:
// a rank-deficient-but-consistent system has no m-column basis at all.
function rowRank(A) {
  const M = A.map((r) => [...r]);
  const m = M.length;
  const n = M[0].length;
  let rank = 0;
  let det = 1n;
  for (let c = 0; c < n && rank < m; c++) {
    let r = -1;
    for (let i = rank; i < m; i++) if (M[i][c] !== 0n) { r = i; break; }
    if (r < 0) continue;
    [M[r], M[rank]] = [M[rank], M[r]];
    const piv = M[rank][c];
    for (let i = rank + 1; i < m; i++) {
      if (M[i][c] === 0n) continue;
      const t = M[i][c];
      for (let j = 0; j < n; j++) M[i][j] = (M[i][j] * piv - t * M[rank][j]) / det;
    }
    det = piv;
    rank++;
  }
  return rank;
}

test('credal: simplexLP == vertexEnumerate on 100 random polytopes', () => {
  const rng = makeRng(20261122);
  let checked = 0;
  for (let t = 0; t < 100; t++) {
    const m = rng.int(1, 3);
    const n = m + rng.int(1, 3);
    const A = [];
    for (let i = 0; i < m; i++) {
      A.push([...Array(n)].map(() => BigInt(rng.int(1, 4))));
    }
    if (rowRank(A) < m) { t--; continue; } // rank-deficient: no basis exists
    const x0 = [...Array(n)].map(() => BigInt(rng.int(1, 3)));
    const b = A.map((row) => row.reduce((s, a, j) => s + a * x0[j], 0n));
    const c = [...Array(n)].map(() => BigInt(rng.int(-4, 4)));

    const lpMax = simplexLP(A, b, c);
    const lpMin = simplexLP(A, b, c.map((v) => -v));
    const verts = vertexEnumerate(A, b, n);
    assert.ok(verts.length > 0, `trial ${t}: feasible by construction, no vertex found`);
    const obj = (x) => x.reduce((s, xj, j) => R.add(s, R.mul({ n: c[j], d: 1n }, xj)), R.zero());
    let vMax = null;
    let vMin = null;
    for (const v of verts) {
      const val = obj(v.x);
      if (vMax === null || R.cmp(val, vMax) > 0) vMax = val;
      if (vMin === null || R.cmp(val, vMin) < 0) vMin = val;
    }
    assert.equal(lpMax.status, 'optimal', `trial ${t}: max status ${lpMax.status}`);
    assert.equal(lpMin.status, 'optimal', `trial ${t}: min status ${lpMin.status}`);
    assert.ok(R.eq(lpMax.value, vMax), `trial ${t}: simplex max ${R.toString(lpMax.value)} != vertex max ${R.toString(vMax)}`);
    assert.ok(R.eq(R.neg(lpMin.value), vMin), `trial ${t}: simplex min ${R.toString(R.neg(lpMin.value))} != vertex min ${R.toString(vMin)}`);
    checked++;
  }
  assert.equal(checked, 100);
});

// ---------- infeasibility certificate ----------
test('credal: simplexLP reports infeasible on contradictory rows', () => {
  const A = [[1n, 1n], [1n, 1n]];
  const b = [1n, 3n];
  const r = simplexLP(A, b, null);
  assert.equal(r.status, 'infeasible');
  assert.ok(r.farkas && r.farkas.y.length === 2);
});

// Every infeasible decision in the 400-case crosscheck corpus must carry a
// Farkas certificate verified against the ORIGINAL compiled rows
// (y^T A >= 0 colwise, y^T b < 0, one coefficient per compiled row) —
// never reduced-system coordinates.
test('credal: every crosscheck infeasible decision carries an original-row Farkas cert', () => {
  const cases = generateCases(400);
  let infeasible = 0;
  for (const c of cases) {
    const prog = toProgram(c.program);
    const compiled = compilePolytope(prog, 'ctx0');
    const d = solveCredal(c.program, 'ctx0');
    if (d.status === 'REFUSED' && d.code === 'CONFLICTING_FIGURES') {
      infeasible++;
      const y = d.details?.farkas;
      assert.ok(
        Array.isArray(y),
        `case ${c.t}: CONFLICTING_FIGURES without farkas certificate`
      );
      const yRat = y.map((s) => R.parse(s));
      assert.equal(
        yRat.length,
        compiled.rows.length,
        `case ${c.t}: farkas len ${yRat.length} != original row count ${compiled.rows.length}`
      );
      const A = compiled.rows.map((r) => r.a);
      const b = compiled.rows.map((r) => r.b);
      assert.ok(
        farkasHolds(yRat, A, b),
        `case ${c.t}: farkas certificate fails y^T A >= 0 / y^T b < 0 on original rows`
      );
    }
  }
  assert.ok(infeasible > 150, `expected many infeasible cases, saw ${infeasible}`);
});

// ---------- world-space kernel ----------
const PROG = (body) => `HYP h0 [0,1]\nHYP h1 [0,1]\nPART_EXHAUSTIVE (h0 h1) [0,1]\n${body}\nEND\n`;

test('credal: baseline Bayes exact over atomic worlds', () => {
  const p = PROG(
    `EVENT e0 [0,1]\nPRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '8/9');
  assert.equal(d.via, 'CREDAL_LP');
});

test('credal: DET_LINK EQUIV collapses joint to single event', () => {
  const p = PROG(
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `LIK e1 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e1 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `DET_LINK e0 EQUIV e1 ctx0 [0,1]\nOBS e0 POS ctx0 [0,1]\nOBS e1 POS ctx0 [0,1]\n` +
    `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '8/9');
});

test('credal: unlinked events give exact Fréchet interval refusal', () => {
  const p = PROG(
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `LIK e1 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e1 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nOBS e1 POS ctx0 [0,1]\n` +
    `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNDERDETERMINED_INTERVAL');
  assert.equal(d.details.cause, 'UNLICENSED_INDEPENDENCE');
  // Fréchet: J0∈[3/5,4/5], J1∈[0,1/10] -> post∈[6/7,1]
  assert.equal(d.details.interval.min, '6/7');
  assert.equal(d.details.interval.max, '1/1');
});

test('credal: DET_LINK IMPLIES forces conditional to 1', () => {
  const p = PROG(
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `LIK e1 POS GIVEN h0 9/10 ctx0 [0,1]\nLIK e1 POS GIVEN h1 1/5 ctx0 [0,1]\n` +
    `DET_LINK e0 IMPLIES e1 ctx0 [0,1]\nOBS e0 POS ctx0 [0,1]\n` +
    `ASK_PROB e1 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '1/1');
});

test('credal: DET_LINK MUTEX + impossible joint evidence -> DEGENERATE_PARTITION', () => {
  // lik marginals 1/4+1/4 stay feasible under MUTEX (joint may be 0), so the
  // polytope is nonempty — but the observed e0∧e1 has zero support by domain.
  const p = PROG(
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 1/4 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/4 ctx0 [0,1]\n` +
    `LIK e1 POS GIVEN h0 1/4 ctx0 [0,1]\nLIK e1 POS GIVEN h1 1/4 ctx0 [0,1]\n` +
    `DET_LINK e0 MUTEX e1 ctx0 [0,1]\nOBS e0 POS ctx0 [0,1]\nOBS e1 POS ctx0 [0,1]\n` +
    `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'DEGENERATE_PARTITION');
});

test('credal: zero LIK support under all hypotheses -> DEGENERATE_PARTITION', () => {
  const p = PROG(
    `EVENT e0 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 0/1 ctx0 [0,1]\nLIK e0 POS GIVEN h1 0/1 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'DEGENERATE_PARTITION');
});

test('credal: zero-support-possible evidence -> ZERO_SUPPORT_POSSIBLE', () => {
  const p = PROG(
    `EVENT e0 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\n` + // h1's e0 likelihood unconstrained
    `LIK e0 POS GIVEN h0 0/1 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'ZERO_SUPPORT_POSSIBLE');
});

test('credal: COND_INDEP licenses joint factorization exactly', () => {
  const p = PROG(
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `LIK e1 POS GIVEN h0 7/10 ctx0 [0,1]\nLIK e1 POS GIVEN h1 3/5 ctx0 [0,1]\n` +
    `COND_INDEP (e0 e1) GIVEN * ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nOBS e1 POS ctx0 [0,1]\n` +
    `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'ANSWER');
  // (1/2·4/5·7/10) / (1/2·4/5·7/10 + 1/2·1/10·3/5) = 28/31
  assert.equal(R.toString(d.value), '28/31');
});

test('credal: JOINT_LIK pins the joint without independence', () => {
  const p = PROG(
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `JOINT_LIK (e0:POS e1:POS) GIVEN h0 3/10 ctx0 [0,1]\n` +
    `JOINT_LIK (e0:POS e1:POS) GIVEN h1 1/10 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nOBS e1 POS ctx0 [0,1]\n` +
    `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'ANSWER');
  // (1/2·3/10) / (1/2·3/10 + 1/2·1/10) = 3/4
  assert.equal(R.toString(d.value), '3/4');
});

test('credal: contradictory figures -> CONFLICTING_FIGURES', () => {
  const p = PROG(
    `EVENT e0 [0,1]\n` +
    `PRIOR h0 3/4 ctx0 [0,1]\nPRIOR h1 3/4 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'CONFLICTING_FIGURES');
});

test('credal: POSTERIOR_STATED consistent with factorization releases', () => {
  const p = PROG(
    `EVENT e0 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `POSTERIOR_STATED h0 GIVEN e0 POS 8/9 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '8/9');
});

test('credal: POSTERIOR_STATED contradicting factorization -> CONFLICTING_FIGURES', () => {
  const p = PROG(
    `EVENT e0 [0,1]\n` +
    `PRIOR h0 1/2 ctx0 [0,1]\nPRIOR h1 1/2 ctx0 [0,1]\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 [0,1]\nLIK e0 POS GIVEN h1 1/10 ctx0 [0,1]\n` +
    `POSTERIOR_STATED h0 GIVEN e0 POS 1/9 ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0`
  );
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'CONFLICTING_FIGURES');
});

test('credal: world budget enforced -> POLYTOPE_DIMENSION_LIMIT', () => {
  const evs = [...Array(6)].map((_, i) => `e${i}`);
  const p =
    `HYP h0 [0,1]\nHYP h1 [0,1]\nHYP h2 [0,1]\nHYP h3 [0,1]\nPART_EXHAUSTIVE (h0 h1 h2 h3) [0,1]\n` +
    evs.map((e) => `EVENT ${e} [0,1]`).join('\n') + '\n' +
    evs.map((e) => `LIK ${e} POS GIVEN h0 1/2 ctx0 [0,1]`).join('\n') + '\n' +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0\nEND\n`;
  const d = solveCredal(p);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'POLYTOPE_DIMENSION_LIMIT'); // 4·2^6 = 256 > 128
});

// ---------- full solve() path: DET_LINK licensed pair ----------
test('solve: DET_LINK EQUIV answers through full path (no UNLICENSED_INDEPENDENCE)', () => {
  const X =
    'Alpha accounts for 1/2 of cases, and Beta holds a 1/2 share. ' +
    'When the case is Alpha, the flag test is positive 4/5 of the time; under Beta, the flag test fires 1/10. ' +
    'When the case is Alpha, the lamp glows 4/5 of the time; under Beta, the lamp glows 1/10. ' +
    'The flag test and the lamp are wired to the same switch. ' +
    'On this case the flag test read positive. On this case the lamp read positive. ' +
    'What is the probability the case is Alpha?';
  // locate spans
  const at = (s) => { const i = X.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const p =
    `HYP h0 ${at('Alpha')}\nHYP h1 ${at('Beta')}\nPART_EXHAUSTIVE (h0 h1) ${at('of cases, and')}\n` +
    `EVENT e0 ${at('flag test')}\nEVENT e1 ${at('lamp')}\n` +
    `PRIOR h0 1/2 ctx0 ${at('1/2 of cases')}\nPRIOR h1 1/2 ctx0 ${at('1/2 share')}\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at('4/5 of the time')}\n` +
    `LIK e0 POS GIVEN h1 1/10 ctx0 ${at('fires 1/10')}\n` +
    `LIK e1 POS GIVEN h0 4/5 ctx0 ${at('glows 4/5')}\n` +
    `LIK e1 POS GIVEN h1 1/10 ctx0 ${at('glows 1/10')}\n` +
    `DET_LINK e0 EQUIV e1 ctx0 ${at('wired to the same switch')}\n` +
    `OBS e0 POS ctx0 ${at('the flag test read positive')}\n` +
    `OBS e1 POS ctx0 ${at('the lamp read positive')}\n`;
  // V2: the sentence-partition ledger is unconditional when X is supplied.
  const progObj = parseBytecode(p + `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0\nEND\n`);
  const ledger = autoSentTags(X, progObj)
    .map((t) => `SENT_TAG ${t.idx} ${t.tag} [${t.span.s},${t.span.e}]`)
    .join('\n');
  const full = p + ledger + `\nASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0\nEND\n`;
  const d = solve(full, X);
  assert.equal(d.status, 'ANSWER', d.code ?? '');
  assert.equal(R.toString(d.value), '8/9');
});

// ---------- SENT_TAG sentence-partition accounting ----------
const X2 =
  'Alpha accounts for 1/2 of cases, and Beta holds a 1/2 share. ' +
  'When the case is Alpha, the flag test is positive 4/5 of the time; under Beta, the flag test fires 1/10. ' +
  'On this case the flag test read positive. ' +
  'What is the probability the case is Alpha?';
const at2 = (s) => { const i = X2.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
const PROG2 =
  `HYP h0 ${at2('Alpha')}\nHYP h1 ${at2('Beta')}\nPART_EXHAUSTIVE (h0 h1) ${at2('of cases, and')}\n` +
  `EVENT e0 ${at2('flag test')}\n` +
  `PRIOR h0 1/2 ctx0 ${at2('1/2 of cases')}\nPRIOR h1 1/2 ctx0 ${at2('1/2 share')}\n` +
  `LIK e0 POS GIVEN h0 4/5 ctx0 ${at2('4/5 of the time')}\n` +
  `LIK e0 POS GIVEN h1 1/10 ctx0 ${at2('fires 1/10')}\n` +
  `OBS e0 POS ctx0 ${at2('the flag test read positive')}`;
const PROG2_ASK = `ASK_PROB h0 GIVEN (e0:POS) ctx0`;

const tagDecls = (X, overrides = {}, skip = new Set()) =>
  sentences(X)
    .filter((seg) => !skip.has(seg.idx))
    .map((seg) => {
      const tag = overrides[seg.idx] ?? (seg.text.includes('?') ? 'QUERY' : 'PREMISE');
      return `SENT_TAG ${seg.idx} ${tag} [${seg.s},${seg.e}]`;
    })
    .join('\n');

const prog2 = (extraDecls = '') =>
  `${PROG2}\n${extraDecls}${PROG2_ASK}\nEND\n`;

test('sent-account: full SENT_TAG cover passes and answers', () => {
  const d = solve(prog2(tagDecls(X2) + '\n'), X2);
  assert.equal(d.status, 'ANSWER', d.code ?? '');
  assert.equal(R.toString(d.value), '8/9');
});

test('sent-account: missing SENT_TAG index refuses', () => {
  const d = solve(prog2(tagDecls(X2, {}, new Set([2])) + '\n'), X2);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
});

test('sent-account: material sentence tagged FLUFF refuses', () => {
  const d = solve(prog2(tagDecls(X2, { 1: 'FLUFF' }) + '\n'), X2);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
});

test('sent-account: ungrounded deterministic-linkage sentence refuses', () => {
  const X3 =
    'Alpha accounts for 1/2 of cases, and Beta holds a 1/2 share. ' +
    'When the case is Alpha, the flag test is positive 4/5 of the time; under Beta, the flag test fires 1/10. ' +
    'The flag test and the lamp are wired to the same switch. ' +
    'On this case the flag test read positive. ' +
    'What is the probability the case is Alpha?';
  // spans must be located in X3 (the linkage sentence shifts later offsets)
  const at3 = (s) => { const i = X3.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const prog3 =
    `HYP h0 ${at3('Alpha')}\nHYP h1 ${at3('Beta')}\nPART_EXHAUSTIVE (h0 h1) ${at3('of cases, and')}\n` +
    `EVENT e0 ${at3('flag test')}\n` +
    `PRIOR h0 1/2 ctx0 ${at3('1/2 of cases')}\nPRIOR h1 1/2 ctx0 ${at3('1/2 share')}\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at3('4/5 of the time')}\n` +
    `LIK e0 POS GIVEN h1 1/10 ctx0 ${at3('fires 1/10')}\n` +
    `OBS e0 POS ctx0 ${at3('the flag test read positive')}\n` +
    `ASK_PROB h0 GIVEN (e0:POS) ctx0\nEND\n`;
  const d = solve(prog3, X3);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
});

// ---------- R3: DET_LINK polarity, context isolation, unique point ----------

// Bulma R2 repro: "The lamp and buzzer never activate together" is exclusion
// wording — it can license MUTEX but must NOT license EQUIV (R2 released
// P(buzzer|lamp)=1/1 off the negated span).
test('R3 det-link polarity: negated span refuses EQUIV, faithful MUTEX answers 0', () => {
  const X =
    'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
    'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
    'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
    'The lamp and buzzer never activate together. ' +
    'This panel was observed with its lamp lit. ' +
    'What is the probability its buzzer sounded?';
  const at = (s) => { const i = X.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const bc = (rel) =>
    `HYP h0 ${at('Ash')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}\n` +
    `EVENT e0 ${at('lamp')}\nEVENT e1 ${at('buzzer')}\n` +
    `PRIOR h0 2/5 ctx0 ${at('40% of panels')}\nPRIOR h1 3/5 ctx0 ${at('60%.')}\n` +
    `LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
    `LIK e1 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash panels and 20% of Fir panels. The lamp')}\n` +
    `LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir panels. The lamp')}\n` +
    `DET_LINK e0 ${rel} e1 ctx0 ${at('The lamp and buzzer never activate together')}\n` +
    `OBS e0 POS ctx0 ${at('was observed with its lamp lit')}\n` +
    sentences(X).map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`).join('\n') +
    `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`;
  const dEquiv = solve(bc('EQUIV'), X);
  assert.equal(dEquiv.status, 'REFUSED');
  assert.equal(dEquiv.code, 'UNLICENSED_INDEPENDENCE');
  const dMutex = solve(bc('MUTEX'), X);
  assert.equal(dMutex.status, 'ANSWER', dMutex.code ?? '');
  assert.equal(R.toString(dMutex.value), '0/1');
});

// IMPLIES declared on mutual-exclusion wording must refuse (the span asserts
// the OPPOSITE polarity of an implication).
test('R3 det-link polarity: IMPLIES refuses mutual-exclusion wording', () => {
  const X =
    'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
    'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
    'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
    'The lamp and buzzer never occur together. ' +
    'This panel was observed with its lamp lit. ' +
    'What is the probability its buzzer sounded?';
  const at = (s) => { const i = X.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const bc =
    `HYP h0 ${at('Ash')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}\n` +
    `EVENT e0 ${at('lamp')}\nEVENT e1 ${at('buzzer')}\n` +
    `PRIOR h0 2/5 ctx0 ${at('40% of panels')}\nPRIOR h1 3/5 ctx0 ${at('60%.')}\n` +
    `LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
    `LIK e1 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash panels and 20% of Fir panels. The lamp')}\n` +
    `LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir panels. The lamp')}\n` +
    `DET_LINK e0 IMPLIES e1 ctx0 ${at('The lamp and buzzer never occur together')}\n` +
    `OBS e0 POS ctx0 ${at('was observed with its lamp lit')}\n` +
    sentences(X).map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`).join('\n') +
    `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`;
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNLICENSED_INDEPENDENCE');
});

// MUTEX on affirmative-only co-occurrence wording ("always activate
// together") carries no exclusion claim — it must not license MUTEX.
test('R3 det-link polarity: MUTEX requires exclusion wording', () => {
  const X =
    'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
    'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
    'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
    'The lamp and buzzer always activate together. ' +
    'This panel was observed with its lamp lit. ' +
    'What is the probability its buzzer sounded?';
  const at = (s) => { const i = X.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const bc =
    `HYP h0 ${at('Ash')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}\n` +
    `EVENT e0 ${at('lamp')}\nEVENT e1 ${at('buzzer')}\n` +
    `PRIOR h0 2/5 ctx0 ${at('40% of panels')}\nPRIOR h1 3/5 ctx0 ${at('60%.')}\n` +
    `LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
    `LIK e1 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash panels and 20% of Fir panels. The lamp')}\n` +
    `LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir panels. The lamp')}\n` +
    `DET_LINK e0 MUTEX e1 ctx0 ${at('The lamp and buzzer always activate together')}\n` +
    `OBS e0 POS ctx0 ${at('was observed with its lamp lit')}\n` +
    sentences(X).map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`).join('\n') +
    `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`;
  const d = solve(bc, X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNLICENSED_INDEPENDENCE');
});

// Bulma R2 repro: a DET_LINK grounded in a disjoint population ("On the
// night-shift, ...") must not license a day-shift observation — the link
// obeys the same shift/site compatibility as LIK vs OBS/ASK_PROB.
test('R3 context: night-shift DET_LINK cannot license a day-shift observation', () => {
  const mk = (shift) =>
    'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
    'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
    'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
    `On the ${shift}-shift, the lamp and buzzer always activate together. ` +
    `A ${shift === 'night' ? 'day' : 'night'}-shift panel was observed with its lamp lit. ` +
    'What is the probability its buzzer sounded?';
  const build = (linkShift, obsShift) => {
    const X = mk(linkShift);
    const at = (s) => { const i = X.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
    const bc =
      `HYP h0 ${at('Ash')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}\n` +
      `EVENT e0 ${at('lamp')}\nEVENT e1 ${at('buzzer')}\n` +
      `PRIOR h0 2/5 ctx0 ${at('40% of panels')}\nPRIOR h1 3/5 ctx0 ${at('60%.')}\n` +
      `LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
      `LIK e1 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash panels and 20% of Fir panels. On')}\n` +
      `LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir panels. On')}\n` +
      `DET_LINK e0 EQUIV e1 ctx0 ${at('the lamp and buzzer always activate together')}\n` +
      `OBS e0 POS ctx0 ${at(`A ${obsShift}-shift panel was observed with its lamp lit`)}\n` +
      sentences(X).map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`).join('\n') +
      `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`;
    return [bc, X];
  };
  const [bcCross, xCross] = build('night', 'day');
  const dCross = solve(bcCross, xCross);
  assert.equal(dCross.status, 'REFUSED');
  assert.equal(dCross.code, 'CONTEXT_MISMATCH');
  // same-shift control still licenses and answers exactly
  const Xc = 'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
    'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
    'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
    'On the night-shift, the lamp and buzzer always activate together. ' +
    'A night-shift panel was observed with its lamp lit. ' +
    'What is the probability its buzzer sounded?';
  const atC = (s) => { const i = Xc.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const bcC =
    `HYP h0 ${atC('Ash')}\nHYP h1 ${atC('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${atC('Ash or Fir')}\n` +
    `EVENT e0 ${atC('lamp')}\nEVENT e1 ${atC('buzzer')}\n` +
    `PRIOR h0 2/5 ctx0 ${atC('40% of panels')}\nPRIOR h1 3/5 ctx0 ${atC('60%.')}\n` +
    `LIK e0 POS GIVEN h0 2/5 ctx0 ${atC('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${atC('20% of Fir')}\n` +
    `LIK e1 POS GIVEN h0 2/5 ctx0 ${atC('40% of Ash panels and 20% of Fir panels. On')}\n` +
    `LIK e1 POS GIVEN h1 1/5 ctx0 ${atC('20% of Fir panels. On')}\n` +
    `DET_LINK e0 EQUIV e1 ctx0 ${atC('the lamp and buzzer always activate together')}\n` +
    `OBS e0 POS ctx0 ${atC('A night-shift panel was observed with its lamp lit')}\n` +
    sentences(Xc).map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`).join('\n') +
    `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`;
  const dSame = solve(bcC, Xc);
  assert.equal(dSame.status, 'ANSWER', dSame.code ?? '');
  assert.equal(R.toString(dSame.value), '1/1');
});

// ---------- R4: universal entailment for deterministic links ----------
//
// Bulma R3 repro: "activate together on some panels, and each also activates
// by itself" released P(buzzer|lamp)=1/1 through DET_LINK EQUIV. Bare
// 'together' is existential in English — EQUIV is forall x: e0(x) <-> e1(x)
// and must require an explicit universal/physical-identity cue with ZERO
// existential, partial, modal or solo-occurrence qualifiers anywhere in the
// sentence. The same discipline gates IMPLIES, MUTEX and COND_INDEP.
const R4_X = (linkSentence) =>
  'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
  'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
  'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
  `${linkSentence} ` +
  'This panel was observed with its lamp lit. ' +
  'What is the probability its buzzer sounded?';

const spanOf = (X, s, from = 0) => {
  const i = X.indexOf(s, from);
  assert.ok(i >= 0, s);
  return `[${i},${i + s.length}]`;
};

const r4Program = (X, decl) => {
  const at = (s, from) => spanOf(X, s, from);
  const bz = X.indexOf('The buzzer sounds');
  assert.ok(bz >= 0);
  return (
    `HYP h0 ${at('Ash')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}\n` +
    `EVENT e0 ${at('lamp')}\nEVENT e1 ${at('buzzer')}\n` +
    `PRIOR h0 2/5 ctx0 ${at('40% of panels')}\nPRIOR h1 3/5 ctx0 ${at('60%.')}\n` +
    `LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
    `LIK e1 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash', bz)}\n` +
    `LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir', bz)}\n` +
    `${decl}\n` +
    `OBS e0 POS ctx0 ${at('was observed with its lamp lit')}\n` +
    sentences(X)
      .map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`)
      .join('\n') +
    `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`
  );
};

test('R4 universal entailment: partial/modal co-activation refuses EQUIV', () => {
  const traps = [
    // Bulma's exact R3 repro — released 1/1 in R3
    'The lamp and buzzer activate together on some panels, and each also activates by itself.',
    'The lamp and buzzer can activate together.',
    'The lamp and buzzer activate together on certain runs.',
    'The lamp and buzzer often activate together.',
    'The lamp and buzzer rarely activate together.',
    'The lamp and buzzer usually activate together.',
    'The lamp and buzzer activate together most of the time.',
    'The lamp and buzzer sometimes activate together.',
    'The lamp and buzzer tend to activate together.',
    'The lamp and buzzer activate together unless the relay is cut.',
    'The lamp and buzzer activate together, but either may activate alone.',
    'The lamp and buzzer activate together except on a few runs.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released ${d.value ? R.toString(d.value) : ''}`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R4 universal entailment: universal EQUIV controls still release 1/1', () => {
  const controls = [
    'The lamp and buzzer always activate together.',
    'The lamp and buzzer are wired to the same switch.',
    'The lamp and buzzer share a relay.',
    'The lamp and buzzer are mechanically coupled.',
    'The lamp and buzzer are locked together.',
    'The lamp activates if and only if the buzzer sounds.',
  ];
  for (const link of controls) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'ANSWER', `"${link}" refused ${d.code}`);
    assert.equal(R.toString(d.value), '1/1', `"${link}" -> ${R.toString(d.value)}`);
  }
});

test('R4 universal entailment: IMPLIES requires the same universality', () => {
  const Xa = R4_X('Whenever the lamp lights, the buzzer sounds.');
  const dOk = solve(
    r4Program(Xa, `DET_LINK e0 IMPLIES e1 ctx0 ${spanOf(Xa, 'Whenever the lamp lights, the buzzer sounds')}`),
    Xa
  );
  assert.equal(dOk.status, 'ANSWER', dOk.code ?? '');
  assert.equal(R.toString(dOk.value), '1/1');
  const traps = [
    'The lamp can trigger the buzzer.',
    'On some runs, whenever the lamp lights, the buzzer sounds.',
    'The lamp may force the buzzer to sound.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 IMPLIES e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R4 universal entailment: MUTEX requires universal exclusion', () => {
  const Xa = R4_X('The lamp and buzzer never activate together.');
  const dOk = solve(
    r4Program(Xa, `DET_LINK e0 MUTEX e1 ctx0 ${spanOf(Xa, 'The lamp and buzzer never activate together')}`),
    Xa
  );
  assert.equal(dOk.status, 'ANSWER', dOk.code ?? '');
  assert.equal(R.toString(dOk.value), '0/1');
  const traps = [
    'The lamp and buzzer usually do not activate together.',
    'The lamp and buzzer seldom activate together.',
    'The lamp and buzzer cannot both activate on some runs.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 MUTEX e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R4 universal entailment: COND_INDEP rejects partial independence', () => {
  const Xa = R4_X('The lamp and buzzer are conditionally independent within each panel type.');
  const dOk = solve(
    r4Program(
      Xa,
      `COND_INDEP (e0 e1) GIVEN * ctx0 ${spanOf(Xa, 'The lamp and buzzer are conditionally independent')}`
    ),
    Xa
  );
  assert.equal(dOk.status, 'ANSWER', dOk.code ?? '');
  // P(h0|e0)=4/7; P(e1|e0)=4/7*2/5+3/7*1/5=11/35
  assert.equal(R.toString(dOk.value), '11/35');
  const traps = [
    'The lamp and buzzer are partially independent within each panel type.',
    'The lamp and buzzer are roughly independent within each panel type.',
    'The lamp and buzzer are independent on some panels.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `COND_INDEP (e0 e1) GIVEN * ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

// Bulma R2 repro: a CONTEXT_SCOPE tag must bind a context the passage text
// actually grounds. Tagging an unlabeled sentence ("The room was quiet.")
// with an invented or misbound context refuses; a real grounded label passes.
test('R3 context: CONTEXT_SCOPE requires text-grounded context', () => {
  const X =
    'Every capsule is either Elm or Fir. Elm accounts for 40% of capsules and Fir accounts for 60%. ' +
    'The lamp lights for 80% of Elm capsules and 20% of Fir capsules. ' +
    'Among lamp-lit capsules, 80% are Elm. ' +
    'This capsule was observed with its lamp lit. ' +
    'The room was quiet. ' +
    'What is the probability it is Elm?';
  const at = (s) => { const i = X.indexOf(s); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
  const segs = sentences(X);
  const tags = (scopeTag) =>
    segs
      .map((s) => {
        const tag = s.text.includes('?') ? 'QUERY' : s.text.includes('quiet') ? scopeTag : 'PREMISE';
        return `SENT_TAG ${s.idx} ${tag} [${s.s},${s.e}]`;
      })
      .join('\n');
  const bc = (scopeTag) =>
    `HYP h0 ${at('Elm')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Elm or Fir')}\n` +
    `EVENT e0 ${at('lamp')}\n` +
    `PRIOR h0 2/5 ctx0 ${at('40% of capsules')}\nPRIOR h1 3/5 ctx0 ${at('60%')}\n` +
    `LIK e0 POS GIVEN h0 4/5 ctx0 ${at('80% of Elm')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
    `POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 ${at('80% are Elm')}\n` +
    `OBS e0 POS ctx0 ${at('observed with its lamp lit')}\n` +
    tags(scopeTag) +
    `\nASK_PROB h0 GIVEN (e0:POS) ctx0\nEND\n`;
  // invented context id — not declared anywhere
  const dGhost = solve(bc('CONTEXT_SCOPE ctx_ghost'), X);
  assert.equal(dGhost.status, 'REFUSED');
  assert.equal(dGhost.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
  // a declared ctx still cannot bind an unlabeled sentence
  const dUnlabeled = solve(bc('CONTEXT_SCOPE ctx0'), X);
  assert.equal(dUnlabeled.status, 'REFUSED');
  assert.equal(dUnlabeled.code, 'UNACCOUNTED_MATERIAL_SENTENCE');
});

// Unique-point fast path: 4-way COND_INDEP pins all 32 masses (rank 32 over
// 32 worlds) — the answer comes straight from w*, no simplex pivots. A
// contradictory variant leaves w* with negative mass and must refuse with a
// verified original-row Farkas certificate.
test('R3 unique point: full-rank 32-world system answers exactly', () => {
  const mk32 = (p0, p1, l0, l1) =>
    `HYP h0 [0,1]\nHYP h1 [0,1]\nPART_EXHAUSTIVE (h0 h1) [0,1]\n` +
    `EVENT e0 [0,1]\nEVENT e1 [0,1]\nEVENT e2 [0,1]\nEVENT e3 [0,1]\n` +
    `PRIOR h0 ${p0} ctx0 [0,1]\nPRIOR h1 ${p1} ctx0 [0,1]\n` +
    l0.map((v, i) => `LIK e${i} POS GIVEN h0 ${v} ctx0 [0,1]`).join('\n') + '\n' +
    l1.map((v, i) => `LIK e${i} POS GIVEN h1 ${v} ctx0 [0,1]`).join('\n') + '\n' +
    `COND_INDEP (e0 e1 e2 e3) GIVEN * ctx0 [0,1]\n` +
    `OBS e0 POS ctx0 [0,1]\nASK_PROB h0 GIVEN (e0:POS) ctx0\nEND`;
  const small = mk32('2/5', '3/5', ['1/2', '2/3', '3/4', '4/5'], ['1/2', '2/3', '3/4', '4/5']);
  const d = solveCredal(small, 'ctx0');
  assert.equal(d.status, 'ANSWER');
  assert.equal(R.toString(d.value), '2/5');
  const dist = mk32('5/11', '6/11', ['5/7', '2/3', '7/13', '11/17'], ['3/7', '3/5', '5/19', '13/23']);
  const d2 = solveCredal(dist, 'ctx0');
  assert.equal(d2.status, 'ANSWER');
  assert.equal(R.toString(d2.value), '25/43');
  // negative unique point: JOINT_LIK exceeding its own marginals — the
  // linear system is full-rank consistent, but w* has a negative coordinate
  const bad =
    `HYP h0 [0,1]\nPART_EXHAUSTIVE (h0) [0,1]\nEVENT e0 [0,1]\nEVENT e1 [0,1]\n` +
    `LIK e0 POS GIVEN h0 1/4 ctx0 [0,1]\nLIK e1 POS GIVEN h0 1/4 ctx0 [0,1]\n` +
    `JOINT_LIK (e0:POS e1:POS) GIVEN h0 1/2 ctx0 [0,1]\n` +
    `ASK_PROB e0 GIVEN () ctx0\nEND`;
  const d3 = solveCredal(bad, 'ctx0');
  assert.equal(d3.status, 'REFUSED');
  assert.equal(d3.code, 'CONFLICTING_FIGURES');
  const prog = toProgram(bad);
  const compiled = compilePolytope(prog, 'ctx0');
  const y = d3.details.farkas.map((s) => R.parse(s));
  assert.equal(y.length, compiled.rows.length);
  assert.ok(farkasHolds(y, compiled.rows.map((r) => r.a), compiled.rows.map((r) => r.b)));
});

// ---------- R5: canonical-license structural invariants ----------
//
// Bulma R4 gate: substring cue + finite deny-list can ALWAYS be wrapped in
// an outer modal/subset/hypothesis clause — "On a subset of panels, the lamp
// and buzzer always activate together" released 1/1 because 'always' sat
// anywhere in the sentence. R5 replaces it with two structural invariants:
//   1. the WHOLE sentence containing the license span must match an anchored
//      ^...$ canonical universal form — an optional declared-population
//      scope phrase, the universal core, and zero other clauses/modifiers;
//   2. a population-wide license naming a strict subset of the exhaustive
//      hypotheses is hypothesis-scoped and cannot license the queried
//      population.
test('R5 structural license: subset/existential/modal outer frames refuse EQUIV', () => {
  const traps = [
    // Bulma's four controlled wrong-release cases (verbatim)
    'On a subset of panels, the lamp and buzzer always activate together, and elsewhere both solo patterns occur.',
    'At least one panel has a lamp and buzzer that are wired to the same switch.',
    'It is possible that the lamp and buzzer are wired to the same switch on every panel.',
    'The lamp and buzzer always activate together on Ash panels.',
    // adversarial outer frames — a substring+denylist approach cannot
    // enumerate these; an anchored template rejects them structurally
    'In general, the lamp and buzzer always activate together.',
    'Unless bypassed, the lamp and buzzer always activate together.',
    'It is believed that the lamp and buzzer always activate together.',
    'Nearly every panel has the lamp and buzzer wired to the same switch.',
    'On a subset of panels, the lamp and buzzer always activate together.',
    'On most runs, the lamp and buzzer always activate together.',
    'The lamp and buzzer always activate together on some panels.',
    'The lamp and buzzer always activate together, most of the time.',
    'The lamp and buzzer always activate together; the siren sometimes joins them.',
    'It is possible that the lamp and buzzer always activate together.',
    'The lamp and buzzer might always activate together.',
    'The lamp and buzzer always activate together on certain panels.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released ${d.value ? R.toString(d.value) : ''}`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R5 hypothesis restriction: licenses naming a strict hyp subset refuse', () => {
  const traps = [
    'The lamp and buzzer always activate together on Ash panels.',
    'On Ash panels, the lamp and buzzer always activate together.',
    'The lamp and buzzer are wired to the same switch on Ash panels.',
    'The lamp and buzzer share a relay on Ash panels.',
    'On Ash panels the lamp and buzzer always activate together.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
  // same restriction applied to COND_INDEP
  const Xi = R4_X('The lamp and buzzer are independent on Ash panels.');
  const di = solve(
    r4Program(Xi, `COND_INDEP (e0 e1) GIVEN * ctx0 ${spanOf(Xi, 'The lamp and buzzer are independent on Ash panels')}`),
    Xi
  );
  assert.equal(di.status, 'REFUSED');
  assert.equal(di.code, 'UNLICENSED_INDEPENDENCE');
});

test('R5 canonical forms: every-strata scopes and declared ctx still license', () => {
  const controls = [
    // the affirmative control — bare universal
    ['The lamp and buzzer always activate together.', 'EQUIV'],
    // physical identity + explicit every-panel quantification
    ['The lamp and buzzer are wired to the same switch on every panel.', 'EQUIV'],
    // a license naming ALL exhaustive hypotheses is population-wide, not scoped
    ['The lamp and buzzer always activate together on both Ash and Fir panels.', 'EQUIV'],
    ['The lamp and buzzer never activate together.', 'MUTEX'],
  ];
  for (const [link, rel] of controls) {
    const X = R4_X(link);
    const want = rel === 'MUTEX' ? '0/1' : '1/1';
    const d = solve(r4Program(X, `DET_LINK e0 ${rel} e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'ANSWER', `"${link}" refused ${d.code}`);
    assert.equal(R.toString(d.value), want, `"${link}" -> ${R.toString(d.value)}`);
  }
});

// ---------- R6: exact population-noun grounding for license scopes ----------
//
// Bulma R5 gate: the every/all quantifier head in licenseScopeOk accepted ANY
// trailing noun phrase — "on every panel with a common relay", "on all intact
// panels", "on Ash and Fir panels fitted with a common relay", "on every
// panel except damaged panels" all released 1/1 through both entry points,
// and the possessive core "Every <anyword>'s" never grounded the noun to the
// queried population ("Every prototype's ..."). R6 removes every wildcard:
// a quantified scope's noun must be EXACTLY the queried population noun
// (optionally ctx- or hypothesis-prefixed), and the possessive subject is
// held to the same grounding. Anything else refuses UNLICENSED_INDEPENDENCE.
test('R6 scope grounding: qualified/exception/restricted universals refuse', () => {
  const traps = [
    // Bulma's five verbatim R5 wrong-release cases
    'On every panel with a common relay, the lamp and buzzer always activate together.',
    'The lamp and buzzer always activate together on all intact panels.',
    'On Ash and Fir panels fitted with a common relay, the lamp and buzzer always activate together.',
    'On every panel except damaged panels, the lamp and buzzer always activate together.',
    // unseen variants — adjectives, without-clauses, relative clauses,
    // unless/having/fitted/where restrictions, foreign head nouns
    'On every standard panel, the lamp and buzzer always activate together.',
    'The lamp and buzzer always activate together on all panels without defects.',
    'On every panel that passes inspection, the lamp and buzzer always activate together.',
    'On each undamaged panel, the lamp and buzzer always activate together.',
    'For all undamaged panels, the lamp and buzzer always activate together.',
    'Across every panel having a common relay, the lamp and buzzer always activate together.',
    'On every panel unless bypassed, the lamp and buzzer always activate together.',
    'On every panel fitted with a common relay, the lamp and buzzer always activate together.',
    'On every panel where the relay is shared, the lamp and buzzer always activate together.',
    'Within every panel in service, the lamp and buzzer always activate together.',
    'During every panel inspection, the lamp and buzzer always activate together.',
    'Under all working panels, the lamp and buzzer always activate together.',
    'On every prototype, the lamp and buzzer always activate together.',
    'The lamp and buzzer always activate together on every single panel that exists.',
    'The lamp and buzzer always activate together on Ash and Fir panels with a common relay.',
    'On both Ash and Fir panels fitted with a common relay, the lamp and buzzer always activate together.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released ${d.value ? R.toString(d.value) : ''}`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R6 possessive grounding: ungrounded possessive subjects refuse', () => {
  const traps = [
    // Bulma's verbatim case — 'prototype' is not the queried population
    "Every prototype's lamp and buzzer are wired to the same switch.",
    // unseen possessive variants — restricted or foreign subjects
    "Each tested unit's lamp and buzzer are wired to the same switch.",
    "Every intact panel's lamp and buzzer are wired to the same switch.",
    "Every panel with a relay's lamp and buzzer are wired to the same switch.",
    "Each undamaged item's lamp and buzzer are wired to the same switch.",
    "Every prototype's lamp and buzzer always activate together.",
    "Every sample's lamp and buzzer are wired to the same switch.",
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R6 canonical scopes: exact population forms still license', () => {
  const controls = [
    // bare universal — no scope phrase at all
    ['The lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    // possessive control — the population noun grounded to the query
    ["Every panel's lamp and buzzer are wired to the same switch.", 'EQUIV', '1/1'],
    ["Every panels' lamp and buzzer are wired to the same switch.", 'EQUIV', '1/1'],
    // quantified bare population noun — lead and trailing scope positions
    ['On every panel, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    ['The lamp and buzzer always activate together on every panel.', 'EQUIV', '1/1'],
    ['On all panels, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    ['For all panels, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    ['Across every panel, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    ['On each panel, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    // complete-hypothesis partition + population noun
    ['On both Ash and Fir panels, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    ['The lamp and buzzer always activate together on both Ash and Fir panels.', 'EQUIV', '1/1'],
    ['On every Ash and Fir panel, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    // stratum-universal independence phrasing still licenses COND_INDEP
    ['The lamp and buzzer are conditionally independent within each panel type.', 'COND_INDEP', '11/35'],
    // MUTEX control
    ['The lamp and buzzer never activate together.', 'MUTEX', '0/1'],
  ];
  for (const [link, rel, want] of controls) {
    const X = R4_X(link);
    const decl =
      rel === 'COND_INDEP'
        ? `COND_INDEP (e0 e1) GIVEN * ctx0 ${spanOf(X, 'are conditionally independent')}`
        : `DET_LINK e0 ${rel} e1 ctx0 ${spanOf(X, link)}`;
    const d = solve(r4Program(X, decl), X);
    assert.equal(d.status, 'ANSWER', `"${link}" refused ${d.code}`);
    assert.equal(R.toString(d.value), want, `"${link}" -> ${R.toString(d.value)}`);
  }
});

// ---------- R7: self-grounding closures for license scopes ----------
//
// Bulma R6 gate: three exact line-level holes released 1/1 through both
// entry points:
//   1. the shared-device NP in "wired to the same <dev>" was a MULTI-WORD
//      wildcard — "switch except during testing" parsed as one device noun
//      phrase, so the exception never reached the anchored template;
//   2. context labels for quantified scope modifiers were collected from
//      the license sentence itself — "On every Refurbished panel, ..."
//      self-grounded 'refurbished' with no other sentence naming it;
//   3. the possessive "Every <N>'s" accepted bare stratum classifiers
//      ("clinic", "crew") as partition dimensions even though the passage
//      never mentions them — a foreign domain self-licensed.
// R7 closes all three: device nouns are single-word, ctx labels ground
// only from the OBS/QUERY sentences, and a possessive subject must be the
// queried population noun (stratum classifiers stay scope-only — and
// concrete ones must appear outside the license sentence).
test('R7 device NP: exception/time tails refuse EQUIV', () => {
  const traps = [
    // Bulma's verbatim R6 blocker
    'The lamp and buzzer are wired to the same switch except during testing.',
    // unseen variants — exception/time clauses after the device noun
    'The lamp and buzzer are wired to the same switch outside maintenance.',
    'The lamp and buzzer are wired to the same switch on weekdays.',
    'The lamp and buzzer are wired to the same switch unless the relay is cut.',
    'The lamp and buzzer are wired to the same switch during testing.',
    'The lamp and buzzer share a relay except on damaged panels.',
    'The lamp and buzzer are connected to the same circuit under load.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released ${d.value ? R.toString(d.value) : ''}`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R7 device NP: single-word device controls still release 1/1', () => {
  const controls = [
    'The lamp and buzzer are wired to the same switch.',
    'The lamp and buzzer are wired to the same bus.',
    'The lamp and buzzer are wired to the same power-source.',
    'The lamp and buzzer share a relay.',
    'The lamp and buzzer share the same power-source.',
    'The lamp and buzzer are wired to the same switch on every panel.',
  ];
  for (const link of controls) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'ANSWER', `"${link}" refused ${d.code}`);
    assert.equal(R.toString(d.value), '1/1', `"${link}" -> ${R.toString(d.value)}`);
  }
});

test('R7 self-grounded ctx: capitalized qualifier cannot invent its own context', () => {
  const traps = [
    // Bulma's verbatim R6 blocker — 'Refurbished' appears nowhere else
    'On every Refurbished panel, the lamp and buzzer always activate together.',
    // unseen variants — same self-grounding hole
    'On every Certified panel, the lamp and buzzer always activate together.',
    'On every Inspected panel, the lamp and buzzer always activate together.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R7 foreign population nouns: ungrounded classifiers refuse', () => {
  const traps = [
    // Bulma's verbatim R6 blockers — possessive subjects are owners, not
    // partition dimensions; a noun the passage never mentions fails closed
    "Every clinic's lamp and buzzer are wired to the same switch.",
    "Every crew's lamp and buzzer are wired to the same switch.",
    // unseen variants — possessives AND bare scope heads
    "Every hospital's lamp and buzzer are wired to the same switch.",
    "Every cohort's lamp and buzzer are wired to the same switch.",
    'On every clinic, the lamp and buzzer always activate together.',
    'For all crews, the lamp and buzzer always activate together.',
    'On each ward, the lamp and buzzer always activate together.',
  ];
  for (const link of traps) {
    const X = R4_X(link);
    const d = solve(r4Program(X, `DET_LINK e0 EQUIV e1 ctx0 ${spanOf(X, link)}`), X);
    assert.equal(d.status, 'REFUSED', `"${link}" released`);
    assert.equal(d.code, 'UNLICENSED_INDEPENDENCE', `"${link}" refused ${d.code}`);
  }
});

test('R7 generic stratum scopes still license', () => {
  const controls = [
    // partition-dimension vocabulary is universal by construction
    ['On both strata, the lamp and buzzer always activate together.', 'EQUIV', '1/1'],
    ['Within either cohort, the lamp and buzzer are conditionally independent.', 'COND_INDEP', '11/35'],
    ['For each cohort, the lamp and buzzer are independent.', 'COND_INDEP', '11/35'],
    ['Within either lane, the lamp and buzzer are independent.', 'COND_INDEP', '11/35'],
    ['The lamp and buzzer are conditionally independent within each panel type.', 'COND_INDEP', '11/35'],
  ];
  for (const [link, rel, want] of controls) {
    const X = R4_X(link);
    const decl =
      rel === 'COND_INDEP'
        ? `COND_INDEP (e0 e1) GIVEN * ctx0 ${spanOf(X, link)}`
        : `DET_LINK e0 ${rel} e1 ctx0 ${spanOf(X, link)}`;
    const d = solve(r4Program(X, decl), X);
    assert.equal(d.status, 'ANSWER', `"${link}" refused ${d.code}`);
    assert.equal(R.toString(d.value), want, `"${link}" -> ${R.toString(d.value)}`);
  }
});

test('R7 restricted license cannot serve an unqualified observation', () => {
  // "On night-shift panels, ..." is a well-formed ctx scope, so the license
  // stays syntactically valid — but a context-restricted rule may never
  // apply to an OBS/QUERY that binds no population.
  const mk = (linkS, obsS) =>
    'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
    'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
    'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
    `${linkS} ` +
    `${obsS} ` +
    'What is the probability its buzzer sounded?';
  const build = (linkS, obsS) => {
    const X = mk(linkS, obsS);
    const at = (s, from = 0) => { const i = X.indexOf(s, from); assert.ok(i >= 0, s); return `[${i},${i + s.length}]`; };
    const bz = X.indexOf('The buzzer sounds');
    const bc =
      `HYP h0 ${at('Ash')}\nHYP h1 ${at('Fir')}\nPART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}\n` +
      `EVENT e0 ${at('lamp')}\nEVENT e1 ${at('buzzer')}\n` +
      `PRIOR h0 2/5 ctx0 ${at('40% of panels')}\nPRIOR h1 3/5 ctx0 ${at('60%.')}\n` +
      `LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}\nLIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}\n` +
      `LIK e1 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash', bz)}\n` +
      `LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir', bz)}\n` +
      `DET_LINK e0 EQUIV e1 ctx0 ${at(linkS.replace(/\.$/, ''))}\n` +
      `OBS e0 POS ctx0 ${at(obsS.replace(/\.$/, ''))}\n` +
      sentences(X).map((s) => `SENT_TAG ${s.idx} ${s.text.includes('?') ? 'QUERY' : 'PREMISE'} [${s.s},${s.e}]`).join('\n') +
      `\nASK_PROB e1 GIVEN (e0:POS) ctx0\nEND\n`;
    return [bc, X];
  };
  // restricted link + unqualified observation -> CONTEXT_MISMATCH
  const [bcBad, xBad] = build(
    'On night-shift panels, the lamp and buzzer always activate together.',
    'This panel was observed with its lamp lit.'
  );
  const dBad = solve(bcBad, xBad);
  assert.equal(dBad.status, 'REFUSED');
  assert.equal(dBad.code, 'CONTEXT_MISMATCH');
  // same-shift control: link AND observation both on night-shift -> 1/1
  const [bcOk, xOk] = build(
    'On night-shift panels, the lamp and buzzer always activate together.',
    'A night-shift panel was observed with its lamp lit.'
  );
  const dOk = solve(bcOk, xOk);
  assert.equal(dOk.status, 'ANSWER', dOk.code ?? '');
  assert.equal(R.toString(dOk.value), '1/1');
});
