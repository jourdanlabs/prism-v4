// prism-credal :: exact-rational credal polytope kernel (PRISM V2 ARCH-SPEC §2).
//
// Replaces the imperative Star-Bayes calculator with a mathematically
// complete exact-rational linear-constraint solver over finite atomic worlds.
// Every bytecode declaration compiles mechanically into exact rational
// linear equations or domain restrictions over w ∈ Q^|Ω|, Ω = H × {0,1}^k
// (m·2^k ≤ 128 worlds). The query P(Target | Evidence) is bounded by a
// Charnes-Cooper fractional linear program — substituted u = w/E(w), t =
// 1/E(w) turns N(w)/E(w) into a pure exact-rational LP. When the interval
// [p_min, p_max] collapses to a point the answer is mathematically entailed
// and released exactly; when it does not, the premises are underdetermined
// and the kernel refuses with tight Fréchet bounds. No silent independence
// assumption is possible anywhere in this file.
//
// All arithmetic is exact BigInt. The simplex tableau stores each row as
// integers with a per-row denominator (actual coeff = T[i][j]/den[i]) and
// gcd-reduces after every pivot — fraction-free by construction.

import * as R from './rational.js';
import { toProgram } from './parser.js';
import { checkSentenceAccounting } from './sentences.js';

export const CREDAL_CODES = [
  'CONFLICTING_FIGURES',
  'DEGENERATE_PARTITION',
  'ZERO_SUPPORT_POSSIBLE',
  'UNDERDETERMINED_INTERVAL',
  'POLYTOPE_DIMENSION_LIMIT',
];

const MAX_WORLDS = 128;
const MAX_PIVOTS = 20000;

// --------------------------------------------------------------------------
// exact simplex:  maximize c·x  s.t.  A x = b, x >= 0
// A: BigInt[m][n], b: BigInt[m] (integer rows — callers pre-scale rationals).
// Tableau has m+1 rows (m constraints + objective) and n+m+1 columns
// (n structural vars, m artificials, rhs). Invariants:
//   - actual coefficient of row i, col j is T[i][j]/den[i]
//   - basic columns: x_{basis[i]} = T[i][rhs]/T[i][basis[i]] >= 0 (primal)
//   - T[i][basis[i]] > 0 and den[i] > 0 throughout
// --------------------------------------------------------------------------
function gcdRow(row, den) {
  let g = den < 0n ? -den : den;
  for (let j = 0; j < row.length; j++) {
    const v = row[j] < 0n ? -row[j] : row[j];
    g = R.gcd(g, v);
    if (g === 1n) return 1n;
  }
  return g;
}

function pivot(T, den, r, c) {
  const p = T[r][c];
  const rr = T[r];
  for (let i = 0; i < T.length; i++) {
    if (i === r) continue;
    const tc = T[i][c];
    if (tc === 0n) continue;
    const ri = T[i];
    for (let j = 0; j < ri.length; j++) {
      if (ri[j] === 0n && rr[j] === 0n) continue;
      ri[j] = ri[j] * p - tc * rr[j];
    }
    den[i] *= p;
    // normalize: every row denominator stays positive. Sign tests elsewhere
    // (ratio test, reduced-cost scan, primal feasibility) read the stored
    // numerators directly and assume den[i] > 0 — a negative pivot element
    // (basis canonicalization, artificial purge) must not corrupt them.
    if (den[i] < 0n) {
      for (let j = 0; j < ri.length; j++) ri[j] = -ri[j];
      den[i] = -den[i];
    }
    const g = gcdRow(ri, den[i]);
    if (g > 1n) {
      for (let j = 0; j < ri.length; j++) ri[j] /= g;
      den[i] /= g;
    }
  }
}

// Express the objective "maximize c·x" as a stored tableau row in the current
// basis: stored[j] = -r_j·L with r_j the reduced cost and L = den[objRow] the
// lcm of basic coefficients, so entering candidates are the j with T[obj][j]<0.
function installObjective(T, den, basis, cvec, objRow) {
  const m = T.length - 1;
  const W = T[0].length;
  let L = 1n;
  for (let i = 0; i < m; i++) L = (L / R.gcd(L, T[i][basis[i]])) * T[i][basis[i]];
  for (let j = 0; j < W; j++) {
    const cj = j < cvec.length ? cvec[j] : 0n;
    let num = -cj * L;
    for (let i = 0; i < m; i++) {
      const cb = basis[i] < cvec.length ? cvec[basis[i]] : 0n;
      if (cb !== 0n) num += cb * T[i][j] * (L / T[i][basis[i]]);
    }
    T[objRow][j] = num;
  }
  den[objRow] = L;
}

// Canonicalize the tableau w.r.t. a hinted basis: pivot each target column
// into any unused row with a nonzero entry (the basis columns are linearly
// independent, so an entry exists in some unused row at each step), then
// check primal feasibility. Used for warm starts and for re-canonicalizing
// after a Dantzig cycle is detected.
function tryCanonical(T, den, basis, n, m, rhsCol, target) {
  if (new Set(target).size !== target.length) return false;
  const used = new Array(m).fill(false);
  for (const col of target) {
    if (col >= n) return false;
    let r = -1;
    for (let i = 0; i < m; i++) {
      if (!used[i] && T[i][col] !== 0n) { r = i; break; }
    }
    if (r < 0) return false;
    pivot(T, den, r, col);
    basis[r] = col;
    used[r] = true;
    if (T[r][col] < 0n) {
      for (let j = 0; j < T[r].length; j++) T[r][j] = -T[r][j];
    }
  }
  for (let i = 0; i < m; i++) {
    if (T[i][rhsCol] < 0n) return false;
  }
  return true;
}

// Run pivots on `allow(j)`-eligible entering columns.
// rule 'bland': first negative reduced cost (provably finite).
// rule 'dantzig': most negative reduced cost (fewer pivots in practice);
//   basis-signature cycling is detected and reported as 'cycle' so the caller
//   can fall back to Bland on the same phase.
function simplexLoop(T, den, basis, objRow, rhsCol, allow, rule = 'bland') {
  const m = T.length - 1;
  const seen = rule === 'dantzig' ? new Set() : null;
  let iters = 0;
  for (;;) {
    if (++iters > MAX_PIVOTS) throw new Error('SIMPLEX_ITERATION_LIMIT');
    let c = -1;
    if (rule === 'dantzig') {
      let best = 0n;
      for (let j = 0; j < rhsCol; j++) {
        if (!allow(j)) continue;
        const v = T[objRow][j];
        if (v < best) {
          best = v;
          c = j;
        }
      }
    } else {
      for (let j = 0; j < rhsCol; j++) {
        if (!allow(j)) continue;
        if (T[objRow][j] < 0n) {
          c = j;
          break;
        }
      }
    }
    if (c < 0) return 'optimal';
    // ratio test: rows with T[i][c]/den[i] > 0 bound the entering variable
    let r = -1;
    for (let i = 0; i < m; i++) {
      const tic = T[i][c];
      if (tic <= 0n) continue;
      if (r < 0) {
        r = i;
        continue;
      }
      // ratio = T[i][rhs]/T[i][c] vs T[r][rhs]/T[r][c] — raw cross-multiply
      // (positive terms; per-row den cancels within each ratio), ties broken
      // by smallest basic var (Bland).
      const lhs = T[i][rhsCol] * T[r][c];
      const rhs = T[r][rhsCol] * tic;
      const cmp = lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
      if (cmp < 0 || (cmp === 0 && basis[i] < basis[r])) r = i;
    }
    if (r < 0) return 'unbounded';
    pivot(T, den, r, c);
    basis[r] = c;
    if (seen) {
      const key = [...basis].sort((a, b2) => a - b2).join('|');
      if (seen.has(key)) return 'cycle';
      seen.add(key);
    }
  }
}

/**
 * simplexLP(A, b, c, warmBasis?) — maximize c·x over {x>=0 : Ax=b}.
 * A: BigInt rows, b: BigInt, c: BigInt coeffs (use c=null for feasibility only).
 * warmBasis: a feasible basis from a previous LP over the same (A,b) — pivoted
 *   in directly to skip phase 1. Falls back to phase 1 if it fails to
 *   canonicalize or is primal-infeasible.
 * Returns:
 *   {status:'optimal', value:{n,d}, x:[{n,d}], basis:[colIdx]}
 *   {status:'infeasible', farkas:{y:[{n,d}], check:{lhsNonneg,rhsNeg}}}
 *   {status:'unbounded'}
 */
export function simplexLP(A, b, c, warmBasis = null) {
  const m = A.length;
  const n = A[0].length;
  const W = n + m + 1; // structural + artificials + rhs
  let rhsCol = W - 1;
  const objRow = m;
  const T = [];
  const den = new Array(m + 1).fill(1n);
  const basis = new Array(m);
  // sign each constraint row was multiplied by to make its rhs nonnegative —
  // Farkas duals extracted from the tableau refer to the SCALED system and
  // must be multiplied back to certify against the caller's (A,b)
  const rowSign = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = new Array(W).fill(0n);
    const bi = b[i] < 0n ? -1n : 1n;
    rowSign[i] = bi;
    for (let j = 0; j < n; j++) row[j] = A[i][j] * bi;
    row[n + i] = 1n; // artificial, coeff +1 (den[i] = 1)
    row[rhsCol] = b[i] * bi;
    T.push(row);
    basis[i] = n + i;
  }
  T.push(new Array(W).fill(0n));

  // ---- optional warm start: pivot each hinted structural column into its
  // row. After m pivots the tableau is canonical w.r.t. the hint; if every
  // rhs then reads nonnegative the hint was a feasible basis and phase 1 is
  // unnecessary. Any failure -> rebuild and run the normal phase 1.
  let warmed = false;
  if (warmBasis && warmBasis.length === m && warmBasis.every((j) => j < n)) {
    warmed = tryCanonical(T, den, basis, n, m, rhsCol, warmBasis);
    if (!warmed) {
      // hint failed — rebuild the tableau and run the standard phase 1
      for (let i = 0; i < m; i++) {
        const row = T[i];
        row.fill(0n);
        const bi = b[i] < 0n ? -1n : 1n;
        rowSign[i] = bi;
        for (let j = 0; j < n; j++) row[j] = A[i][j] * bi;
        row[n + i] = 1n;
        row[rhsCol] = b[i] * bi;
        den[i] = 1n;
        basis[i] = n + i;
      }
      T[m].fill(0n);
      den[m] = 1n;
    }
  }

  if (!warmed) {
    // ---- phase 1: maximize -(sum of artificials) ----
    const c1 = new Array(n + m).fill(0n);
    for (let j = n; j < n + m; j++) c1[j] = -1n;
    installObjective(T, den, basis, c1, objRow);
    const neverArt = (j) => true; // phase-1 allows any col; artificials never win anyway
    const s1 = simplexLoop(T, den, basis, objRow, rhsCol, neverArt);
    if (s1 === 'unbounded') throw new Error('SIMPLEX_PHASE1_UNBOUNDED');
    const v1 = { n: T[objRow][rhsCol], d: den[objRow] };
    if (!R.isZero(v1)) {
      // infeasible: Farkas certificate from the phase-1 duals.
      // The stored objective row holds s_j = -r_j·L (r_j the reduced cost,
      // L = den[objRow]). For artificial column n+i, c_j = -1 gives
      // r_j = -1 - y'_i, so s_{n+i} = (1 + y'_i)·L and
      //   y'_i = T[obj][n+i]/L - 1.
      // The duals refer to the b-sign-scaled tableau system; multiply each
      // back by rowSign to certify against the caller's (A,b):
      //   yᵀA_j >= 0 for every j,  yᵀb < 0.
      const y = [];
      for (let i = 0; i < m; i++) {
        const yi = R.make(T[objRow][n + i] - den[objRow], den[objRow]);
        y.push(R.mul(yi, { n: rowSign[i], d: 1n }));
      }
      let lhsNonneg = true;
      for (let j = 0; j < n && lhsNonneg; j++) {
        let s = R.zero();
        for (let i = 0; i < m; i++) {
          s = R.add(s, R.mul(y[i], { n: A[i][j], d: 1n }));
        }
        if (R.cmp(s, R.zero()) < 0) lhsNonneg = false;
      }
      let yb = R.zero();
      for (let i = 0; i < m; i++) {
        yb = R.add(yb, R.mul(y[i], { n: b[i], d: 1n }));
      }
      const rhsNeg = R.cmp(yb, R.zero()) < 0;
      return {
        status: 'infeasible',
        farkas: { y, yb, check: { lhsNonneg, rhsNeg } },
      };
    }

    // drive every basic artificial (degenerate, value 0) out of the basis.
  // A row whose artificial stays basic can drift nonzero in phase 2 and the
  // "feasible" basis then silently violates Ax=b. Pivot on any nonzero
  // original column — entering value is rhs/coeff = 0 either way — then
  // renormalize the row sign so the basic coefficient stays positive.
  // A row with no nonzero original column is redundant (0=0): drop it.
  for (let i = 0; i < T.length - 1; i++) {
    if (basis[i] < n) continue;
    let pc = -1;
    for (let j = 0; j < n; j++) {
      if (T[i][j] > 0n) { pc = j; break; }
    }
    if (pc < 0) {
      for (let j = 0; j < n; j++) {
        if (T[i][j] !== 0n) { pc = j; break; }
      }
    }
    if (pc < 0) {
      // redundant row: remove it from the tableau entirely
      T.splice(i, 1);
      den.splice(i, 1);
      basis.splice(i, 1);
      i--;
      continue;
    }
    pivot(T, den, i, pc);
    basis[i] = pc;
    if (T[i][pc] < 0n) {
      for (let j = 0; j < T[i].length; j++) T[i][j] = -T[i][j];
    }
  }
  }

  const objRow2 = T.length - 1; // rows may have been dropped above
  if (c === null) {
    return { status: 'optimal', value: R.zero(), x: null, basis };
  }

  // ---- phase 2: strict Bland throughout — smallest-index entering column,
  // smallest-index basic variable on degenerate ratio ties. Provably finite
  // even on degenerate vertices where Dantzig can cycle.
  installObjective(T, den, basis, c, objRow2);
  const noArt = (j) => j < n;
  const s2 = simplexLoop(T, den, basis, objRow2, rhsCol, noArt, 'bland');
  if (s2 === 'unbounded') return { status: 'unbounded' };
  const value = { n: T[objRow2][rhsCol], d: den[objRow2] };
  const x = new Array(n).fill(null).map(() => R.zero());
  for (let i = 0; i < basis.length; i++) {
    const bcol = basis[i];
    if (bcol < n) x[bcol] = R.make(T[i][rhsCol], T[i][bcol]);
  }
  return { status: 'optimal', value: R.make(value.n, value.d), x, basis };
}

// --------------------------------------------------------------------------
// row reduction: drop linearly dependent rows once per solve so every LP runs
// on a smaller full-rank tableau. A dependent row whose rhs doesn't follow
// the same combination is an early infeasibility certificate.
// --------------------------------------------------------------------------
export function reduceSystem(A, b) {
  const m = A.length;
  const n = A[0].length;
  // integer Bareiss elimination over the augmented [A|b]: every intermediate
  // entry stays an exact minor (the /det division is exact), so coefficients
  // never blow up and no per-row gcd pass is needed. `perm` tracks which
  // ORIGINAL row each surviving row came from so Farkas duals computed on
  // the reduced system lift back to original row coordinates.
  const M = A.map((r, i) => [...r, b[i]]);
  const perm = [...Array(m).keys()];
  let rank = 0;
  let det = 1n;
  for (let c = 0; c < n && rank < m; c++) {
    let r = -1;
    for (let i = rank; i < m; i++) if (M[i][c] !== 0n) { r = i; break; }
    if (r < 0) continue;
    [M[r], M[rank]] = [M[rank], M[r]];
    [perm[r], perm[rank]] = [perm[rank], perm[r]];
    const p = M[rank][c];
    const pr = M[rank];
    for (let i = rank + 1; i < m; i++) {
      const t = M[i][c];
      const ri = M[i];
      // Bareiss rows share ONE scale convention: even when t === 0 the row
      // must still be rescaled by p/det or it stays in last step's minor
      // scale and the next pivot's division mixes scales (inexact BigInt
      // division → silent corruption).
      for (let j = 0; j <= n; j++) {
        if (ri[j] === 0n && pr[j] === 0n) continue;
        ri[j] = (ri[j] * p - t * pr[j]) / det;
      }
    }
    det = p;
    rank++;
  }
  for (let i = rank; i < m; i++) {
    if (M[i][n] !== 0n) return { infeasible: true };
  }
  const keep = perm.slice(0, rank);
  return {
    A: keep.map((i) => A[i]),
    b: keep.map((i) => b[i]),
    rank,
    keep, // original row index of each surviving row
  };
}

// verify a Farkas dual against an ORIGINAL (A,b): yᵀA >= 0 colwise, yᵀb < 0
export function farkasHolds(y, A, b) {
  const m = A.length;
  const n = A[0].length;
  if (!y || y.length !== m) return false;
  for (let j = 0; j < n; j++) {
    let s = R.zero();
    for (let i = 0; i < m; i++) {
      if (y[i].n !== 0n && A[i][j] !== 0n) s = R.add(s, R.mul(y[i], { n: A[i][j], d: 1n }));
    }
    if (R.cmp(s, R.zero()) < 0) return false;
  }
  let yb = R.zero();
  for (let i = 0; i < m; i++) {
    if (y[i].n !== 0n && b[i] !== 0n) yb = R.add(yb, R.mul(y[i], { n: b[i], d: 1n }));
  }
  return R.cmp(yb, R.zero()) < 0;
}

// --------------------------------------------------------------------------
// atomic world space
// --------------------------------------------------------------------------
// World: {h: hypIdx, bits: number[]} — bits[evIdx] ∈ {0,1}.
function* cartesianBits(k) {
  const bits = new Array(k).fill(0);
  for (;;) {
    yield [...bits];
    let i = 0;
    while (i < k) {
      bits[i] ^= 1;
      if (bits[i] === 1) break;
      i++;
    }
    if (i >= k) return;
  }
}

const ctxOk = (c, qc) => c === '*' || c === qc;

function matchWorld(world, spec) {
  // spec: {h: hypIdx|null, evs: [[evIdx, bit]]}
  if (spec.h !== null && world.h !== spec.h) return false;
  for (const [e, b] of spec.evs) if (world.bits[e] !== b) return false;
  return true;
}

// --------------------------------------------------------------------------
// polytope compiler: program decls -> integer rows A w = b over worlds
// --------------------------------------------------------------------------
export function compilePolytope(prog, qc) {
  const idx = {
    partitions: [], priors: [], liks: [], posts: [], joints: [],
    detLinks: [], condIndeps: [], hyps: [], events: [],
  };
  for (const d of prog.decls) {
    switch (d.op) {
      case 'PART_EXHAUSTIVE': idx.partitions.push(d); break;
      case 'HYP': idx.hyps.push(d); break;
      case 'EVENT': idx.events.push(d); break;
      case 'PRIOR': if (ctxOk(d.ctx, qc)) idx.priors.push(d); break;
      case 'LIK': if (ctxOk(d.ctx, qc)) idx.liks.push(d); break;
      case 'POSTERIOR_STATED': if (ctxOk(d.ctx, qc)) idx.posts.push(d); break;
      case 'JOINT_LIK': if (ctxOk(d.ctx, qc)) idx.joints.push(d); break;
      case 'DET_LINK': if (ctxOk(d.ctx, qc)) idx.detLinks.push(d); break;
      case 'COND_INDEP': if (ctxOk(d.ctx, qc)) idx.condIndeps.push(d); break;
      default: break;
    }
  }

  const q = prog.query;
  const partition =
    q.targetKind === 'hyp'
      ? idx.partitions.find((p) => p.hyps.includes(q.target)) ?? idx.partitions[0] ?? null
      : idx.partitions[0] ?? null;
  if (!partition) {
    return { error: { code: 'MISSING_BASE_RATE', detail: 'no PART_EXHAUSTIVE partition' } };
  }
  const hyps = partition.hyps;
  const hypIdx = new Map(hyps.map((h, i) => [h, i]));

  // events referenced by the query or by any ctx-eligible decl
  const evSet = new Set();
  for (const g of q.given) evSet.add(g.ev);
  if (q.targetKind === 'ev') evSet.add(q.target);
  for (const l of idx.liks) evSet.add(l.ev);
  for (const p of idx.posts) evSet.add(p.ev);
  for (const j of idx.joints) for (const s of j.spec) evSet.add(s.ev);
  for (const dl of idx.detLinks) { evSet.add(dl.a); evSet.add(dl.b); }
  for (const ci of idx.condIndeps) for (const e of ci.evs) evSet.add(e);
  const evs = [...evSet].filter((e) => idx.events.some((d) => d.id === e));
  const evIdx = new Map(evs.map((e, i) => [e, i]));

  const k = evs.length;
  if (hyps.length * 2 ** k > MAX_WORLDS) {
    return {
      error: {
        code: 'POLYTOPE_DIMENSION_LIMIT',
        detail: `${hyps.length}·2^${k} = ${hyps.length * 2 ** k} worlds > ${MAX_WORLDS}`,
      },
    };
  }

  // ---- worlds + DET_LINK domain restrictions ----
  let worlds = [];
  for (let h = 0; h < hyps.length; h++) {
    for (const bits of cartesianBits(k)) worlds.push({ h, bits });
  }
  for (const dl of idx.detLinks) {
    const a = evIdx.get(dl.a);
    const b = evIdx.get(dl.b);
    if (a === undefined || b === undefined) continue;
    worlds = worlds.filter((w) => {
      const va = w.bits[a];
      const vb = w.bits[b];
      if (dl.rel === 'EQUIV') return va === vb;
      if (dl.rel === 'IMPLIES') return !(va === 1 && vb === 0);
      if (dl.rel === 'MUTEX') return !(va === 1 && vb === 1);
      return true;
    });
  }
  const W = worlds.length;
  if (W === 0) {
    return {
      error: {
        code: 'CONFLICTING_FIGURES',
        detail: 'DET_LINK domain restrictions annihilate every atomic world',
      },
    };
  }

  // ---- predicate masks -> indicator lookups (cached) ----
  const indCache = new Map();
  const ind = (spec) => {
    const key = `${spec.h ?? '*'}|${spec.evs.map(([e, b]) => `${e}:${b}`).join(',')}`;
    let arr = indCache.get(key);
    if (!arr) {
      arr = worlds.map((w) => (matchWorld(w, spec) ? 1n : 0n));
      indCache.set(key, arr);
    }
    return arr;
  };

  const rows = [];
  const rowsMeta = []; // parallel: the decl that produced each row

  // normalization: Σ w = 1 (the partition is exhaustive by construction)
  {
    const a = new Array(W).fill(1n);
    rows.push({ a, b: 1n });
    rowsMeta.push(partition);
  }
  // PRIOR h v=n/d:  d·Σ_{ω⊨h} w = n
  for (const p of idx.priors) {
    const h = hypIdx.get(p.hyp);
    if (h === undefined) continue;
    const a = ind({ h, evs: [] }).map((v) => v * p.value.d);
    rows.push({ a, b: p.value.n });
    rowsMeta.push(p);
  }
  // LIK e pol GIVEN h v=n/d:  d·Σ_{ω⊨e^pol∧h} w − n·Σ_{ω⊨h} w = 0
  for (const l of idx.liks) {
    const h = hypIdx.get(l.hyp);
    const e = evIdx.get(l.ev);
    if (h === undefined || e === undefined) continue;
    const bit = l.pol === 'POS' ? 1 : 0;
    const a1 = ind({ h, evs: [[e, bit]] });
    const a2 = ind({ h, evs: [] });
    const a = a1.map((v, i) => v * l.value.d - a2[i] * l.value.n);
    rows.push({ a, b: 0n });
    rowsMeta.push(l);
  }
  // POSTERIOR_STATED h GIVEN e pol v=n/d:  d·Σ_{h∧e} w − n·Σ_{e} w = 0
  for (const p of idx.posts) {
    const h = hypIdx.get(p.hyp);
    const e = evIdx.get(p.ev);
    if (h === undefined || e === undefined) continue;
    const bit = p.pol === 'POS' ? 1 : 0;
    const a1 = ind({ h, evs: [[e, bit]] });
    const a2 = ind({ h: null, evs: [[e, bit]] });
    const a = a1.map((v, i) => v * p.value.d - a2[i] * p.value.n);
    rows.push({ a, b: 0n });
    rowsMeta.push(p);
  }
  // JOINT_LIK (spec) GIVEN h v=n/d:  d·Σ_{spec∧h} w − n·Σ_h w = 0
  for (const jl of idx.joints) {
    const h = hypIdx.get(jl.hyp);
    if (h === undefined) continue;
    const spec = jl.spec
      .filter((s) => evIdx.has(s.ev))
      .map((s) => [evIdx.get(s.ev), s.pol === 'POS' ? 1 : 0]);
    if (spec.length !== jl.spec.length) continue;
    const a1 = ind({ h, evs: spec });
    const a2 = ind({ h, evs: [] });
    const a = a1.map((v, i) => v * jl.value.d - a2[i] * jl.value.n);
    rows.push({ a, b: 0n });
    rowsMeta.push(jl);
  }
  // COND_INDEP (evs) GIVEN h — for every subset S (|S|≥2) of events whose
  // marginals are known under h:  P(∧_{e∈S} e | h) = ∏ m_e
  //   pd·Σ_{h∧∧POS(S)} w − pn·Σ_h w = 0
  const marginalOf = (ev, h) => {
    const pos = idx.liks.find((l) => l.ev === ev && l.pol === 'POS' && l.hyp === h);
    if (pos) return pos.value;
    const neg = idx.liks.find((l) => l.ev === ev && l.pol === 'NEG' && l.hyp === h);
    if (neg) return R.sub(R.one(), neg.value);
    return null;
  };
  for (const ci of idx.condIndeps) {
    const hypsFor = ci.hyp === '*' ? hyps : [ci.hyp];
    for (const hname of hypsFor) {
      const h = hypIdx.get(hname);
      if (h === undefined) continue;
      const licensed = ci.evs
        .filter((e) => evIdx.has(e))
        .map((e) => ({ ev: e, i: evIdx.get(e), m: marginalOf(e, hname) }))
        .filter((x) => x.m !== null);
      // all subsets |S|>=2 of licensed events
      const L = licensed;
      const total = 1 << L.length;
      for (let mask = 0; mask < total; mask++) {
        if ((mask & (mask - 1)) === 0) continue; // need >=2 bits
        const S = [];
        for (let t = 0; t < L.length; t++) if (mask & (1 << t)) S.push(L[t]);
        let pn = 1n, pd = 1n;
        for (const s of S) {
          pn *= s.m.n;
          pd *= s.m.d;
        }
        const g = R.gcd(pn, pd);
        pn /= g;
        pd /= g;
        const spec = S.map((s) => [s.i, 1]);
        const a1 = ind({ h, evs: spec });
        const a2 = ind({ h, evs: [] });
        const a = a1.map((v, i) => v * pd - a2[i] * pn);
        rows.push({ a, b: 0n });
        rowsMeta.push(ci);
      }
    }
  }

  return { partition, hyps, hypIdx, evs, evIdx, worlds, rows, rowsMeta, idx };
}

// --------------------------------------------------------------------------
// naive vertex enumeration (independent cross-check for tests)
// Enumerates every basic feasible solution of {w>=0 : Aw=b} by choosing basis
// column sets and solving exactly. O(C(n,m)) — for small cross-check systems.
// --------------------------------------------------------------------------
function exactSolveSquare(M, b) {
  // Solve M x = b over Q via Bareiss-style elimination. Returns null if singular.
  const n = M.length;
  const A = M.map((row, i) => [...row.map((v) => BigInt(v)), BigInt(b[i])]);
  let det = 1n;
  for (let c = 0; c < n; c++) {
    let r = -1;
    for (let i = c; i < n; i++) if (A[i][c] !== 0n) { r = i; break; }
    if (r < 0) return null;
    if (r !== c) { [A[r], A[c]] = [A[c], A[r]]; }
    const p = A[c][c];
    for (let i = c + 1; i < n; i++) {
      const t = A[i][c];
      for (let j = c; j <= n; j++) A[i][j] = (A[i][j] * p - t * A[c][j]) / det;
    }
    det = p;
  }
  // back-substitute: rows are scaled; solve from last row up
  const x = new Array(n).fill(0n).map(() => ({ n: 0n, d: 1n }));
  for (let i = n - 1; i >= 0; i--) {
    // A[i] now has nonzeros possibly at cols >= i (fraction-free upper form)
    // row value: Σ_j A[i][j] x_j = A[i][n]  — entries are scaled by det ratios,
    // but ratios cancel in x = A[i][n]/A[i][i] after substitution terms cancel.
    let num = { n: A[i][n], d: 1n };
    for (let j = i + 1; j < n; j++) {
      if (A[i][j] !== 0n) {
        num = R.sub(num, R.mul({ n: A[i][j], d: 1n }, x[j]));
      }
    }
    x[i] = R.div(num, { n: A[i][i], d: 1n });
  }
  return x;
}

/**
 * vertexEnumerate(A, b, n) — all basic feasible solutions of Ax=b, x>=0.
 * Returns [{x:[rationals], basis:[colIdx]}]. Small systems only.
 */
export function vertexEnumerate(A, b, n) {
  const m = A.length;
  const out = [];
  const cols = [...Array(n).keys()];
  const combos = [];
  const choose = (start, k, acc) => {
    if (k === 0) { combos.push([...acc]); return; }
    for (let i = start; i <= n - k; i++) {
      acc.push(cols[i]);
      choose(i + 1, k - 1, acc);
      acc.pop();
    }
  };
  choose(0, m, []);
  for (const basis of combos) {
    const M = A.map((row) => basis.map((j) => row[j]));
    const xb = exactSolveSquare(M, b);
    if (!xb) continue;
    if (xb.some((v) => R.cmp(v, R.zero()) < 0)) continue;
    const x = new Array(n).fill(null).map(() => R.zero());
    basis.forEach((j, i) => { x[j] = xb[i]; });
    // reject duplicate vertices (degenerate bases)
    if (out.some((v) => v.x.every((xv, i) => R.eq(xv, x[i])))) continue;
    out.push({ x, basis });
  }
  return out;
}

// --------------------------------------------------------------------------
// solveCredal — the single unified decision theorem (ARCH-SPEC §2.2)
// --------------------------------------------------------------------------
function refusal(code, details) {
  return { status: 'REFUSED', code, details };
}

/**
 * solveCredal(program, qc?) — exact credal decision for the program's
 * ASK_PROB query. Returns {status:'ANSWER',...} or {status:'REFUSED',...}.
 */
export function solveCredal(program, qc = null) {
  const prog = toProgram(program);
  const q = prog.query;
  const ctx = qc ?? q.ctx;
  const compiled = compilePolytope(prog, ctx);
  if (compiled.error) {
    return refusal(compiled.error.code, { detail: compiled.error.detail });
  }
  const { hyps, hypIdx, evs, evIdx, worlds, rows, idx } = compiled;
  const W = worlds.length;
  const n = W;
  const origA = rows.map((r) => r.a);
  const origB = rows.map((r) => r.b);
  // one exact row-reduction shared by every LP below: dependent rows are
  // dropped (a dependent row with inconsistent rhs certifies infeasibility
  // immediately); all later LPs run on the smaller full-rank system.
  const reduced = reduceSystem(origA, origB);
  // A Farkas dual reported to the caller must certify against the ORIGINAL
  // compiled rows — never the reduced coordinates. Duals found on the
  // reduced system lift by scattering into the surviving rows' original
  // indices (kept rows ARE original rows; dropped rows get coefficient 0).
  // The lifted vector is always re-verified; on any mismatch the fallback
  // re-runs phase 1 on the original rows directly.
  const originalRowFarkas = (farkas) => {
    if (farkas && farkas.check.lhsNonneg && farkas.check.rhsNeg) {
      const y = new Array(origA.length).fill(null).map(() => R.zero());
      reduced.keep.forEach((origIdx, k) => {
        y[origIdx] = farkas.y[k];
      });
      if (farkasHolds(y, origA, origB)) return y.map((v) => R.toString(v));
    }
    const probe = simplexLP(origA, origB, null);
    return probe.farkas && probe.farkas.check.lhsNonneg && probe.farkas.check.rhsNeg
      ? probe.farkas.y.map((v) => R.toString(v))
      : null;
  };
  if (reduced.infeasible) {
    // certify against the un-reduced system: the reduction's row ops can find
    // the inconsistency but a certificate must come from the real tableau
    return refusal('CONFLICTING_FIGURES', {
      detail: 'linearly inconsistent premise rows — declared figures are mutually inconsistent',
      farkas: originalRowFarkas(null),
    });
  }
  const A = reduced.A;
  const b = reduced.b;

  // ---------- masks ----------
  const bitOf = (e) => evIdx.get(e);
  const specMask = (spec) =>
    worlds.map((w) => (matchWorld(w, spec) ? 1n : 0n));
  const andMask = (m1, m2) => m1.map((v, i) => v * m2[i]);

  const eMask = q.given.length
    ? q.given.every((g) => bitOf(g.ev) !== undefined)
      ? specMask({ h: null, evs: q.given.map((g) => [bitOf(g.ev), g.pol === 'POS' ? 1 : 0]) })
      : new Array(W).fill(0n) // a given on an event outside the model is unsatisfiable
    : new Array(W).fill(1n);
  let tMask;
  if (q.targetKind === 'hyp') {
    const h = hypIdx.get(q.target);
    if (h === undefined) {
      return refusal('MISSING_BASE_RATE', { detail: `target ${q.target} outside partition` });
    }
    tMask = specMask({ h, evs: [] });
  } else {
    const e = bitOf(q.target);
    tMask = specMask({ h: null, evs: [[e, 1]] });
  }
  if (q.negated) tMask = tMask.map((v) => (v === 0n ? 1n : 0n));
  const nMask = andMask(tMask, eMask);

  const witnesses = prog.decls.filter(
    (d) =>
      (d.op === 'PRIOR' || d.op === 'LIK' || d.op === 'POSTERIOR_STATED' ||
        d.op === 'JOINT_LIK' || d.op === 'COND_INDEP' || d.op === 'DET_LINK' ||
        d.op === 'OBS') &&
      ctxOk(d.ctx, ctx)
  );

  // ---------- unique-point fast path ---------------------------------------
  // Full column rank (rank === W): the equality system already pins a single
  // mass assignment w* — the "polytope" is one point. Solve it directly in
  // exact integer arithmetic and evaluate E(w*), N(w*) with zero simplex
  // pivots. A w* coordinate < 0 makes the polytope empty: refuse with an
  // original-row Farkas certificate.
  if (reduced.rank === W) {
    const wStar = exactSolveSquare(A, b);
    if (wStar !== null) {
      if (wStar.some((v) => R.cmp(v, R.zero()) < 0)) {
        return refusal('CONFLICTING_FIGURES', {
          detail:
            'unique solution of the premise system places negative mass — feasible polytope is empty',
          farkas: originalRowFarkas(null),
        });
      }
      const dotInt = (mask) =>
        mask.reduce((s, m, j) => (m === 0n ? s : R.add(s, R.mul({ n: m, d: 1n }, wStar[j]))), R.zero());
      const eStar = dotInt(eMask);
      if (R.isZero(eStar)) {
        return refusal('DEGENERATE_PARTITION', {
          detail: 'conditioning evidence has zero support over every mass assignment in the polytope',
          eMax: '0/1',
          evs: q.given.map((g) => `${g.ev}:${g.pol}`),
        });
      }
      const value = R.div(dotInt(nMask), eStar);
      const selfGiven =
        q.targetKind === 'ev' && q.given.some((g) => g.ev === q.target);
      return {
        status: 'ANSWER',
        value,
        target: q.target,
        via: selfGiven ? 'TAUTOLOGY' : 'CREDAL_LP',
        interval: { min: value, max: value },
        witnesses,
        credal: {
          worlds: W,
          eMin: R.toString(eStar),
          eMax: R.toString(eStar),
          uStar: wStar
            .map((w, i) => ({ i, v: R.toString(R.div(w, eStar)) }))
            .filter((x) => x.v !== '0/1'),
          tStar: R.toString(R.div(R.one(), eStar)),
        },
      };
    }
    // wStar === null cannot happen at full column rank — fall through to LPs
  }

  // ---------- 1. feasibility + evidence support ----------
  // The eMax LP detects polytope infeasibility itself — no separate
  // feasibility pass needed. When there are no givens a feasibility-only
  // LP is still required.
  const hasGiven = q.given.length > 0;
  let eMin = R.one();
  let eMax = R.one();
  let feas = null;
  let lpMaxRef = null;
  if (hasGiven) {
    const cE = eMask;
    const cNeg = cE.map((v) => -v);
    const lpMax = simplexLP(A, b, cE);
    lpMaxRef = lpMax;
    if (lpMax.status === 'infeasible') {
      feas = lpMax;
    } else {
      eMax = lpMax.status === 'optimal' ? lpMax.value : R.zero();
      if (R.isZero(eMax)) {
        return refusal('DEGENERATE_PARTITION', {
          detail: 'conditioning evidence has zero support over every mass assignment in the polytope',
          eMax: '0/1',
          evs: q.given.map((g) => `${g.ev}:${g.pol}`),
        });
      }
      // warm start: lpMax's optimal basis is feasible for the same (A,b)
      const lpMin = simplexLP(A, b, cNeg, lpMax.basis);
      eMin = lpMin.status === 'optimal' ? R.neg(lpMin.value) : R.zero();
      if (R.isZero(eMin)) {
        return refusal('ZERO_SUPPORT_POSSIBLE', {
          detail: 'evidence support is not guaranteed positive over the polytope',
          eMin: '0/1',
          eMax: R.toString(eMax),
        });
      }
    }
  } else {
    feas = simplexLP(A, b, null);
  }
  if (feas && feas.status === 'infeasible') {
    return refusal('CONFLICTING_FIGURES', {
      detail: 'feasible polytope is empty — declared figures are mutually inconsistent',
      farkas: originalRowFarkas(feas.farkas),
    });
  }

  // ---------- 3. Charnes-Cooper: min/max N(w)/E(w) ----------
  // variables: u_0..u_{W-1}, t (index W)
  const rowsCC = [];
  for (let i = 0; i < A.length; i++) {
    const a = new Array(W + 1).fill(0n);
    for (let j = 0; j < W; j++) a[j] = A[i][j];
    a[W] = -b[i];
    rowsCC.push({ a, b: 0n });
  }
  {
    // Σ_{ω⊨E} u = 1  (the Charnes-Cooper normalization)
    const a = new Array(W + 1).fill(0n);
    for (let j = 0; j < W; j++) a[j] = eMask[j];
    rowsCC.push({ a, b: 1n });
  }
  const ACC = rowsCC.map((r) => r.a);
  const bCC = rowsCC.map((r) => r.b);
  const cObj = new Array(W + 1).fill(0n);
  for (let j = 0; j < W; j++) cObj[j] = nMask[j];
  const cNeg = cObj.map((v) => -v);

  // warm-start the CC LP from the w-space optimal basis: for a w-vertex with
  // E(w*)>0, u*=w*/E(w*), t*=1/E(w*) is feasible for the CC system, so the
  // hint [w-basis..., t] canonicalizes to a feasible CC basis directly.
  const lpE = hasGiven ? lpMaxRef : feas;
  const warmCC =
    lpE && lpE.status === 'optimal' && lpE.basis && lpE.basis.length === A.length
      ? [...lpE.basis, W]
      : null;
  const lpPMax = simplexLP(ACC, bCC, cObj, warmCC);
  const lpPMin = simplexLP(
    ACC,
    bCC,
    cNeg,
    lpPMax.status === 'optimal' ? lpPMax.basis : null
  );
  const pMax = lpPMax.status === 'optimal' ? lpPMax.value : null;
  const pMin = lpPMin.status === 'optimal' ? R.neg(lpPMin.value) : null;

  if (pMin === null || pMax === null) {
    return refusal('UNDERDETERMINED_INTERVAL', {
      detail: 'fractional LP failed to bound the query',
      lpStatus: [lpPMin.status, lpPMax.status],
    });
  }

  if (R.eq(pMin, pMax)) {
    const selfGiven =
      q.targetKind === 'ev' && q.given.some((g) => g.ev === q.target);
    return {
      status: 'ANSWER',
      value: pMin,
      target: q.target,
      via: selfGiven ? 'TAUTOLOGY' : 'CREDAL_LP',
      interval: { min: pMin, max: pMax },
      witnesses,
      credal: {
        worlds: W,
        eMin: R.toString(eMin),
        eMax: R.toString(eMax),
        uStar: lpPMax.x
          .slice(0, W)
          .map((v, i) => ({ i, v: R.toString(v) }))
          .filter((x) => x.v !== '0/1'),
        tStar: R.toString(lpPMax.x[W]),
      },
    };
  }

  // ---------- underdetermined: tight Fréchet bounds + diagnostic cause ----------
  const needEvs = new Set(q.given.map((g) => g.ev));
  if (q.targetKind === 'ev') needEvs.add(q.target);
  const missingPriors = hyps.filter((h) => !idx.priors.some((p) => p.hyp === h));
  const missingLiks = [];
  for (const g of q.given) {
    for (const h of hyps) {
      if (!idx.liks.some((l) => l.ev === g.ev && l.hyp === h)) {
        missingLiks.push(`${g.ev}|${h}`);
      }
    }
  }
  const postReverse = idx.posts.find(
    (p) =>
      (q.targetKind === 'hyp' && p.hyp === q.target) ||
      needEvs.has(p.ev) ||
      (q.targetKind === 'hyp' && p.ev === q.target)
  );
  const anyUnlicensedPair = (() => {
    const evs_ = [...needEvs];
    for (let i = 0; i < evs_.length; i++) {
      for (let j = i + 1; j < evs_.length; j++) {
        const a = evs_[i], bb = evs_[j];
        const linked = idx.detLinks.some(
          (dl) => (dl.a === a && dl.b === bb) || (dl.a === bb && dl.b === a)
        );
        const ciCover = hyps.every((h) =>
          idx.condIndeps.some(
            (ci) =>
              (ci.hyp === h || ci.hyp === '*') &&
              ci.evs.includes(a) &&
              ci.evs.includes(bb)
          )
        );
        const jlCover = idx.joints.some((jl) => {
          const s = new Set(jl.spec.map((x) => x.ev));
          return s.has(a) && s.has(bb);
        });
        if (!linked && !ciCover && !jlCover) return true;
      }
    }
    return false;
  })();
  const cause = anyUnlicensedPair
    ? 'UNLICENSED_INDEPENDENCE'
    : missingPriors.length > 0
      ? 'MISSING_BASE_RATE'
      : postReverse
        ? 'REVERSED_CONDITIONAL'
        : 'INCOMPLETE_FACTORIZATION';
  return refusal('UNDERDETERMINED_INTERVAL', {
    detail: 'premises do not determine a unique probability — exact credal interval reported',
    interval: { min: R.toString(pMin), max: R.toString(pMax) },
    cause,
    eMin: R.toString(eMin),
    eMax: R.toString(eMax),
    missingPriors,
    missingLiks,
  });
}
