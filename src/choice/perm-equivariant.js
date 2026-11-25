// perm-equivariant-choice :: reference Permutation-Equivariant Choice scorer.
// (ARCH-SPEC §2.2 — Path A.) Two set-equivariant competition layers with ZERO
// positional embeddings; every set-wide reduction (attention denominator,
// weighted value sum) iterates elements in a CANONICAL order derived from
// content, not from input position. Result: logits are bitwise equivariant —
// PRISM_choice(X, π(C)) === π(PRISM_choice(X, C)) exactly, so the option-order
// flip rate is 0.000% by construction rather than by calibration.

import { createHash } from 'node:crypto';

function sha256hex(s) {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

// Deterministic seeded RNG (mulberry32) for fixed weights.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const DIM = 16;
const rand = mulberry32(0x50524953); // "PRIS"
const W = {
  q1: matrix(DIM, DIM), k1: matrix(DIM, DIM), v1: matrix(DIM, DIM),
  q2: matrix(DIM, DIM), k2: matrix(DIM, DIM), v2: matrix(DIM, DIM),
  m1a: matrix(DIM, DIM), m1b: vector(DIM), m2a: matrix(DIM, DIM), m2b: vector(DIM),
  head: vector(DIM), bias: (rand() - 0.5) * 0.1,
};
function matrix(r, c) {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (rand() - 0.5) * 0.4)
  );
}
function vector(n) {
  return Array.from({ length: n }, () => (rand() - 0.5) * 0.4);
}

const matvec = (M, v) => M.map((row) => row.reduce((a, w, i) => a + w * v[i], 0));
const vecfma = (a, b, s) => a.map((x, i) => x + b[i] * s);
const dot = (a, b) => a.reduce((x, v, i) => x + v * b[i], 0);
const tanhVec = (v) => v.map(Math.tanh);

// Per-token deterministic pseudo-embedding (hash -> unit-ish vector).
function tokenVec(tok) {
  const out = new Array(DIM);
  for (let i = 0; i < DIM; i++) {
    const h = sha256hex(`${tok}::${i}`);
    out[i] = (parseInt(h.slice(0, 8), 16) / 0xffffffff) * 2 - 1;
  }
  return out;
}

const tokenize = (s) => s.toLowerCase().split(/[^a-z0-9%]+/).filter(Boolean);

// featurize(context, choice) -> u_k : content-only embedding.
function featurize(context, choice) {
  const ct = tokenize(choice);
  const xt = new Set(tokenize(context));
  const u = new Array(DIM).fill(0);
  for (const t of ct) {
    const tv = tokenVec(t);
    for (let i = 0; i < DIM; i++) u[i] += tv[i];
  }
  const n = Math.max(1, ct.length);
  for (let i = 0; i < DIM; i++) u[i] /= n;
  let overlap = 0;
  for (const t of ct) if (xt.has(t)) overlap++;
  u[0] += overlap / n; // lexical relevance channel
  u[1] += Math.min(1, ct.length / 32); // length channel
  return u;
}

// Canonical order over a set of vectors: sort by sha256 of the IEEE-754 bytes.
// Permutation-invariant BY CONSTRUCTION — identical for any input ordering.
function canonOrder(vectors) {
  const buf = new ArrayBuffer(8);
  const dv = new DataView(buf);
  return vectors
    .map((v, i) => {
      const bytes = v
        .map((x) => {
          dv.setFloat64(0, x);
          let s = '';
          for (let b = 0; b < 8; b++) s += String(dv.getUint8(b)).padStart(3, '0');
          return s;
        })
        .join('|');
      return { v, i, key: sha256hex(bytes) };
    })
    .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

// One set-equivariant competition layer: content attention + shared MLP.
// No positional information exists anywhere in this function.
function setLayer(U, Wq, Wk, Wv) {
  const ordered = canonOrder(U);
  const K = ordered.map((o) => matvec(Wk, o.v));
  const V = ordered.map((o) => matvec(Wv, o.v));
  const out = new Array(U.length);
  for (let k = 0; k < U.length; k++) {
    const q = matvec(Wq, U[k]);
    // softmax over canonically-ordered logits: denominators accumulate in a
    // fixed order -> bitwise identical under any input permutation.
    const logits = ordered.map((_, j) => dot(q, K[j]) / Math.sqrt(DIM));
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const denom = exps.reduce((a, b) => a + b, 0);
    const ctx = new Array(DIM).fill(0);
    for (let j = 0; j < ordered.length; j++) {
      vecfma(ctx, V[j], exps[j] / denom);
    }
    out[k] = U[k].map((x, i) => x + ctx[i]); // residual
  }
  return out;
}

function mlp(u, Wa, ba, Wb, bb) {
  return tanhVec(vecfma(matvec(Wb, tanhVec(vecfma(matvec(Wa, u), ba, 1))), bb, 1));
}

/**
 * scoreChoices(context, choices) -> {logits, probs, argmax, tie, abstain}
 * Exactly permutation-equivariant: permuting `choices` permutes outputs
 * identically (asserted bitwise in the symmetry test).
 *
 * MAXIMAL-TIE CONTRACT: when two or more options carry bitwise-identical
 * maximal logits, no choice may be silently released as "the" pick —
 * position is the only thing distinguishing them, and position is not
 * evidence. The scorer reports `tie` (the tied indices), sets `abstain:
 * true`, and still returns a `argmax` — the CANONICAL winner, chosen by
 * content hash rather than input position, so the named winner is identical
 * under every permutation of the choice list. Callers that release picks
 * must honor `abstain`.
 */
export function scoreChoices(context, choices) {
  const U = choices.map((c) => featurize(context, c));
  let V = setLayer(U, W.q1, W.k1, W.v1);
  V = V.map((u) => mlp(u, W.m1a, W.m1b, W.m2a, W.m2b));
  V = setLayer(V, W.q2, W.k2, W.v2);
  const logits = V.map((u) => dot(W.head, u) + W.bias);
  const m = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - m));
  // denominator reduced in a content-derived order (sorted exp values) ->
  // bitwise identical under any input permutation.
  const denom = [...exps].sort((a, b) => b - a).reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / denom);
  // maximal set + canonical winner: among bitwise-equal maximal logits pick
  // by content hash (position-independent); otherwise the unique argmax.
  const maxLogit = Math.max(...logits);
  const tiedIdx = [];
  for (let i = 0; i < logits.length; i++) if (logits[i] === maxLogit) tiedIdx.push(i);
  let argmax;
  let tie = null;
  if (tiedIdx.length === 1) {
    argmax = tiedIdx[0];
  } else {
    tie = tiedIdx;
    argmax = tiedIdx
      .map((i) => ({ i, key: sha256hex(choices[i]) }))
      .sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))[0].i;
  }
  return { logits, probs, argmax, tie, abstain: tie !== null };
}

// All K! permutations of [0..K-1].
export function permutations(K) {
  const out = [];
  const used = new Array(K).fill(false);
  const cur = [];
  const rec = () => {
    if (cur.length === K) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < K; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

const applyPerm = (arr, perm) => perm.map((i) => arr[i]);

/**
 * Mathematical symmetry test: run all K! permutations and verify the output
 * is exactly the input permutation applied to the base output — bitwise.
 * Returns { flips, maxLogitDrift, flipRate } — all zero by construction.
 */
export function symmetryTest(context, choices) {
  const base = scoreChoices(context, choices);
  let flips = 0;
  let maxLogitDrift = 0;
  const perms = permutations(choices.length).slice(1); // skip identity
  for (const p of perms) {
    const res = scoreChoices(context, applyPerm(choices, p));
    const expected = applyPerm(base.logits, p);
    for (let i = 0; i < res.logits.length; i++) {
      const drift = Math.abs(res.logits[i] - expected[i]);
      if (drift > maxLogitDrift) maxLogitDrift = drift;
    }
    // map permuted argmax back to base indexing: res.argmax is an index in the
    // permuted array; the choice it names is p[res.argmax] in base indexing.
    if (p[res.argmax] !== base.argmax) flips++;
  }
  const flipRate = perms.length === 0 ? 0 : flips / perms.length;
  return { flips, maxLogitDrift, flipRate, permutationsTested: perms.length };
}
