import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  scoreChoices,
  symmetryTest,
  permutations,
} from '../src/choice/perm-equivariant.js';
import { VennAbersCalibrator } from '../src/calib/venn-abers.js';
import { MahalanobisOOD, abstain } from '../src/calib/mahalanobis.js';
import { mulberry32 } from '../src/choice/perm-equivariant.js';

// ---------- permutation-equivariant choice ----------
const CONTEXT =
  'A packet relay dropped 3 of 40 frames before the patch and 0 of 40 after. Which statement is supported?';
const CHOICES = [
  'The patch eliminated frame loss in this sample.',
  'The relay is guaranteed loss-free going forward.',
  'Frame loss was caused by upstream congestion.',
  'The patch had no measurable effect.',
  'Forty frames were dropped after patching.',
];

test('choice: deterministic scoring', () => {
  const a = scoreChoices(CONTEXT, CHOICES);
  const b = scoreChoices(CONTEXT, CHOICES);
  assert.deepEqual(a.logits, b.logits);
});

test('choice: 0.000% option-order flip across all K! permutations (K=2..5)', () => {
  for (let k = 2; k <= 5; k++) {
    const res = symmetryTest(CONTEXT, CHOICES.slice(0, k));
    assert.equal(res.flips, 0, `K=${k}: ${res.flips} argmax flips`);
    assert.equal(res.maxLogitDrift, 0, `K=${k}: logit drift ${res.maxLogitDrift}`);
    assert.equal(res.flipRate, 0);
    assert.equal(res.permutationsTested, factorial(k) - 1);
  }
});

test('choice: exact equivariance — logits permute bitwise', () => {
  const base = scoreChoices(CONTEXT, CHOICES);
  const p = [3, 0, 4, 1, 2];
  const res = scoreChoices(CONTEXT, p.map((i) => CHOICES[i]));
  for (let i = 0; i < res.logits.length; i++) {
    assert.equal(res.logits[i], base.logits[p[i]]);
    assert.equal(res.probs[i], base.probs[p[i]]);
  }
});

test('choice: maximal tie picks a canonical winner and flags abstention', () => {
  // identical token content -> bitwise-identical logits -> a real tie.
  const tied = ['KILN LAMP LIT!', 'kiln lamp lit?'];
  const a = scoreChoices(CONTEXT, tied);
  assert.equal(a.logits[0], a.logits[1]);
  assert.equal(a.probs[0], 0.5);
  assert.ok(a.tie !== null, 'tie should be reported');
  assert.equal(a.abstain, true, 'a maximal tie must not release a pick');
  // canonical winner is content-derived: the same CHOICE wins in either order
  const rev = [tied[1], tied[0]];
  const b = scoreChoices(CONTEXT, rev);
  assert.equal(tied[a.argmax], rev[b.argmax], 'winner must be permutation-stable');
  // a tie can never flip the winner under permutation
  const sym = symmetryTest(CONTEXT, tied);
  assert.equal(sym.flips, 0);
});

function factorial(n) {
  return n <= 1 ? 1 : n * factorial(n - 1);
}

// ---------- Venn-Abers ----------
test('venn-abers: intervals ordered, contained in [0,1], point in interval', () => {
  const rows = [];
  const rng = mulberry32(42);
  // calibration where true P(y=1|score) ~ score
  for (let i = 0; i < 200; i++) {
    const s = rng();
    rows.push({ score: s, y: rng() < s ? 1 : 0 });
  }
  const va = new VennAbersCalibrator().fit(rows);
  for (const s of [0.1, 0.3, 0.5, 0.7, 0.9]) {
    const { pLow, pHigh, pCal } = va.predict(s);
    assert.ok(0 <= pLow && pLow <= pHigh && pHigh <= 1, `interval [${pLow},${pHigh}]`);
    assert.ok(pLow <= pCal && pCal <= pHigh || Math.abs(pCal - pHigh) < 1e-9);
  }
});

test('venn-abers: perfectly separated calibration gives tight intervals', () => {
  const rows = [
    ...Array.from({ length: 50 }, (_, i) => ({ score: i / 100, y: 0 })),
    ...Array.from({ length: 50 }, (_, i) => ({ score: 0.5 + i / 100, y: 1 })),
  ];
  const va = new VennAbersCalibrator().fit(rows);
  const low = va.predict(0.2);
  const high = va.predict(0.8);
  assert.ok(low.pHigh < 0.3, `low-score pHigh ${low.pHigh} should be small`);
  assert.ok(high.pLow > 0.7, `high-score pLow ${high.pLow} should be large`);
  assert.ok(high.pCal > low.pCal);
});

test('venn-abers: mondrian strata are independent', () => {
  const rows = [
    ...Array.from({ length: 40 }, () => ({ score: 0.5, y: 0, stratum: 'A' })),
    ...Array.from({ length: 40 }, () => ({ score: 0.5, y: 1, stratum: 'B' })),
  ];
  const va = new VennAbersCalibrator((r) => r.stratum).fit(rows);
  const a = va.predict(0.5, 'A');
  const b = va.predict(0.5, 'B');
  assert.ok(a.pCal < 0.2, `stratum A pCal ${a.pCal} should be low`);
  assert.ok(b.pCal > 0.8, `stratum B pCal ${b.pCal} should be high`);
});

test('venn-abers: unseen stratum falls back to pooled calibration', () => {
  const va = new VennAbersCalibrator((r) => r.stratum);
  va.fit([{ score: 0.4, y: 0, stratum: 'A' }, { score: 0.6, y: 1, stratum: 'A' }]);
  const r = va.predict(0.5, 'NEVER_SEEN');
  assert.ok(r.pLow <= r.pCal && r.pCal <= r.pHigh);
});

// ---------- Mahalanobis OOD ----------
test('mahalanobis: in-distribution small, far point large, abstain fires', () => {
  const rng = mulberry32(7);
  const cal = Array.from({ length: 120 }, () => [rng(), rng(), rng()]);
  const ood = new MahalanobisOOD().fit(cal, 0.99);
  const near = ood.distance([0.5, 0.5, 0.5]);
  const far = ood.distance([5, -4, 9]);
  assert.ok(near < far, `${near} !< ${far}`);
  assert.equal(ood.isOOD([5, -4, 9]), true);
  assert.equal(ood.isOOD([0.5, 0.5, 0.5]), false);
});

test('abstain: wide interval or OOD -> abstain with reason', () => {
  assert.deepEqual(
    abstain({ pLow: 0.1, pHigh: 0.9, tauWidth: 0.5 }).reason,
    'INTERVAL_TOO_WIDE'
  );
  assert.equal(
    abstain({ pLow: 0.4, pHigh: 0.6, distance: 10, threshold: 3, tauWidth: 0.5 }).reason,
    'OUT_OF_CALIBRATED_DOMAIN'
  );
  assert.equal(abstain({ pLow: 0.4, pHigh: 0.6, distance: 1, threshold: 3 }).abstain, false);
});
