// mahalanobis :: encoder-manifold OOD distance + abstention threshold.
// d_OOD(H) = sqrt((x-mu)^T S^-1 (x-mu)) with ridge-regularized covariance.
// Threshold = the q-quantile of calibration distances (default q=0.99).
// (Statistical layer — float arithmetic is correct here; the exact-rational
// domain is the kernel's, not the calibrator's.)

export function mean(vectors) {
  const d = vectors[0].length;
  const mu = new Array(d).fill(0);
  for (const v of vectors) for (let i = 0; i < d; i++) mu[i] += v[i];
  return mu.map((x) => x / vectors.length);
}

export function covariance(vectors, mu, ridge = 1e-6) {
  const d = mu.length;
  const S = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const v of vectors) {
    for (let i = 0; i < d; i++) {
      const di = v[i] - mu[i];
      for (let j = 0; j <= i; j++) S[i][j] += di * (v[j] - mu[j]);
    }
  }
  const n = Math.max(1, vectors.length - 1);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      S[i][j] /= n;
      S[j][i] = S[i][j];
    }
    S[i][i] += ridge; // ridge on diagonal after symmetrize
  }
  return S;
}

// Gauss-Jordan inverse with partial pivoting (small d).
export function invert(S) {
  const d = S.length;
  const A = S.map((row, i) => [...row, ...row.map((_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < d; col++) {
    let piv = col;
    for (let r = col + 1; r < d; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    }
    if (Math.abs(A[piv][col]) < 1e-15) {
      A[piv][col] = 1e-15;
    }
    [A[col], A[piv]] = [A[piv], A[col]];
    const inv = 1 / A[col][col];
    for (let j = 0; j < 2 * d; j++) A[col][j] *= inv;
    for (let r = 0; r < d; r++) {
      if (r === col) continue;
      const f = A[r][col];
      for (let j = 0; j < 2 * d; j++) A[r][j] -= f * A[col][j];
    }
  }
  return A.map((row) => row.slice(d));
}

export class MahalanobisOOD {
  constructor(ridge = 1e-6) {
    this.ridge = ridge;
    this.mu = null;
    this.inv = null;
    this.threshold = null;
    this.calDistances = [];
  }
  fit(vectors, q = 0.99) {
    this.mu = mean(vectors);
    const S = covariance(vectors, this.mu, this.ridge);
    this.inv = invert(S);
    this.calDistances = vectors.map((v) => this.distance(v)).sort((a, b) => a - b);
    const idx = Math.min(this.calDistances.length - 1, Math.floor(q * this.calDistances.length));
    this.threshold = this.calDistances[idx];
    return this;
  }
  distance(x) {
    const d = this.mu.length;
    const diff = x.map((v, i) => v - this.mu[i]);
    let acc = 0;
    for (let i = 0; i < d; i++) {
      let row = 0;
      for (let j = 0; j < d; j++) row += this.inv[i][j] * diff[j];
      acc += diff[i] * row;
    }
    return Math.sqrt(Math.max(0, acc));
  }
  isOOD(x, threshold = this.threshold) {
    return this.distance(x) > threshold;
  }
}

/**
 * abstain({interval:{pLow,pHigh}, distance, threshold, tauWidth})
 * Combined abstention rule (ARCH-SPEC §4): ABSTAIN when the Venn-Abers
 * interval is too wide OR the point is off the calibrated manifold.
 */
export function abstain({ pLow, pHigh, distance = null, threshold = null, tauWidth = 0.5 }) {
  if (pHigh - pLow > tauWidth) {
    return { abstain: true, reason: 'INTERVAL_TOO_WIDE', width: pHigh - pLow };
  }
  if (distance !== null && threshold !== null && distance > threshold) {
    return { abstain: true, reason: 'OUT_OF_CALIBRATED_DOMAIN', distance, threshold };
  }
  return { abstain: false };
}
