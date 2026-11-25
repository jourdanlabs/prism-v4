// venn-abers :: Inductive Mondrian Venn-Abers Predictor (ARCH-SPEC §4).
// Distribution-free finite-sample probability intervals [p_low, p_high]
// stratified by mode/task family, plus the regularized point estimate
// p_cal = p_high / (1 - p_low + p_high).
//
// Isotonic core: PAVA over score-sorted calibration labels. For a test score
// s we fit twice — once with the test point labeled 0, once labeled 1 — and
// read off the two fitted values. That interval is the Venn-Abers predictor.

// PAVA: rows sorted by score; equal scores grouped. Returns blocks in order
// with fitted (pooled) means.
function isotonicBlocks(rows) {
  // group equal scores
  const blocks = [];
  for (const r of rows) {
    const last = blocks[blocks.length - 1];
    if (last && last.score === r.score) {
      last.sumY += r.y;
      last.n += 1;
      last.hasTest = last.hasTest || r.test;
    } else {
      blocks.push({ score: r.score, sumY: r.y, n: 1, hasTest: !!r.test });
    }
  }
  // pool adjacent violators (mean strictly decreasing -> merge)
  const fit = blocks.map((b) => ({ ...b, mean: b.sumY / b.n }));
  const out = [];
  for (const b of fit) {
    out.push(b);
    while (out.length >= 2 && out[out.length - 2].mean > out[out.length - 1].mean) {
      const a = out.splice(out.length - 2, 1)[0];
      const c = out[out.length - 1];
      c.sumY += a.sumY;
      c.n += a.n;
      c.mean = c.sumY / c.n;
      c.hasTest = c.hasTest || a.hasTest;
    }
  }
  return out;
}

function fitAndRead(calRows, score, y) {
  const rows = [...calRows, { score, y, test: true }].sort(
    (a, b) => a.score - b.score
  );
  const blocks = isotonicBlocks(rows);
  const mine = blocks.find((b) => b.hasTest);
  return mine ? mine.mean : null;
}

export class VennAbersCalibrator {
  /**
   * fit(rows): rows = [{score:number in [0,1], y:0|1, stratum?:string}]
   * Mondrian: when stratumFn is supplied, separate calibrators per stratum.
   */
  constructor(stratumFn = null) {
    this.stratumFn = stratumFn;
    this.strata = new Map(); // stratum -> [{score,y}]
  }
  fit(rows) {
    for (const r of rows) {
      const key = this.stratumFn ? this.stratumFn(r) : 'ALL';
      if (!this.strata.has(key)) this.strata.set(key, []);
      this.strata.get(key).push({ score: r.score, y: r.y });
    }
    for (const rows2 of this.strata.values()) {
      rows2.sort((a, b) => a.score - b.score);
    }
    return this;
  }
  /**
   * predict(score, stratum?) -> {pLow, pHigh, pCal, n}
   * Falls back to the pooled 'ALL' stratum when the requested stratum has
   * too few calibration rows (<2) or was never seen.
   */
  predict(score, stratum = null) {
    const key = this.stratumFn ? stratum ?? 'ALL' : 'ALL';
    let rows = this.strata.get(key);
    if (!rows || rows.length < 2) rows = this.strata.get('ALL') ?? rows;
    if (!rows || rows.length < 2) {
      // cold start: maximally wide interval — abstention should trigger
      return { pLow: 0, pHigh: 1, pCal: 0.5, n: rows?.length ?? 0, stratum: key };
    }
    const p0 = fitAndRead(rows, score, 0);
    const p1 = fitAndRead(rows, score, 1);
    const pLow = Math.max(0, Math.min(p0, p1));
    const pHigh = Math.min(1, Math.max(p0, p1));
    const pCal = pHigh / (1 - pLow + pHigh);
    return { pLow, pHigh, pCal, n: rows.length, stratum: key };
  }
}
