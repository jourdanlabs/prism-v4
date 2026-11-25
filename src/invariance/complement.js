// invariance-gate :: Slot-4 query-target negation / complement sanity.
// Factor heads must satisfy p(Target) + p(NOT Target) == 1/1 EXACTLY.

import * as R from '../kernel/rational.js';
import { solve } from '../kernel/solve.js';

// Flip the query's negation flag and re-solve under identical premises.
export function complementCheck({ program, X }) {
  const negProg = {
    decls: program.decls,
    query: { ...program.query, negated: !program.query.negated },
  };
  const d1 = solve(program, X);
  if (d1.status !== 'ANSWER') {
    return { ok: true, skipped: true, reason: `slot-1 ${d1.status}:${d1.code ?? ''}` };
  }
  const d2 = solve(negProg, X);
  if (d2.status !== 'ANSWER') {
    return {
      ok: false,
      code: 'COMPLEMENT_VIOLATION',
      reason: `negated query refused: ${d2.code}`,
    };
  }
  const sum = R.add(d1.value, d2.value);
  if (!R.isOne(sum)) {
    return {
      ok: false,
      code: 'COMPLEMENT_VIOLATION',
      p: R.toString(d1.value),
      pNeg: R.toString(d2.value),
      sum: R.toString(sum),
    };
  }
  return { ok: true, p: R.toString(d1.value), pNeg: R.toString(d2.value) };
}
