// crosscheck case generator: emits compiled credal polytopes + the JS
// solver's decisions for independent verification by scripts/vertex_check.py
// (Fraction-exact vertex enumeration). Cases deliberately include degenerate
// vertices and subset-mass polytopes (missing PRIOR/LIK rows leave mass free
// over proper subsets of the worlds).
//   usage: node scripts/crosscheck.mjs [N] > cases.json
import * as R from '../src/kernel/rational.js';
import { toProgram } from '../src/kernel/parser.js';
import { compilePolytope, solveCredal } from '../src/kernel/credal.js';
import { pathToFileURL } from 'node:url';
import { makeRng } from '../src/synth/rng.js';

// Build the full crosscheck corpus deterministically (seeded rng). Exported
// so tests can replay the identical case set — e.g. verifying every
// infeasible decision's Farkas certificate against original compiled rows.
export function generateCases(N = 400) {
const rng = makeRng(20261122);

function rat(r) {
  const d = 1 + Math.floor(r() * 9);
  const n = Math.floor(r() * (d + 1));
  return R.make(BigInt(n), BigInt(d));
}

const cases = [];
for (let t = 0; t < N; t++) {
  const prog = [];
  const nH = rng.chance(0.8) ? 2 : 3;
  const nEv = rng.chance(0.7) ? 2 : 1;
  for (let h = 0; h < nH; h++) prog.push(`HYP h${h} [0,1]`);
  prog.push(`PART_EXHAUSTIVE (${[...Array(nH)].map((_, h) => `h${h}`).join(' ')}) [0,1]`);
  for (let e = 0; e < nEv; e++) prog.push(`EVENT e${e} [0,1]`);
  // priors: a normalized composition (so the mix is dominated by feasible
  // systems — conflicts then come from genuinely contradictory premises,
  // not unnormalized mass)
  {
    const parts = [];
    let rem = 24;
    for (let h = 0; h < nH; h++) {
      const w = h === nH - 1 ? rem : Math.max(1, Math.floor(rng.next() * rem));
      parts.push(w);
      rem -= w;
    }
    for (let h = 0; h < nH; h++) {
      if (rng.chance(0.85)) {
        prog.push(`PRIOR h${h} ${parts[h]}/24 ctx0 [0,1]`);
      }
    }
  }
  for (let e = 0; e < nEv; e++) {
    for (let h = 0; h < nH; h++) {
      const v = rng.chance(0.15) ? R.zero() : rng.chance(0.15) ? R.one() : rat(rng.next);
      if (rng.chance(0.85)) prog.push(`LIK e${e} POS GIVEN h${h} ${R.toString(v)} ctx0 [0,1]`);
    }
  }
  if (rng.chance(0.35)) {
    prog.push(`POSTERIOR_STATED h${Math.floor(rng.next() * nH)} GIVEN e0 POS ${R.toString(rat(rng.next))} ctx0 [0,1]`);
  }
  if (nEv === 2 && rng.chance(0.4)) {
    const spec = rng.chance(0.5) ? 'e0:POS e1:POS' : 'e0:POS e1:NEG';
    prog.push(`JOINT_LIK (${spec}) GIVEN h${Math.floor(rng.next() * nH)} ${R.toString(rat(rng.next))} ctx0 [0,1]`);
  }
  if (nEv === 2 && rng.chance(0.3)) {
    prog.push(`DET_LINK e0 ${rng.pick(['EQUIV', 'IMPLIES', 'MUTEX'])} e1 ctx0 [0,1]`);
  }
  if (nEv === 2 && rng.chance(0.3)) prog.push(`COND_INDEP (e0 e1) GIVEN * ctx0 [0,1]`);

  const roll = rng.next();
  let ask;
  if (nEv === 1) {
    ask =
      roll < 0.5
        ? `ASK_PROB h0 GIVEN (e0:POS) ctx0`
        : roll < 0.75
          ? `ASK_PROB e0 GIVEN (e0:POS) ctx0`
          : `ASK_PROB e0 GIVEN () ctx0`;
  } else if (roll < 0.4) ask = `ASK_PROB h0 GIVEN (e0:POS) ctx0`;
  else if (roll < 0.6) ask = `ASK_PROB e0 GIVEN (e1:POS) ctx0`;
  else if (roll < 0.75) ask = `ASK_PROB e0 GIVEN (e0:POS) ctx0`;
  else if (roll < 0.9) ask = `ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`;
  else ask = `ASK_PROB e0 GIVEN () ctx0`;

  const program = [...prog, ask, 'END'].join('\n');
  const decision = solveCredal(program);
  const out = { t, program, decision };
  const c = compilePolytope(toProgram(program), 'ctx0');
  if (!c.error) {
    const specMask = (spec) =>
      c.worlds.map((w) => {
        const m =
          (spec.h === null || w.h === spec.h) &&
          spec.evs.every(([e, b]) => w.bits[e] === b);
        return m ? '1' : '0';
      });
    const bitOf = (e) => c.evIdx.get(e);
    const q = toProgram(program).query;
    const eMask = q.given.length
      ? q.given.every((g) => bitOf(g.ev) !== undefined)
        ? specMask({ h: null, evs: q.given.map((g) => [bitOf(g.ev), g.pol === 'POS' ? 1 : 0]) })
        : c.worlds.map(() => '0')
      : c.worlds.map(() => '1');
    let tMask;
    if (q.targetKind === 'hyp') {
      const h = c.hypIdx.get(q.target);
      tMask = h === undefined ? null : specMask({ h, evs: [] });
    } else {
      const e = bitOf(q.target);
      tMask = e === undefined ? null : specMask({ h: null, evs: [[e, 1]] });
    }
    if (tMask && q.negated) tMask = tMask.map((v) => (v === '1' ? '0' : '1'));
    const nMask = tMask ? tMask.map((v, i) => (v === '1' && eMask[i] === '1' ? '1' : '0')) : null;
    out.compiled = {
      rows: c.rows.map((r) => ({ a: r.a.map((v) => v.toString()), b: r.b.toString() })),
      worlds: c.worlds.length,
      eMask,
      tMask,
      nMask,
    };
  }
  cases.push(out);
}
return cases;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const cases = generateCases(Number(process.argv[2] ?? 400));
  console.log(
    JSON.stringify(cases, (k, v) =>
      typeof v === 'bigint' ? v.toString() : v && typeof v === 'object' && 'n' in v && 'd' in v ? { n: v.n.toString(), d: v.d.toString() } : v
    )
  );
}
