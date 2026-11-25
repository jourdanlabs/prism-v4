// prism-kernel :: deterministic exact-rational factor-graph solver.
// Enforces Bulma's four obligations (source fidelity, ownership witness,
// partition & factorization, exact rational arithmetic) and emits the 7 typed
// structural refusals + 3 auxiliary codes. Refusal precedence is fixed and
// documented here (ARCH-SPEC §3.1):
//   1  PARSE_ERROR                (structural, from parser)
//   2  DUPLICATE_EVIDENCE         (>=2 OBS on same EvId, or dup in query givens)
//   3  CONFLICTING_FIGURES        (same premise key, incompatible rationals)
//   4  CONTEXT_MISMATCH           (needed premise exists only at other CtxId)
//   5  MISSING_OBSERVATION_WITNESS(aux: query obs without OBS witness decl)
//   6  SCOPED_OR_RETRACTED_PREMISE(needed premise span covered by NON_ASSERTED)
//   7  REVERSED_CONDITIONAL       (needed factor missing; POSTERIOR_STATED in
//                                  the reverse direction offered as substitute)
//   8  MISSING_BASE_RATE          (LIK+OBS present without exhaustive PRIOR set)
//   9  INCOMPLETE_FACTORIZATION   (aux: needed LIK missing, no substitute)
//  10  UNLICENSED_INDEPENDENCE    (>=2 obs multiplied without COND_INDEP, or a
//                                  distinct event target marginalized over
//                                  given evidence without COND_INDEP covering
//                                  target+evidence under every hypothesis)
//  11  CONFLICTING_FIGURES        (CONFLICT decl over needed spans, or stated
//                                  posterior contradicting computed posterior)
//  12  UNACCOUNTED_MATERIAL_SENTENCE (sentence-partition account: a material
//                                  sentence grounded by no decl, or a SENT_TAG
//                                  partition that does not cover 0..m-1)
//  13  DEGENERATE_PARTITION       (aux: zero evidence support over the whole
//                                  credal polytope — computed exactly, not
//                                  guessed from a single factor)
//  14  ZERO_SUPPORT_POSSIBLE      (aux: min E(w)=0 < max E(w) — conditioning
//                                  is possible under some masses but not all)
//  15  UNDERDETERMINED_INTERVAL   (aux: [p_min,p_max] nondegenerate — refused
//                                  with tight exact Fréchet bounds + cause)
//  16  CONFLICTING_FIGURES        (credal polytope empty — Farkas certificate)
// The numeric decision itself is the exact-rational credal polytope kernel
// (src/kernel/credal.js): every decl compiles to linear constraints over
// atomic worlds; the conditional query is bounded by Charnes-Cooper LP.

import * as R from './rational.js';
import { toProgram, ParseError } from './parser.js';
import { solveCredal } from './credal.js';
import {
  checkSentenceAccounting,
  validateLicenseDecls,
  sentences,
  sentenceAt,
  sentenceMateriality,
  findPoisoningPremiseDoubt,
  queryConditionalGivenSpan,
  entityNames,
  ctxLabels,
  contextLabelPolarity,
} from './sentences.js';

export const REFUSAL_CODES = [
  'MISSING_BASE_RATE',
  'REVERSED_CONDITIONAL',
  'UNLICENSED_INDEPENDENCE',
  'DUPLICATE_EVIDENCE',
  'CONFLICTING_FIGURES',
  'CONTEXT_MISMATCH',
  'SCOPED_OR_RETRACTED_PREMISE',
  // auxiliary codes (beyond the core 7):
  'MISSING_OBSERVATION_WITNESS',
  'INCOMPLETE_FACTORIZATION',
  'DEGENERATE_PARTITION',
  // PRISM V2 credal + sentence-partition codes:
  'ZERO_SUPPORT_POSSIBLE',
  'UNDERDETERMINED_INTERVAL',
  'UNACCOUNTED_MATERIAL_SENTENCE',
  'POLYTOPE_DIMENSION_LIMIT',
];

export class Refusal extends Error {
  constructor(code, details = {}) {
    super(`REFUSED: ${code}`);
    this.code = code;
    this.details = details;
  }
}

const spansOverlap = (a, b) => a.s < b.e && b.s < a.e;
const ctxOk = (premiseCtx, qctx) => premiseCtx === '*' || premiseCtx === qctx;

function keyOf(d) {
  switch (d.op) {
    case 'PRIOR':
      return `PRIOR|${d.hyp}|${d.ctx}`;
    case 'LIK':
      return `LIK|${d.ev}|${d.pol}|${d.hyp}|${d.ctx}`;
    case 'POSTERIOR_STATED':
      return `POST|${d.hyp}|${d.ev}|${d.pol}|${d.ctx}`;
    default:
      return null;
  }
}

// Index the program's premises.
function indexProgram(prog) {
  const idx = {
    priors: [], liks: [], posts: [], obs: [], condIndeps: [],
    nonAsserted: [], conflicts: [], partitions: [], hyps: [], events: [],
    detLinks: [], joints: [], sentTags: [],
  };
  for (const d of prog.decls) {
    switch (d.op) {
      case 'PRIOR':
        idx.priors.push(d);
        break;
      case 'LIK':
        idx.liks.push(d);
        break;
      case 'POSTERIOR_STATED':
        idx.posts.push(d);
        break;
      case 'OBS':
        idx.obs.push(d);
        break;
      case 'COND_INDEP':
        idx.condIndeps.push(d);
        break;
      case 'DET_LINK':
        idx.detLinks.push(d);
        break;
      case 'JOINT_LIK':
        idx.joints.push(d);
        break;
      case 'SENT_TAG':
        idx.sentTags.push(d);
        break;
      case 'NON_ASSERTED':
        idx.nonAsserted.push(d);
        break;
      case 'CONFLICT':
        idx.conflicts.push(d);
        break;
      case 'PART_EXHAUSTIVE':
        idx.partitions.push(d);
        break;
      case 'HYP':
        idx.hyps.push(d);
        break;
      case 'EVENT':
        idx.events.push(d);
        break;
      default:
        break;
    }
  }
  return idx;
}

function refusal(code, details) {
  return { status: 'REFUSED', code, details };
}

/**
 * solve(program, X) -> decision
 *   {status:'ANSWER', value:{n,d}, target, witnesses:[decls]}
 *   {status:'REFUSED', code, details}
 * Never throws for refusal paths; only throws ParseError for malformed programs.
 */
export function solve(program, X = null) {
  const prog = toProgram(program, X === null ? null : X.length);
  const idx = indexProgram(prog);
  const q = prog.query;
  const qc = q.ctx;

  try {
    const witnesses = [];

    // ---- (2) DUPLICATE_EVIDENCE -------------------------------------------
    const seenObs = new Set();
    for (const o of idx.obs) {
      const k = `${o.ev}`;
      if (seenObs.has(k)) {
        return refusal('DUPLICATE_EVIDENCE', { ev: o.ev, span: o.span });
      }
      seenObs.add(k);
    }
    const seenGiven = new Set();
    for (const g of q.given) {
      if (seenGiven.has(g.ev)) {
        return refusal('DUPLICATE_EVIDENCE', { ev: g.ev, inQuery: true });
      }
      seenGiven.add(g.ev);
    }

    // ---- (3) CONFLICTING_FIGURES: duplicate key, incompatible value -------
    const figSeen = new Map();
    for (const d of [...idx.priors, ...idx.liks, ...idx.posts]) {
      const k = keyOf(d);
      const prev = figSeen.get(k);
      if (prev !== undefined) {
        if (!R.eq(prev.value, d.value)) {
          return refusal('CONFLICTING_FIGURES', {
            key: k,
            values: [R.toString(prev.value), R.toString(d.value)],
            spans: [prev.span, d.span],
          });
        }
        // identical restatement -> dedupe (keep first)
      } else {
        figSeen.set(k, d);
      }
    }
    const priors = idx.priors.filter((d, i) => figSeen.get(keyOf(d)) === d);
    const liks = idx.liks.filter((d) => figSeen.get(keyOf(d)) === d);
    const posts = idx.posts.filter((d) => figSeen.get(keyOf(d)) === d);

    // lookups
    const priorAt = (hyp, ctx) =>
      priors.find((p) => p.hyp === hyp && ctxOk(p.ctx, ctx));
    const priorAnyCtx = (hyp) => priors.find((p) => p.hyp === hyp);
    const likAt = (ev, pol, hyp, ctx) =>
      liks.find((l) => l.ev === ev && l.pol === pol && l.hyp === hyp && ctxOk(l.ctx, ctx));
    const likAnyCtx = (ev, pol, hyp) =>
      liks.find((l) => l.ev === ev && l.pol === pol && l.hyp === hyp);
    const postFor = (hyp, sig, ctx) =>
      posts.find(
        (p) =>
          p.hyp === hyp &&
          sig.length === 1 &&
          p.ev === sig[0].ev &&
          p.pol === sig[0].pol &&
          ctxOk(p.ctx, ctx)
      );
    // Posts matching the query's own signature are direct answers (decided
    // by the credal kernel), not reversed-conditional substitutes — they are
    // excluded here once isDirectPostSig exists (see below; JS consts are
    // evaluated in order so this lookup runs lazily at call time).
    const postConnecting = (evSet, hypSet) =>
      posts.find((p) => hypSet.has(p.hyp) && evSet.has(p.ev) && !isDirectPostSig(p));
    const obsAt = (ev, pol, ctx) => idx.obs.find((o) => o.ev === ev && o.pol === pol && ctxOk(o.ctx, ctx));
    const obsAnyCtx = (ev, pol) => idx.obs.find((o) => o.ev === ev && o.pol === pol);
    const isNonAsserted = (span) => idx.nonAsserted.find((na) => spansOverlap(na.span, span));

    const needCtxCheck = (decl, what) => {
      // decl exists but at incompatible ctx -> CONTEXT_MISMATCH
      if (decl) {
        return refusal('CONTEXT_MISMATCH', { what, haveCtx: decl.ctx, queryCtx: qc, span: decl.span });
      }
      return null;
    };

    // ---- (3b) license-decl validation --------------------------------------
    // DET_LINK / COND_INDEP are licenses, not facts: when X is supplied the
    // bound sentence must actually reference the declared events (by name or
    // a resolvable plural anaphor) and express the claimed relation or
    // affirmative independence. Invalid licenses drop out of the effective
    // program — the event pair then stands unlicensed and the license-bound
    // sentence loses its grounding — they never silently license.
    const droppedLicenses = new Set();
    if (X !== null) {
      const doubt = findPoisoningPremiseDoubt(X, prog);
      if (doubt) {
        return refusal('SCOPED_OR_RETRACTED_PREMISE', {
          span: doubt.span,
          reason: `premise doubt or retraction: "${doubt.marker}"`,
        });
      }
      for (const bad of validateLicenseDecls(X, prog)) droppedLicenses.add(bad.decl);
    }
    const effProg = droppedLicenses.size
      ? { ...prog, decls: prog.decls.filter((d) => !droppedLicenses.has(d)) }
      : prog;
    const licCondIndeps = idx.condIndeps.filter((d) => !droppedLicenses.has(d));
    const licDetLinks = idx.detLinks.filter((d) => !droppedLicenses.has(d));

    // A stated posterior matching the query signature is the direct answer
    // hypothesis (verified through the credal kernel below); any OTHER
    // connecting posterior is the reversed-conditional substitute.
    const isDirectPostSig = (p) =>
      q.targetKind === 'hyp' &&
      p.hyp === q.target &&
      q.given.length === 1 &&
      p.ev === q.given[0].ev &&
      p.pol === q.given[0].pol;
    const directPost = posts.find((p) => isDirectPostSig(p) && ctxOk(p.ctx, qc)) ?? null;

    // ---- sentence ledger (closure — runs at its precedence step and also
    // before any credal-consult release so no path answers past it) --------
    const sentenceCheck = () => {
      if (X === null) return null;
      const account = checkSentenceAccounting(X, effProg);
      if (!account.ok) {
        return refusal(account.code, { detail: account.detail, span: account.span });
      }
      return null;
    };

    // ---- release helper: stated-vs-computed check + witness assembly ------
    const finishAnswer = (credal) => {
      if (q.targetKind === 'hyp') {
        const stated = postFor(q.target, q.given, qc);
        const computed = q.negated ? R.sub(R.one(), credal.value) : credal.value;
        if (stated && !R.eq(stated.value, computed)) {
          return refusal('CONFLICTING_FIGURES', {
            stated: R.toString(stated.value),
            computed: R.toString(computed),
            span: stated.span,
          });
        }
      }
      for (const w of credal.witnesses ?? []) {
        if (!witnesses.includes(w)) witnesses.push(w);
      }
      return {
        status: 'ANSWER',
        value: credal.value,
        target: q.target,
        via: credal.via ?? 'CREDAL_LP',
        interval: credal.interval,
        credal: credal.credal,
        witnesses,
      };
    };

    function gatherPremises() {
      // Resolve the premise set the query needs, at query ctx. Returns
      // {partition, priors, likRows, obsDecls, missing} or a refusal object.
      const givenSig = q.given;
      const evSet = new Set(givenSig.map((g) => g.ev));

      // partition containing the target (hyp targets only)
      let partition = null;
      if (q.targetKind === 'hyp') {
        partition = idx.partitions.find((p) => p.hyps.includes(q.target)) ?? null;
      } else {
        partition = idx.partitions[0] ?? null;
      }
      const missing = { priors: [], liks: [], ctxMismatch: null, obsMissing: [] };

      // obs witnesses
      const obsDecls = [];
      for (const g of givenSig) {
        const o = obsAt(g.ev, g.pol, qc);
        if (o) {
          obsDecls.push(o);
          continue;
        }
        // A given may be witnessed by the query sentence's own explicit
        // conditional clause ("given that the notch is present?") when no
        // separate declarative OBS exists. Ungrounded givens still miss.
        if (X !== null) {
          const cspan = queryConditionalGivenSpan(X, prog, g.ev, g.pol);
          if (cspan) {
            obsDecls.push({
              op: 'OBS', ev: g.ev, pol: g.pol, ctx: qc, span: cspan,
              viaQueryConditional: true,
            });
            continue;
          }
        }
        const other = obsAnyCtx(g.ev, g.pol);
        if (other) {
          missing.ctxMismatch = { what: `OBS ${g.ev}`, decl: other };
          continue;
        }
        // OBS may exist at right ctx but opposite polarity? that is a different
        // observation, not a witness for this one.
        missing.obsMissing.push(g);
      }

      let priorMap = new Map();
      let likRows = [];
      if (partition) {
        for (const h of partition.hyps) {
          const p = priorAt(h, qc);
          if (p) priorMap.set(h, p);
          else {
            const other = priorAnyCtx(h);
            if (other) missing.ctxMismatch = missing.ctxMismatch ?? { what: `PRIOR ${h}`, decl: other };
            else missing.priors.push(h);
          }
        }
        for (const g of givenSig) {
          for (const h of partition.hyps) {
            // NEG givens can be served by an explicit NEG lik or by 1 - POS lik
            let l = likAt(g.ev, g.pol, h, qc);
            let complemented = false;
            if (!l && g.pol === 'NEG') {
              const pos = likAt(g.ev, 'POS', h, qc);
              if (pos) {
                l = pos;
                complemented = true;
              }
            }
            if (l) likRows.push({ ev: g.ev, pol: g.pol, hyp: h, decl: l, complemented });
            else {
              const other = likAnyCtx(g.ev, g.pol, h) ?? likAnyCtx(g.ev, 'POS', h);
              if (other) {
                missing.ctxMismatch = missing.ctxMismatch ?? { what: `LIK ${g.ev} GIVEN ${h}`, decl: other };
              } else {
                missing.liks.push({ ev: g.ev, pol: g.pol, hyp: h });
              }
            }
          }
        }
      }
      // event target also needs LIK(target|h) for every partition hyp —
      // except when the target itself sits in the evidence set, where the
      // query is self-conditioning and no marginalization is consulted.
      const targetLikRows = [];
      const selfGiven = q.targetKind === 'ev' && q.given.some((g) => g.ev === q.target);
      if (q.targetKind === 'ev' && partition && !selfGiven) {
        for (const h of partition.hyps) {
          let l = likAt(q.target, 'POS', h, qc);
          let complemented = false;
          if (!l) {
            const negL = likAt(q.target, 'NEG', h, qc);
            if (negL) {
              l = negL;
              complemented = true;
            }
          }
          if (l) targetLikRows.push({ ev: q.target, hyp: h, decl: l, complemented });
          else {
            const other = likAnyCtx(q.target, 'POS', h) ?? likAnyCtx(q.target, 'NEG', h);
            if (other) {
              missing.ctxMismatch = missing.ctxMismatch ?? { what: `LIK ${q.target} GIVEN ${h}`, decl: other };
            } else {
              missing.liks.push({ ev: q.target, pol: 'POS', hyp: h, target: true });
            }
          }
        }
      }

      return { partition, priorMap, likRows, targetLikRows, obsDecls, missing, evSet };
    }

    // ---- same-event query conditioning ------------------------------------
    // ASK_PROB e GIVEN (..., e:pol, ...) is a tautology — but V2 computes it
    // through the credal kernel like every other query: numerator and
    // denominator are the same world mask, so the Charnes-Cooper LP returns
    // exactly 1/1 (0/1 under NEG polarity). What changed: zero-support
    // detection is no longer a single-factor heuristic. The kernel bounds
    // E(w) over the WHOLE polytope and refuses DEGENERATE_PARTITION when
    // max E(w) = 0 — so an impossible JOINT observation (lamp∧buzzer where
    // every hypothesis forbids the pair) is caught mathematically even
    // though each marginal likelihood is positive.

    const needed = gatherPremises();

    // ---- (4) CONTEXT_MISMATCH ---------------------------------------------
    if (needed.missing.ctxMismatch) {
      return needCtxCheck(needed.missing.ctxMismatch.decl, needed.missing.ctxMismatch.what);
    }
    // A POSTERIOR_STATED that would directly answer the query but lives only
    // at an incompatible CtxId is the same refusal — never reach across ctxs.
    if (!directPost) {
      const otherPost = posts.find((p) => isDirectPostSig(p) && !ctxOk(p.ctx, qc));
      if (otherPost) {
        return refusal('CONTEXT_MISMATCH', {
          what: `POSTERIOR_STATED ${otherPost.hyp} GIVEN ${otherPost.ev}`,
          haveCtx: otherPost.ctx,
          queryCtx: qc,
          span: otherPost.span,
        });
      }
    }

    // ---- (5) MISSING_OBSERVATION_WITNESS (aux) -----------------------------
    if (needed.missing.obsMissing.length > 0) {
      return refusal('MISSING_OBSERVATION_WITNESS', { givens: needed.missing.obsMissing });
    }

    // ---- (5b) source-population context ------------------------------------
    // When X is supplied, premise spans carry population labels ("day shift",
    // "North ward"); an anaphoric query ("this item") binds to the observed
    // item's population. A needed premise labeled for a disjoint population
    // is a context mismatch even when declared CtxIds collapse — and when the
    // needed premises span multiple populations with no observed label to
    // disambiguate, the query population is underdetermined.
    if (X !== null) {
      const names = entityNames(X, prog);
      const excl = new Set(names.map((n) => n.toLowerCase()));
      const labelsOf = (span) => {
        if (!span) return new Set();
        const seg = sentenceAt(X, span.s);
        return seg ? ctxLabels(seg.text, excl) : new Set();
      };
      // the observed population = labels on the observation-witness sentences
      // union labels on the query sentence itself
      const obsLabels = new Set();
      const negObsLabels = new Set();
      const collectObservedContext = (seg) => {
        if (!seg) return;
        const { affirmed, excluded } = contextLabelPolarity(seg.text, excl);
        for (const label of affirmed) obsLabels.add(label);
        for (const label of excluded) negObsLabels.add(label);
      };
      for (const o of needed.obsDecls) {
        if (o.span) collectObservedContext(sentenceAt(X, o.span.s));
      }
      for (const seg of sentences(X)) {
        if (sentenceMateriality(seg, names).kind === 'query') collectObservedContext(seg);
      }
      for (const label of negObsLabels) obsLabels.delete(label);
      const neededPremises = [
        ...needed.priorMap.values(),
        ...needed.likRows.map((r) => r.decl),
        ...needed.targetLikRows.map((r) => r.decl),
        ...(directPost ? [directPost] : []),
      ];
      // License decls obey the same population-compatibility law as the
      // numeric premises they authorize: a DET_LINK/COND_INDEP/JOINT_LIK
      // grounded in a population disjoint from the observed one cannot
      // license the queried pair either ("On the night-shift, ..." never
      // licenses a day-shift observation).
      const needEvs = new Set(q.given.map((g) => g.ev));
      if (q.targetKind === 'ev') needEvs.add(q.target);
      for (const dl of licDetLinks) {
        if (ctxOk(dl.ctx, qc) && needEvs.has(dl.a) && needEvs.has(dl.b)) {
          neededPremises.push(dl);
        }
      }
      for (const ci of licCondIndeps) {
        if (ctxOk(ci.ctx, qc) && ci.evs.filter((e) => needEvs.has(e)).length >= 2) {
          neededPremises.push(ci);
        }
      }
      for (const jl of idx.joints) {
        if (ctxOk(jl.ctx, qc) && jl.spec.filter((s) => needEvs.has(s.ev)).length >= 2) {
          neededPremises.push(jl);
        }
      }
      const disjoint = new Set();
      const distinct = new Set();
      const restricted = new Set();
      const denied = new Set();
      // Ops whose labels are RESTRICTIONS: a DET_LINK / COND_INDEP / LIK
      // carrying a population qualifier applies only inside that
      // population. When the query binds none, the restricted rule can
      // never apply — a single labeled license against an unqualified
      // observation is a mismatch, not a universal.
      const RESTRICTED_OPS = new Set(['DET_LINK', 'COND_INDEP', 'LIK']);
      for (const d of neededPremises) {
        const L = labelsOf(d.span);
        if (L.size === 0) continue;
        const key = [...L].sort().join('|');
        if ([...L].some((l) => negObsLabels.has(l))) denied.add(key);
        if (obsLabels.size > 0) {
          if (![...L].some((l) => obsLabels.has(l))) disjoint.add(key);
        } else {
          distinct.add(key);
          if (RESTRICTED_OPS.has(d.op)) restricted.add(key);
        }
      }
      // Sole-population discourse frame (same doctrine as the framing-
      // subject guard in ctxLabelMentions: scene-setting for the one
      // population a passage ever names must not manufacture a
      // CONTEXT_MISMATCH): when EVERY needed premise carries the same
      // single label set and no sentence in the passage affirms or
      // denies any other population label, that label names the
      // passage's only population and an anaphoric query binds it by
      // discourse. A labeled subset against wider unlabeled premises,
      // or a second population named anywhere, keeps the refusal.
      let solePopulationFrame = false;
      if (obsLabels.size === 0 && denied.size === 0 && disjoint.size === 0 &&
          distinct.size === 1 && neededPremises.length > 0 &&
          neededPremises.every((d) => labelsOf(d.span).size > 0)) {
        const soleLabels = new Set([...distinct][0].split('|'));
        const affirmedUnion = new Set();
        let anyExcluded = false;
        for (const seg of sentences(X)) {
          const pol = contextLabelPolarity(seg.text, excl);
          for (const l of pol.affirmed) affirmedUnion.add(l);
          if (pol.excluded.size > 0) anyExcluded = true;
        }
        solePopulationFrame = !anyExcluded &&
          affirmedUnion.size === soleLabels.size &&
          [...affirmedUnion].every((l) => soleLabels.has(l));
      }
      if (!solePopulationFrame && (denied.size > 0 || disjoint.size > 0 || restricted.size > 0 || (obsLabels.size === 0 && distinct.size > 1))) {
        const shown = obsLabels.size > 0 ? disjoint : restricted.size > 0 ? restricted : distinct;
        return refusal('CONTEXT_MISMATCH', {
          detail:
            denied.size > 0
              ? `premise population(s) {${[...denied].join(',')}} explicitly denied by the observation/question — the queried item is affirmatively not a member`
              : obsLabels.size > 0
                ? `premise population(s) {${[...shown].join(',')}} disjoint from observed population {${[...obsLabels].join(',')}}`
                : restricted.size > 0
                  ? `premise population(s) {${[...restricted].join(',')}} restricted to a population the query binds none of — a context-restricted license cannot apply to an unqualified observation`
                  : `needed premises span ${shown.size} distinct populations; the query binds none`,
          span: neededPremises.find((d) => labelsOf(d.span).size > 0)?.span ?? null,
        });
      }
    }

    // ---- (6) SCOPED_OR_RETRACTED_PREMISE -----------------------------------
    {
      const checkList = [
        ...needed.obsDecls,
        ...[...needed.priorMap.values()],
        ...needed.likRows.map((r) => r.decl),
        ...needed.targetLikRows.map((r) => r.decl),
        ...(directPost ? [directPost] : []),
      ];
      for (const d of checkList) {
        if (d.span) {
          const na = isNonAsserted(d.span);
          if (na) {
            return refusal('SCOPED_OR_RETRACTED_PREMISE', {
              span: d.span, reason: na.reason, premise: d.op,
            });
          }
        }
      }
    }

    // ---- (7) REVERSED_CONDITIONAL ------------------------------------------
    // A consistent POSTERIOR_STATED for the query target itself, matching
    // the single observed given, is a direct answer — never a reversed
    // substitute — even when the passage also states the complementary
    // hypothesis's posterior for the same evidence. Before any reversed
    // refusal, verify every stated posterior for that (ev, pol, ctx): no
    // conflicting duplicates for one hyp, the distinct-hyp sum at most
    // 1/1, and exactly 1/1 when every partition member is stated.
    // Inconsistent stated posteriors refuse CONFLICTING_FIGURES; with no
    // matching stated posterior for the target the reversed refusal
    // below stands untouched.
    const directPostGroupCheck = () => {
      if (!directPost || q.targetKind !== 'hyp' || q.given.length !== 1) return null;
      const g = q.given[0];
      const group = posts.filter(
        (p) => p.ev === g.ev && p.pol === g.pol && ctxOk(p.ctx, qc)
      );
      const byHyp = new Map();
      for (const p of group) {
        const prev = byHyp.get(p.hyp);
        if (prev && !R.eq(prev.value, p.value)) {
          return refusal('CONFLICTING_FIGURES', {
            key: `POST|${p.hyp}|${p.ev}|${p.pol}|${qc}`,
            values: [R.toString(prev.value), R.toString(p.value)],
            spans: [prev.span, p.span],
          });
        }
        if (!prev) byHyp.set(p.hyp, p);
      }
      const distinct = [...byHyp.values()];
      const total = R.sum(distinct.map((p) => p.value));
      if (R.cmp(total, R.one()) > 0) {
        return refusal('CONFLICTING_FIGURES', {
          reason: 'stated posteriors for the same evidence sum above 1/1',
          sum: R.toString(total),
          spans: distinct.map((p) => p.span),
        });
      }
      if (
        needed.partition &&
        needed.partition.hyps.every((h) => byHyp.has(h))
      ) {
        const partTotal = R.sum(
          needed.partition.hyps.map((h) => byHyp.get(h).value)
        );
        if (!R.isOne(partTotal)) {
          return refusal('CONFLICTING_FIGURES', {
            reason: 'stated posteriors cover every partition member but do not sum to 1/1',
            sum: R.toString(partTotal),
            spans: needed.partition.hyps.map((h) => byHyp.get(h).span),
          });
        }
      }
      return null;
    };
    {
      const conflict = directPostGroupCheck();
      if (conflict) return conflict;
    }
    const hypSet = new Set(needed.partition ? needed.partition.hyps : idx.hyps.map((h) => h.id));
    if (
      !directPost &&
      (needed.missing.priors.length > 0 || needed.missing.liks.length > 0 || !needed.partition) &&
      postConnecting(needed.evSet, hypSet, qc)
    ) {
      const post = postConnecting(needed.evSet, hypSet, qc);
      return refusal('REVERSED_CONDITIONAL', {
        span: post.span,
        stated: `POSTERIOR_STATED ${post.hyp} GIVEN ${post.ev}`,
        needed: needed.missing,
      });
    }

    // ---- direct stated posterior -------------------------------------------
    // A POSTERIOR_STATED compiles to the homogeneous row
    // d*P(H,E) - n*P(E) = 0, which every P(E)=0 mass trivially satisfies.
    // With NO priors or likelihoods at the query ctx the credal LP
    // therefore reports eMin = 0 (ZERO_SUPPORT_POSSIBLE) even though
    // P(H|E) is directly stated and E is witnessed. In exactly that
    // pure case the stated rational is itself the certified answer.
    // The moment any PRIOR/LIK/JOINT_LIK premise at the query ctx is in
    // play the credal verdict stands untouched: a stratum forcing
    // P(E)=0 refuses DEGENERATE, a contradicting factorization refuses
    // CONFLICTING, and priors without likelihoods keep refusing
    // ZERO_SUPPORT_POSSIBLE (the V2 replay contract).
    const directPostCertification = () => {
      if (!directPost || q.targetKind !== 'hyp' || q.given.length !== 1) return null;
      const g = q.given[0];
      if (!needed.obsDecls.some((o) => o.ev === g.ev && o.pol === g.pol)) return null;
      if (priors.some((p) => ctxOk(p.ctx, qc))) return null;
      if (liks.some((l) => ctxOk(l.ctx, qc))) return null;
      if (idx.joints.some((j) => ctxOk(j.ctx, qc))) return null;
      const value = q.negated ? R.sub(R.one(), directPost.value) : directPost.value;
      return {
        value,
        via: 'POSTERIOR_STATED',
        interval: { min: value, max: value },
        witnesses: [directPost],
        credal: { eMin: '0/1', directPosterior: true },
      };
    };

    // ---- (8)/(9) incomplete factorization -> credal consult -----------------
    // When the Bayes factorization is incomplete the credal kernel is the
    // decider: deterministic links, stated posteriors and joint masses can
    // still pin the answer uniquely (100% of POSTERIOR_STATED queries route
    // here). The kernel's mathematical verdicts — empty polytope, zero
    // evidence support, possible-zero support — propagate directly. When the
    // kernel merely cannot pin a unique value the structural missing-factor
    // code fires instead.
    const factorsMissing =
      !needed.partition ||
      needed.missing.priors.length > 0 ||
      needed.missing.liks.length > 0;
    if (factorsMissing) {
      const consult = solveCredal(effProg, qc);
      if (consult.status === 'ANSWER') {
        const ledger = sentenceCheck();
        if (ledger) return ledger;
        for (const d of needed.obsDecls) witnesses.push(d);
        return finishAnswer(consult);
      }
      if (consult.status === 'REFUSED' && consult.code === 'ZERO_SUPPORT_POSSIBLE') {
        const cert = directPostCertification();
        if (cert) {
          const ledger = sentenceCheck();
          if (ledger) return ledger;
          for (const d of needed.obsDecls) witnesses.push(d);
          return finishAnswer(cert);
        }
      }
      if (
        consult.status === 'REFUSED' &&
        (consult.code === 'DEGENERATE_PARTITION' ||
          consult.code === 'ZERO_SUPPORT_POSSIBLE' ||
          consult.code === 'CONFLICTING_FIGURES' ||
          consult.code === 'POLYTOPE_DIMENSION_LIMIT')
      ) {
        return refusal(consult.code, consult.details);
      }
    }

    // ---- (8) MISSING_BASE_RATE ---------------------------------------------
    if (!needed.partition) {
      return refusal('MISSING_BASE_RATE', { reason: 'no PART_EXHAUSTIVE partition' });
    }
    if (needed.missing.priors.length > 0) {
      return refusal('MISSING_BASE_RATE', { missingHyps: needed.missing.priors });
    }
    {
      const priorSum = R.sum([...needed.priorMap.values()].map((p) => p.value));
      if (!R.isOne(priorSum)) {
        return refusal('MISSING_BASE_RATE', {
          reason: 'priors do not sum to 1 over partition',
          sum: R.toString(priorSum),
        });
      }
    }

    // ---- (9) INCOMPLETE_FACTORIZATION (aux) --------------------------------
    if (needed.missing.liks.length > 0) {
      return refusal('INCOMPLETE_FACTORIZATION', { missing: needed.missing.liks });
    }

    // ---- (10) UNLICENSED_INDEPENDENCE ---------------------------------------
    // Every event PAIR that must co-occur in the query's evidence or in the
    // marginalization needs a joint-determination license under every
    // partition hyp: a COND_INDEP covering the pair, a DET_LINK relating it
    // (EQUIV/IMPLIES/MUTEX all fix the joint pattern), or a JOINT_LIK
    // declaring the joint mass. Deterministically linked signals ("wired to
    // the same switch") are licensed BY the link — the kernel then computes
    // the exact answer through the constrained polytope rather than
    // assuming independence. Absent any license the honest move is refusal.
    {
      const needEvs = new Set(q.given.map((g) => g.ev));
      if (q.targetKind === 'ev') needEvs.add(q.target);
      const evArr = [...needEvs];
      const pairLicensed = (ea, eb, h) => {
        for (const ci of licCondIndeps) {
          if ((ci.hyp === h || ci.hyp === '*') && ctxOk(ci.ctx, qc) &&
              ci.evs.includes(ea) && ci.evs.includes(eb)) return true;
        }
        for (const dl of licDetLinks) {
          if (ctxOk(dl.ctx, qc) &&
              ((dl.a === ea && dl.b === eb) || (dl.a === eb && dl.b === ea))) return true;
        }
        for (const jl of idx.joints) {
          const s = new Set(jl.spec.map((x) => x.ev));
          if (jl.hyp === h && s.has(ea) && s.has(eb) && ctxOk(jl.ctx, qc)) return true;
        }
        return false;
      };
      for (let i = 0; i < evArr.length; i++) {
        for (let j = i + 1; j < evArr.length; j++) {
          for (const h of needed.partition.hyps) {
            if (!pairLicensed(evArr[i], evArr[j], h)) {
              // the honest interval is still computed — the refusal carries
              // the exact Fréchet bounds the unlicensed pair leaves open
              const details = { hyp: h, evs: [evArr[i], evArr[j]], needed: evArr };
              const consult = solveCredal(effProg, qc);
              if (consult.status === 'ANSWER') {
                details.interval = { min: consult.value, max: consult.value };
              } else if (consult.details && consult.details.interval) {
                details.interval = consult.details.interval;
              }
              return refusal('UNLICENSED_INDEPENDENCE', details);
            }
          }
        }
      }
      for (const d of licCondIndeps) witnesses.push(d);
      for (const d of licDetLinks) witnesses.push(d);
      for (const d of idx.joints) witnesses.push(d);
    }

    // ---- (11a) CONFLICT decl over needed spans ------------------------------
    const neededSpans = [
      ...needed.obsDecls.map((d) => d.span),
      ...[...needed.priorMap.values()].map((d) => d.span),
      ...needed.likRows.map((r) => r.decl.span),
      ...needed.targetLikRows.map((r) => r.decl.span),
    ].filter(Boolean);
    for (const c of idx.conflicts) {
      const aHit = neededSpans.some((sp) => spansOverlap(sp, c.spanA));
      const bHit = neededSpans.some((sp) => spansOverlap(sp, c.spanB));
      if (aHit && bHit) {
        return refusal('CONFLICTING_FIGURES', { spanA: c.spanA, spanB: c.spanB, via: 'CONFLICT decl' });
      }
    }

    // ---- (12) exhaustive sentence-partition account -------------------------
    // Every material sentence of X must be grounded by a decl; a silent
    // material drop ("the lamp and buzzer are wired to the same switch"
    // with no DET_LINK) is dropped evidence and refuses here. The ledger is
    // unconditional whenever X is supplied and is checked against the
    // effective program (invalid licenses ground nothing).
    const ledger = sentenceCheck();
    if (ledger) return ledger;

    // ---- compute: exact-rational credal polytope solve ----------------------
    // Replaces the imperative Star-Bayes calculator. Every decl compiled to
    // linear constraints over atomic worlds; the conditional bounded by a
    // Charnes-Cooper fractional LP. Unique optimum -> exact release;
    // nondegenerate interval -> UNDERDETERMINED_INTERVAL refusal carrying
    // tight Fréchet bounds and a diagnostic cause. 100% of queries —
    // including POSTERIOR_STATED — decide here.
    const credal = solveCredal(effProg, qc);
    if (credal.status === 'REFUSED') {
      if (credal.code === 'ZERO_SUPPORT_POSSIBLE') {
        const cert = directPostCertification();
        if (cert) {
          for (const d of needed.obsDecls) witnesses.push(d);
          for (const p of needed.priorMap.values()) witnesses.push(p);
          for (const r of needed.likRows) witnesses.push(r.decl);
          for (const r of needed.targetLikRows) witnesses.push(r.decl);
          return finishAnswer(cert);
        }
      }
      return { status: 'REFUSED', code: credal.code, details: credal.details };
    }

    // ---- (13) done -----------------------------------------------------------
    for (const d of needed.obsDecls) witnesses.push(d);
    for (const p of needed.priorMap.values()) witnesses.push(p);
    for (const r of needed.likRows) witnesses.push(r.decl);
    for (const r of needed.targetLikRows) witnesses.push(r.decl);

    return finishAnswer(credal);
  } catch (e) {
    if (e instanceof ParseError) throw e;
    throw e;
  }
}
