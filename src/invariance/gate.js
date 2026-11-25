// invariance-gate :: the deterministic B=4 Counterfactual Invariance Verifier.
// (ARCH-SPEC §3.2 — one GPU micro-batch, zero extra serial latency.)
//
//   Slot 1  ORIGINAL            extract(X) -> Program_1 -> solve -> decision
//   Slot 2  LITERAL NUMBER-SWAP permute bound numeric literals -> extract(X')
//           -> ownership invariance law: same proposition keys, values track
//           the literals now sitting at each bound site. Any stale/hallucinated
//           or positional binding -> REFUSED: UNSTABLE_BINDING
//   Slot 3  CONTEXT ABLATION    asserted-head suppression on load-bearing
//           spans -> REFUSED: SCOPED_OR_RETRACTED_PREMISE
//   Slot 4  COMPLEMENT          p(T) + p(NOT T) === 1/1 (exact on factor heads)
//
// The gate never trusts the extractor. It verifies it, counterfactually.

import * as R from '../kernel/rational.js';
import { toProgram, ParseError } from '../kernel/parser.js';
import { solve } from '../kernel/solve.js';
import { literalsInSpan, scanLiterals } from '../kernel/literals.js';
import { applyPermutation, pairwiseSwaps, allPermutations } from './swap.js';
import { ablationCheck, heuristicAssertedHead, sentenceBounds } from './ablation.js';
import { complementCheck } from './complement.js';
import { ownershipCheck, nameMaps, ctxWitnessCheck } from './ownership.js';
import {
  checkSemanticEntailment,
  verifyEntailmentWitness,
  programNeedsEntailment,
  WITNESS_MODEL,
} from './entailment.js';
import {
  assertsIndependence,
  validateLicenseDecls,
} from '../kernel/sentences.js';

export const GATE_CODES = [
  'UNSTABLE_BINDING',
  'UNBOUND_LITERAL',
  'UNBOUND_OBSERVATION',
  'SCOPED_OR_RETRACTED_PREMISE',
  'COMPLEMENT_VIOLATION',
  'CONTEXT_MISMATCH',
  'UNLICENSED_INDEPENDENCE',
  'SEMANTIC_NOT_ENTAILED',
  'ENTAILMENT_WITNESS_MISMATCH',
  'ENTAILMENT_WITNESS_UNAVAILABLE',
];

const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);

function keyOf(d) {
  switch (d.op) {
    case 'PRIOR':
      return `PRIOR|${d.hyp}|${d.ctx}`;
    case 'LIK':
      return `LIK|${d.ev}|${d.pol}|${d.hyp}|${d.ctx}`;
    case 'POSTERIOR_STATED':
      return `POST|${d.hyp}|${d.ev}|${d.pol}|${d.ctx}`;
    default:
      return `${d.op}|${d.id ?? ''}`;
  }
}

// A numeric premise is "bound" iff its span contains a literal equal to its
// declared value — pointer-grounding self-consistency on the original input.
// Returns { bound:[{decl,key,site,literals,siteIdx}], sites:[unique sites
// sorted by position] }. siteIdx indexes into the sorted `sites` array
// (the same array applyPermutation permutes and remaps).
function boundSites(prog, X) {
  const bound = [];
  const siteList = [];
  const siteKey = new Map(); // "s:e" -> index into siteList
  const claimedBy = new Map(); // "s:e" -> proposition key that claimed it
  for (const d of prog.decls) {
    if (!NUMERIC_OPS.has(d.op) || !d.span) continue;
    const lits = literalsInSpan(X, d.span);
    const matches = lits.filter((l) => R.eq(l.value, d.value));
    // A broad span holding two equal literals (coordinated arms) binds each
    // factor to its own site: prefer a site no distinct factor claimed yet.
    const hit =
      matches.find((l) => {
        const owner = claimedBy.get(`${l.s}:${l.e}`);
        return owner === undefined || owner === keyOf(d);
      }) ?? matches[0] ?? null;
    if (hit) claimedBy.set(`${hit.s}:${hit.e}`, keyOf(d));
    let siteIdx = -1;
    if (hit) {
      const k = `${hit.s}:${hit.e}`;
      if (!siteKey.has(k)) {
        siteKey.set(k, siteList.length);
        siteList.push(hit);
      }
      siteIdx = siteKey.get(k);
    }
    bound.push({ decl: d, key: keyOf(d), site: hit, literals: lits, siteIdx });
  }
  const order = siteList.map((_, i) => i).sort((a, b) => siteList[a].s - siteList[b].s);
  const sorted = order.map((i) => siteList[i]);
  const remapIdx = new Map(order.map((old, newI) => [old, newI]));
  for (const b of bound) if (b.siteIdx >= 0) b.siteIdx = remapIdx.get(b.siteIdx);
  return { bound, sites: sorted };
}

/**
 * verifyInvariance({
 *   X, extract, assertedHead?, delta?, fullPermsThreshold?, qualScore?,
 *   entailmentWitness?, requireEntailmentWitness?, offlineSyntheticBench?,
 *   witnessModel?, ollamaHost?, allowCustomWitnessEndpoint?
 * }) -> {
 *   verdict: 'PASS' | 'REFUSED',
 *   code?, decision, slots: {original, swaps:[], ablation, complement, entailment}
 * }
 * extract(X, info) -> bytecode string | AST | JSON program. `info` carries
 * {remappedSites, assign} for counterfactual calls (semantic extractors may
 * use it; untrusted extractors are free to ignore it — the gate re-verifies
 * from the emitted program, not from info).
 *
 * The entailment slot is the fail-closed semantic leg: a would-be PASS only
 * releases behind a SHA-256-bound receipt from the independent Qwen
 * entailment witness — computed here unless a verified `entailmentWitness`
 * is supplied. The ONLY escape is `offlineSyntheticBench === true` (pure
 * synthetic unit benches); `requireEntailmentWitness === false` relaxes the
 * always-on rule to the semantic-trigger set (DET_LINK / COND_INDEP / bound
 * ctx / FLUFF) but never removes it.
 */
// Prototype-pollution-immune option reads (W2): every field — above all
// the witness-escape flags — counts only as an OWN property of the exact
// object passed in. Destructuring defaults would walk Object.prototype.
const hasOwn = Object.hasOwn;
function ownGateOpt(input, key, fallback) {
  if (!input || (typeof input !== 'object' && typeof input !== 'function')) return fallback;
  if (!hasOwn(input, key)) return fallback;
  const v = input[key];
  return v === undefined && key !== 'ollamaHost' ? fallback : v;
}

export async function verifyInvariance(input) {
  const X = ownGateOpt(input, 'X', undefined);
  const extract = ownGateOpt(input, 'extract', undefined);
  const assertedHead = ownGateOpt(input, 'assertedHead', heuristicAssertedHead);
  const delta = ownGateOpt(input, 'delta', 0.4);
  const fullPermsThreshold = ownGateOpt(input, 'fullPermsThreshold', 4);
  const maxSwaps = ownGateOpt(input, 'maxSwaps', null);
  const qualScore = ownGateOpt(input, 'qualScore', null);
  const entailmentWitness = ownGateOpt(input, 'entailmentWitness', null);
  const requireEntailmentWitness = ownGateOpt(input, 'requireEntailmentWitness', true);
  const offlineSyntheticBench = ownGateOpt(input, 'offlineSyntheticBench', false) === true;
  const witnessModel = ownGateOpt(input, 'witnessModel', WITNESS_MODEL);
  const ollamaHost = ownGateOpt(input, 'ollamaHost', undefined);
  const allowCustomWitnessEndpoint = ownGateOpt(input, 'allowCustomWitnessEndpoint', false) === true;
  const slots = { swaps: [], ablation: null, complement: null, entailment: null };
  const failures = [];

  // ---------- Slot 1: original ----------
  let prog1;
  try {
    prog1 = toProgram(await extract(X, { slot: 'original' }), X.length);
  } catch (e) {
    if (e instanceof ParseError) {
      return {
        verdict: 'REFUSED', code: 'PARSE_ERROR',
        details: { message: e.message }, slots, decision: null,
      };
    }
    throw e;
  }
  const decision1 = solve(prog1, X);
  slots.original = { decision: decision1 };

  // id -> name maps for the source-grounded witnesses (HYP/EVENT spans are
  // the pointer-grounding contract: a HYP span covers its name).
  const { nameById, evNameById } = nameMaps(X, prog1);

  // ---------- Slot 2: literal number-swap ownership invariance ----------
  const { bound, sites } = boundSites(prog1, X);
  // slot-1 self-consistency: every declared rational must appear as a literal
  // inside its own bound span (pointer-grounding). A declared value with no
  // matching literal on its span is a hallucinated number.
  for (const b of bound) {
    if (!b.site) {
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail: `declared ${b.decl.op} value ${R.toString(b.decl.value)} not present in bound span`,
        span: b.decl.span,
      });
      slots.swaps.push({ kind: 'self-check', ok: false, key: b.key });
    }
  }
  // ---------- Slot 2a: source-grounded ownership witness -------------------
  // Numeric invariance alone cannot see a persistent misbinding: an extractor
  // that systematically assigns each literal to the wrong owner releases the
  // same wrong answer on the original AND every counterfactual. Re-derive
  // each bound literal's grammatical owner from X (clause containment +
  // governor direction) and refuse mismatches.
  for (const f of ownershipCheck({ X, decls: prog1.decls, nameById, evNameById, prog: prog1 })) {
    failures.push(f);
    slots.swaps.push({ kind: 'ownership', ok: false, detail: f.detail });
  }
  // ---------- Slot 2b: extraction completeness (literal coverage) ----------
  // Every probability-like literal in X must be bound by a numeric decl whose
  // value matches it, or lie inside a NON_ASSERTED span. A live unbound figure
  // is dropped evidence — e.g. the extractor silently picked one of two
  // conflicting figures. Bare integer runs ("320 records", "1998") are exempt:
  // they are counts/labels, not probabilistic premises.
  const probLike = (lit) => !/^\d+$/.test(lit.text);
  const boundSiteKeys = new Set(
    bound.filter((b) => b.site).map((b) => `${b.site.s}:${b.site.e}`)
  );
  const nonAssertedSpans = prog1.decls
    .filter((d) => d.op === 'NON_ASSERTED' && d.span)
    .map((d) => d.span);
  for (const lit of scanLiterals(X)) {
    if (!probLike(lit)) continue;
    if (boundSiteKeys.has(`${lit.s}:${lit.e}`)) continue;
    if (nonAssertedSpans.some((sp) => lit.s >= sp.s && lit.e <= sp.e)) continue;
    failures.push({
      code: 'UNBOUND_LITERAL',
      detail:
        `literal "${lit.text}" at [${lit.s},${lit.e}] bound by no numeric ` +
        'decl and not marked NON_ASSERTED',
      span: { s: lit.s, e: lit.e },
    });
    slots.swaps.push({ kind: 'coverage', ok: false, text: lit.text });
  }

  // ---------- Slot 2c: observation coverage + likelihood direction ----------
  // An event name occurring inside an observation-marked sentence must be
  // ledgered by an OBS decl whose span covers that occurrence — otherwise the
  // extractor dropped an asserted observation (the released answer then rests
  // on partial evidence). And a LIK bound inside a posterior-direction
  // clause ("Among cases where E, q are H") asserts P(H|E), not P(E|H) —
  // the reversed-conditional smuggle.
  const OBS_SENTENCE =
    /\bobserv\w+|\bcame back\b|\bread (?:positive|negative)\b|\bwas recorded\b|\breturned (?:positive|negative)\b|\bshowed (?:positive|negative)\b|\bturned out\b/i;
  // Posterior-direction clause: "Among cases where <ev> ..." or
  // "Among <event-name> cases ...". NOT "Among <hyp-name> cases" — that is a
  // likelihood clause conditioning on the hypothesis.
  const AMONG_WHERE = /\bamong\b[^.]{0,160}?\bwhere\b/i;
  const AMONG_X_CASES = /\bamong\s+([^.]{1,60}?)\s+cases\b/gi;
  const hypNames = prog1.decls
    .filter((d) => d.op === 'HYP' && d.span)
    .map((d) => X.slice(d.span.s, d.span.e).trim().toLowerCase());
  const evNames = prog1.decls
    .filter((d) => d.op === 'EVENT' && d.span)
    .map((d) => X.slice(d.span.s, d.span.e).trim().toLowerCase());
  const posteriorSentence = (sent) => {
    if (AMONG_WHERE.test(sent)) return true;
    for (const m of sent.matchAll(AMONG_X_CASES)) {
      const mod = m[1].toLowerCase();
      const hasEv = evNames.some((n) => n && mod.includes(n));
      const hasHyp = hypNames.some((n) => n && mod.includes(n));
      if (hasEv && !hasHyp) return true;
    }
    return false;
  };
  const obsDecls = prog1.decls.filter((d) => d.op === 'OBS' && d.span);
  const obsSpans = obsDecls.map((d) => d.span);
  // structural half of the same law: a ledgered OBS that the query does not
  // condition on is dropped evidence — the answer silently ignores it.
  const givens = new Set((prog1.query?.given ?? []).map((g) => `${g.ev}:${g.pol}`));
  for (const o of obsDecls) {
    if (!givens.has(`${o.ev}:${o.pol}`)) {
      failures.push({
        code: 'UNBOUND_OBSERVATION',
        detail: `OBS ${o.ev}:${o.pol} ledgered but not conditioned on in the query`,
        span: o.span,
      });
      slots.swaps.push({ kind: 'obs-coverage', ok: false, event: o.ev });
    }
  }
  for (const ev of prog1.decls.filter((d) => d.op === 'EVENT' && d.span)) {
    const name = X.slice(ev.span.s, ev.span.e).trim();
    if (name.length < 2) continue;
    let from = 0;
    let found = 0;
    while (found++ < 64) {
      const i = X.indexOf(name, from);
      if (i < 0) break;
      from = i + 1;
      if (obsSpans.some((sp) => i >= sp.s && i + name.length <= sp.e)) continue;
      const sb = sentenceBounds(X, i);
      const sent = X.slice(sb.s, sb.e);
      if (sent.includes('?')) continue; // a question never asserts an observation
      if (OBS_SENTENCE.test(sent)) {
        failures.push({
          code: 'UNBOUND_OBSERVATION',
          detail:
            `event "${name}" occurs in an observation sentence at ` +
            `[${i},${i + name.length}] but no OBS decl ledgers it`,
          span: { s: i, e: i + name.length },
        });
        slots.swaps.push({ kind: 'obs-coverage', ok: false, event: ev.id });
      }
    }
  }
  // ---------- Slot 2d: license grounding ----------
  // A license (COND_INDEP / DET_LINK) must itself be pointer-grounded AND
  // semantically faithful: its span's sentence must reference every declared
  // event the license relates (by name or a resolvable plural anaphor) and
  // actually express the claimed relation — affirmative independence for
  // COND_INDEP, the declared EQUIV/IMPLIES/MUTEX direction for DET_LINK.
  // "The room was quiet" or an unrelated temperature/humidity clause is a
  // license smuggled past the kernel. assertsIndependence + the span
  // validator are shared with the kernel's own license check.
  for (const d of prog1.decls.filter((d) => d.op === 'COND_INDEP' && d.span)) {
    if (!assertsIndependence(X.slice(d.span.s, d.span.e))) {
      failures.push({
        code: 'UNLICENSED_INDEPENDENCE',
        detail: `COND_INDEP span [${d.span.s},${d.span.e}] does not affirmatively assert independence in the text`,
        span: d.span,
      });
      slots.swaps.push({ kind: 'license', ok: false });
    }
  }
  for (const bad of validateLicenseDecls(X, prog1)) {
    failures.push({
      code: 'UNLICENSED_INDEPENDENCE',
      detail: bad.reason,
      span: bad.decl.span,
    });
    slots.swaps.push({ kind: 'license', ok: false, detail: bad.reason });
  }
  for (const d of prog1.decls.filter((d) => d.op === 'LIK' && d.span)) {
    const sb = sentenceBounds(X, d.span.s);
    const sent = X.slice(sb.s, Math.max(sb.e, d.span.e));
    if (posteriorSentence(sent)) {
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail:
          `LIK ${d.ev}|${d.hyp} bound inside a posterior-direction clause ` +
          `("${sent.slice(0, 80)}...") — literal asserts P(hyp|ev)`,
        span: d.span,
      });
      slots.swaps.push({ kind: 'direction', ok: false, key: keyOf(d) });
    }
  }

  // ---------- Slot 2e: distinct-population ctx witness ----------------------
  // A needed premise whose own sentence grounds it to a population the
  // question does not ask about cannot serve the query — refusing
  // CONTEXT_MISMATCH even when the extractor collapsed distinct CtxIds.
  const excludeLc = new Set(
    [...nameById.values(), ...evNameById.values()].map((n) => n.toLowerCase())
  );
  for (const f of ctxWitnessCheck({ X, decision: decision1, excludeLc })) {
    failures.push(f);
    slots.swaps.push({ kind: 'ctx-witness', ok: false, detail: f.detail });
  }

  if (sites.length >= 2) {
    const assigns = [
      ...pairwiseSwaps(sites),
      ...(sites.length <= fullPermsThreshold
        ? allPermutations(sites.length).filter(
            (a) => !pairwiseSwaps(sites).some((s) => s.every((v, i) => v === a[i]))
          )
        : []),
    ];
    // de-dup permutation assignments
    const seen = new Set();
    let perms = assigns.filter((a) => {
      const k = a.join(',');
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    if (maxSwaps !== null) perms = perms.slice(0, maxSwaps);
    for (const assign of perms) {
      const { X: Xp, remappedSites, remapSpan } = applyPermutation(X, sites, assign);
      let prog2 = null;
      let ok = true;
      let detail = null;
      try {
        prog2 = toProgram(
          await extract(Xp, {
            slot: 'swap',
            remappedSites,
            assign,
            remapSpan,
            // per numeric decl (in P1 declaration order): its bound site index
            declSiteIdx: bound.map((b) => b.siteIdx),
          }),
          Xp.length
        );
      } catch (e) {
        ok = false;
        detail = `extract threw on counterfactual: ${e.message}`;
      }
      if (ok) {
        const keys1 = new Map(bound.filter((b) => b.site).map((b) => [b.key, b]));
        const num2 = prog2.decls.filter((d) => NUMERIC_OPS.has(d.op));
        const keys2 = new Map(num2.map((d) => [keyOf(d), d]));
        // ownership set must be identical
        for (const k of keys1.keys()) {
          if (!keys2.has(k)) {
            ok = false;
            detail = `proposition ${k} vanished under literal permutation`;
          }
        }
        for (const k of keys2.keys()) {
          if (!keys1.has(k)) {
            ok = false;
            detail = `proposition ${k} appeared under literal permutation`;
          }
        }
        if (ok) {
          // values must track the literal now at each bound site
          for (const b of bound) {
            if (!b.site) continue;
            const expected = remappedSites[b.siteIdx].value;
            const d2 = keys2.get(b.key);
            if (!R.eq(d2.value, expected)) {
              ok = false;
              detail =
                `${b.key}: value ${R.toString(d2.value)} did not track literal ` +
                `permutation (expected ${R.toString(expected)})`;
            }
          }
          // pointer-grounding on the counterfactual text
          if (ok) {
            for (const d of num2) {
              const lits = literalsInSpan(Xp, d.span);
              if (!lits.some((l) => R.eq(l.value, d.value))) {
                ok = false;
                detail = `${keyOf(d)}: declared value absent from span on counterfactual`;
              }
            }
          }
          // ownership must stay grammatically consistent on the counterfactual
          if (ok) {
            const o2 = ownershipCheck({ X: Xp, decls: prog2.decls, nameById, evNameById, prog: prog2 });
            if (o2.length > 0) {
              ok = false;
              detail = `counterfactual ownership: ${o2[0].detail}`;
            }
          }
        }
      }
      slots.swaps.push({ assign, ok, detail });
      if (!ok) failures.push({ code: 'UNSTABLE_BINDING', detail, assign });
    }
  }

  // ---------- Slot 3: context ablation on load-bearing spans ----------
  const loadBearing = (decision1.witnesses ?? [])
    .map((d) => d.span)
    .filter(Boolean);
  // also include every NON_ASSERTED-adjacent numeric premise? No — check only
  // the spans the decision actually stands on.
  const abl = ablationCheck({ X, spans: loadBearing, assertedHead, delta });
  slots.ablation = abl;
  if (!abl.ok) {
    failures.push({
      code: 'SCOPED_OR_RETRACTED_PREMISE',
      detail: `context suppresses assertion of span [${abl.span.s},${abl.span.e}] ` +
        `(a_iso=${abl.aIso} a_full=${abl.aFull})`,
      span: abl.span,
    });
  }

  // ---------- Slot 4: complement ----------
  let comp;
  if (qualScore) {
    const p1 = qualScore(X, prog1.query, false);
    const p2 = qualScore(X, prog1.query, true);
    const eps = 1e-6;
    comp =
      p1 === null || p2 === null || Math.abs(p1 + p2 - 1) <= eps
        ? { ok: true, p: p1, pNeg: p2 }
        : { ok: false, code: 'COMPLEMENT_VIOLATION', p: p1, pNeg: p2 };
  } else {
    comp = complementCheck({ program: prog1, X });
  }
  slots.complement = comp;
  if (!comp.ok) failures.push({ code: 'COMPLEMENT_VIOLATION', detail: comp });

  // ---------- Slot 5: independent semantic entailment ----------
  // The cross-model leg. Runs only on a would-be PASS — refusal paths are
  // already closed — and requires a witness receipt bound by SHA-256 to this
  // exact passage and canonical program. A forged English release that
  // survives algebra + deterministic witnesses cannot ship without it.
  if (
    !offlineSyntheticBench &&
    decision1.status === 'ANSWER' &&
    failures.length === 0 &&
    (requireEntailmentWitness !== false || programNeedsEntailment(prog1, X))
  ) {
    const config = { model: witnessModel, host: ollamaHost, allowCustomWitnessEndpoint };
    const witness =
      entailmentWitness ??
      (await checkSemanticEntailment(prog1, X, config));
    slots.entailment = witness;
    const v = verifyEntailmentWitness(witness, prog1, X, config);
    if (!v.ok) failures.push({ code: v.code, detail: v.detail });
  }

  // ---------- verdict ----------
  if (decision1.status === 'REFUSED') {
    return {
      verdict: 'REFUSED',
      code: decision1.code,
      details: decision1.details,
      slots,
      decision: decision1,
      gateFindings: failures,
    };
  }
  if (failures.length > 0) {
    const order = ['UNSTABLE_BINDING', 'UNBOUND_LITERAL', 'UNBOUND_OBSERVATION', 'CONTEXT_MISMATCH', 'UNLICENSED_INDEPENDENCE', 'SCOPED_OR_RETRACTED_PREMISE', 'COMPLEMENT_VIOLATION', 'SEMANTIC_NOT_ENTAILED', 'ENTAILMENT_WITNESS_UNAUTHENTICATED', 'ENTAILMENT_WITNESS_MISMATCH', 'ENTAILMENT_WITNESS_UNAVAILABLE'];
    failures.sort((a, b) => order.indexOf(a.code) - order.indexOf(b.code));
    return {
      verdict: 'REFUSED', code: failures[0].code, details: failures[0].detail,
      slots, decision: decision1, gateFindings: failures,
    };
  }
  return { verdict: 'PASS', slots, decision: decision1 };
}
