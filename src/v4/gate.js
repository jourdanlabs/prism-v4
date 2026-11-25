// prism-v4 :: Substrate 2 — exact BigInt credal gate + counterfactual
// literal-swap invariance, orchestrating Substrates 1 and 3.
//
// verifyV4 releases an answer ONLY when all three legs pass:
//   2A exact credal math: solve(prog, null) (pure exact Bayesian/credal
//      algebra, no text accounting) returns ANSWER with lo === hi;
//   2B counterfactual swaps: bound literals are pairwise permuted, the
//      extractor is re-run statelessly on each X', and every bound
//      factor — matched by semantic key op|hypName|evName|pol|ctx, so
//      reordered ids cannot spoof it — must carry the literal value now
//      sitting at its original bound site;
//   3  the sealed cross-model witness (src/v4/witness.js) certifies
//      every semantic claim in the compiled program.
// Any failure refuses immediately with its typed code. Zero LLM math.

import * as R from '../kernel/rational.js';
import { toProgram, ParseError } from '../kernel/parser.js';
import { solve } from '../kernel/solve.js';
import { literalsInSpan } from '../kernel/literals.js';
import { applyPermutation } from '../invariance/swap.js';
import { compileV4, makeV4OllamaExtractor, V4_EXTRACTOR_MODEL } from './extractor.js';
import {
  checkV4SemanticEntailment,
  verifyV4Witness,
  V4_WITNESS_MODEL,
} from './witness.js';

export const V4_GATE_CODES = [
  'EXTRACTION_FAILED',
  'PARSE_ERROR',
  'UNSTABLE_BINDING',
  'SEMANTIC_NOT_ENTAILED',
  'ENTAILMENT_WITNESS_UNAUTHENTICATED',
  'ENTAILMENT_WITNESS_MISMATCH',
  'ENTAILMENT_WITNESS_UNAVAILABLE',
  'UNDERDETERMINED_INTERVAL',
];

const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);
const DEFAULT_WITNESS_HOST = 'http://127.0.0.1:11434';

const hasOwn = Object.hasOwn;
function ownOpt(input, key, fallback) {
  if (!input || (typeof input !== 'object' && typeof input !== 'function')) return fallback;
  if (!hasOwn(input, key)) return fallback;
  const v = input[key];
  return v === undefined ? fallback : v;
}

// Resilient swap-key name normalization: a counterfactual re-extraction
// may emit a morphological variant ("repaired units" vs "repaired
// unit", "fixed-lens" vs "fixed lens", a dropped leading article).
// Lowercase, trim, strip a leading article, fold hyphens/underscores/
// runs of spaces to one space, and conservatively singularize only the
// trailing word.
export function normSwapName(s) {
  let t = String(s ?? '')
    .trim()
    .toLowerCase()
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  t = t.replace(/^(?:a|an|the) /, '');
  if (!t) return t;
  const words = t.split(' ');
  let last = words[words.length - 1];
  if (last.length >= 5 && /ies$/.test(last)) {
    last = `${last.slice(0, -3)}y`;
  } else if (/(?:s|x|z|ch|sh)es$/.test(last)) {
    last = last.slice(0, -2);
  } else if (/s$/.test(last) && !/(?:ss|us)$/.test(last)) {
    last = last.slice(0, -1);
  }
  words[words.length - 1] = last;
  return words.join(' ');
}

// If a program is single-population (every decl + query share one ctx),
// that ctx is an arbitrary sole-population label: swap keys use ctx0
// (compileV4 already canonicalizes raw JSON this way; this keeps
// direct-AST extractors equally resilient). Multi-stratum programs
// (>=2 ctxs) return null and keep the exact ctx in the key.
function singlePopulationCtx(prog) {
  if (!prog || !Array.isArray(prog.decls)) return null;
  const distinct = new Set();
  for (const d of prog.decls) {
    if (d.ctx !== undefined && d.ctx !== null) distinct.add(d.ctx);
  }
  if (prog.query?.ctx !== undefined && prog.query?.ctx !== null) {
    distinct.add(prog.query.ctx);
  }
  return distinct.size === 1 ? 'ctx0' : null;
}

// Semantic factor key: op|hypName|evName|pol|ctx (phrasing- and
// id-order-independent; ids alone can be reordered by a re-extraction).
function semanticKey(d, hypNames, evNames, X, prog) {
  const hypName = (id) =>
    normSwapName(hypNames?.[id] ?? spanTextOf(prog, X, 'HYP', id) ?? id);
  const evName = (id) =>
    normSwapName(evNames?.[id] ?? spanTextOf(prog, X, 'EVENT', id) ?? id);
  const ctx = singlePopulationCtx(prog) ?? d.ctx;
  switch (d.op) {
    case 'PRIOR':
      return `PRIOR|${hypName(d.hyp)}||${ctx}`;
    case 'LIK':
      return `LIK|${hypName(d.hyp)}|${evName(d.ev)}|${d.pol}|${ctx}`;
    case 'POSTERIOR_STATED':
      return `POSTERIOR_STATED|${hypName(d.hyp)}|${evName(d.ev)}|${d.pol}|${ctx}`;
    default:
      return `${d.op}|${d.id ?? ''}`;
  }
}

function spanTextOf(prog, X, op, id) {
  const d = prog.decls.find((x) => x.op === op && x.id === id);
  return d?.span ? X.slice(d.span.s, d.span.e).trim() : null;
}

// Unique bound literal sites of the active numeric decls, in document
// order, each factor claiming a distinct site (coordinated equal
// literals bind distinct sites, exactly as compileV4 claims them).
function boundSitesV4(prog, X) {
  const bound = [];
  const claimedBy = new Map();
  const siteList = [];
  const siteIdxOf = new Map();
  for (const d of prog.decls) {
    if (!NUMERIC_OPS.has(d.op) || !d.span) continue;
    const idKey =
      d.op === 'PRIOR'
        ? `PRIOR|${d.hyp}|${d.ctx}`
        : d.op === 'LIK'
          ? `LIK|${d.ev}|${d.pol}|${d.hyp}|${d.ctx}`
          : `POST|${d.hyp}|${d.ev}|${d.pol}|${d.ctx}`;
    const matches = literalsInSpan(X, d.span).filter((l) => R.eq(l.value, d.value));
    const hit =
      matches.find((l) => {
        const owner = claimedBy.get(`${l.s}:${l.e}`);
        return owner === undefined || owner === idKey;
      }) ?? null;
    if (hit) claimedBy.set(`${hit.s}:${hit.e}`, idKey);
    let siteIdx = -1;
    if (hit) {
      const k = `${hit.s}:${hit.e}`;
      if (!siteIdxOf.has(k)) {
        siteIdxOf.set(k, siteList.length);
        siteList.push(hit);
      }
      siteIdx = siteIdxOf.get(k);
    }
    bound.push({ decl: d, site: hit, siteIdx });
  }
  // sort sites by position (applyPermutation order) and remap indices
  const order = siteList.map((_, i) => i).sort((a, b) => siteList[a].s - siteList[b].s);
  const sorted = order.map((i) => siteList[i]);
  const remap = new Map(order.map((old, n) => [old, n]));
  for (const b of bound) if (b.siteIdx >= 0) b.siteIdx = remap.get(b.siteIdx);
  return { bound, sites: sorted };
}

// Mathematically coherent counterfactual swaps: literals are permuted
// ONLY within their owning factor group (PRIOR / LIK / POSTERIOR_STATED),
// so a swap can never turn "1/3 are A and 2/3 are B" into the incoherent
// "0.9 are A and 2/3 are B" (sum 1.57) that a cross-group PRIOR<->LIK
// swap produces. Preference order: a distinct-value PRIOR pair in one
// ctx (keeps sum(PRIOR) === 1), then LIK pairs for the same event across
// distinct hypotheses, then any other distinct-value LIK pair, then
// POSTERIOR_STATED pairs in one (ev, pol, ctx) group (keeps the stated
// posterior distribution summing to 1). The first pair of each category
// is taken before any second pair, so maxSwaps = 2 typically yields one
// PRIOR swap + one LIK swap. Returns permutation assignments over the
// sorted `sites` array (assign[i] = source site for position i).
export function coherentSwapAssigns(bound, sites, maxSwaps = 2) {
  const ownerOfSite = new Map();
  for (const b of bound) {
    if (b.site && b.siteIdx >= 0 && !ownerOfSite.has(b.siteIdx)) {
      ownerOfSite.set(b.siteIdx, b.decl);
    }
  }
  const idxByOp = { PRIOR: [], LIK: [], POSTERIOR_STATED: [] };
  for (const [idx, d] of ownerOfSite.entries()) {
    if (idxByOp[d.op]) idxByOp[d.op].push(idx);
  }
  for (const k of Object.keys(idxByOp)) idxByOp[k].sort((a, b) => a - b);

  const pairsOf = (idxs, okPair) => {
    const out = [];
    for (let a = 0; a < idxs.length; a++) {
      for (let b = a + 1; b < idxs.length; b++) {
        const i = idxs[a];
        const j = idxs[b];
        if (sites[i].text === sites[j].text) continue;
        if (R.eq(sites[i].value, sites[j].value)) continue;
        if (okPair && !okPair(ownerOfSite.get(i), ownerOfSite.get(j))) continue;
        out.push([i, j]);
      }
    }
    return out;
  };

  const priorPairs = pairsOf(idxByOp.PRIOR, (x, y) => x.ctx === y.ctx);
  const likSameEv = pairsOf(
    idxByOp.LIK,
    (x, y) => x.ev === y.ev && x.hyp !== y.hyp
  );
  const likSameKeys = new Set(likSameEv.map((p) => p.join(':')));
  const likOther = pairsOf(idxByOp.LIK, null).filter(
    (p) => !likSameKeys.has(p.join(':'))
  );
  const postPairs = pairsOf(
    idxByOp.POSTERIOR_STATED,
    (x, y) => x.ctx === y.ctx && x.ev === y.ev && x.pol === y.pol
  );

  const categories = [priorPairs, likSameEv, likOther, postPairs];
  const ordered = [];
  const seenPair = new Set();
  const pushPair = (p) => {
    const k = p.join(':');
    if (seenPair.has(k)) return;
    seenPair.add(k);
    ordered.push(p);
  };
  for (const cat of categories) if (cat.length > 0) pushPair(cat[0]);
  for (const cat of categories) for (const p of cat) pushPair(p);

  const limit =
    typeof maxSwaps === 'number' ? Math.max(0, maxSwaps) : ordered.length;
  return ordered.slice(0, limit).map(([i, j]) => {
    const assign = sites.map((_, k) => k);
    assign[i] = j;
    assign[j] = i;
    return assign;
  });
}

// Program-level convenience (also the unit-test surface for swap choice).
export function coherentSwapsForProgram(prog, X, maxSwaps = 2) {
  const { bound, sites } = boundSitesV4(prog, X);
  return { bound, sites, assigns: coherentSwapAssigns(bound, sites, maxSwaps) };
}

// ---- Unambiguous token-subset name alignment (counterfactual swaps) ----
// A re-extraction on X' may expand/trim a noun modifier ("toxic" vs
// "toxic specimen", "sound cable section" vs "sound section"). Names are
// aligned in three phases — exact normSwapName, then token-subset, then
// shared-token — and a fuzzy match is accepted ONLY when it is mutually
// unique (strict 1-to-1). Any ambiguity or leftover fails closed.
const ALIGN_STOPWORDS = new Set([
  'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to',
  'is', 'are', 'was', 'were',
]);

function alignTokens(normName) {
  return new Set(
    String(normName ?? '')
      .split(' ')
      .filter((t) => t && !ALIGN_STOPWORDS.has(t))
  );
}

function isTokenSubset(a, b) {
  if (a.size === 0) return false;
  for (const t of a) if (!b.has(t)) return false;
  return true;
}

// names1/names2: Map id -> normSwapName(name). Returns Map id2 -> id1
// covering every id, or null when no unambiguous bijection exists.
export function alignEntityNames(names1, names2) {
  if (names1.size !== names2.size) return null;
  const map = new Map();
  const used1 = new Set();
  const matchPhase = (relation) => {
    let progress = true;
    while (progress) {
      progress = false;
      for (const [id1, n1] of names1) {
        if (used1.has(id1)) continue;
        const cands = [];
        for (const [id2, n2] of names2) {
          if (!map.has(id2) && relation(n1, n2)) cands.push(id2);
        }
        if (cands.length !== 1) continue;
        const id2 = cands[0];
        let back = 0;
        for (const [oid1, on1] of names1) {
          if (!used1.has(oid1) && relation(on1, names2.get(id2))) back++;
        }
        if (back !== 1) continue;
        map.set(id2, id1);
        used1.add(id1);
        progress = true;
      }
    }
  };
  matchPhase((a, b) => a !== '' && a === b);
  matchPhase((a, b) => {
    const ta = alignTokens(a);
    const tb = alignTokens(b);
    return isTokenSubset(ta, tb) || isTokenSubset(tb, ta);
  });
  matchPhase((a, b) => {
    const ta = alignTokens(a);
    const tb = alignTokens(b);
    for (const t of ta) if (tb.has(t)) return true;
    return false;
  });
  return map.size === names1.size ? map : null;
}

function entityNameMap(prog, names, X, op, ids) {
  const m = new Map();
  for (const id of ids) {
    m.set(id, normSwapName(names?.[id] ?? spanTextOf(prog, X, op, id) ?? id));
  }
  return m;
}

function referencedIds(decls, field) {
  const s = new Set();
  for (const d of decls) if (d[field]) s.add(d[field]);
  return s;
}

// Normalize whatever an extractor returns into a compiled bundle:
// raw v4 JSON (compiled here), a compileV4 result, a program AST
// (optionally carrying hypNames/evNames), or bytecode text.
function normalizeExtracted(raw, X) {
  if (raw && typeof raw === 'object' && (Array.isArray(raw.hyps) || Array.isArray(raw.events))) {
    const c = compileV4(raw, X);
    return { program: c.program, errors: c.errors, hypNames: c.hypNames, evNames: c.evNames };
  }
  if (raw && typeof raw === 'object' && raw.program && Array.isArray(raw.program.decls)) {
    return {
      program: raw.program,
      errors: raw.errors ?? [],
      hypNames: raw.hypNames ?? raw.program.hypNames ?? {},
      evNames: raw.evNames ?? raw.program.evNames ?? {},
    };
  }
  if (typeof raw === 'string' || (raw && Array.isArray(raw.decls) && raw.query)) {
    const program = toProgram(raw, X.length);
    const hypNames = {};
    const evNames = {};
    if (raw && typeof raw === 'object') {
      Object.assign(hypNames, raw.hypNames ?? {});
      Object.assign(evNames, raw.evNames ?? {});
    }
    for (const d of program.decls) {
      if (d.op === 'HYP' && hypNames[d.id] === undefined) {
        hypNames[d.id] = spanTextOf(program, X, 'HYP', d.id) ?? d.id;
      }
      if (d.op === 'EVENT' && evNames[d.id] === undefined) {
        evNames[d.id] = spanTextOf(program, X, 'EVENT', d.id) ?? d.id;
      }
    }
    return { program, errors: [], hypNames, evNames };
  }
  throw new Error('extractor returned an unrecognized result');
}

/**
 * verifyV4({ X, extract?, witnessCheck?, maxSwaps = 2, model?,
 *            witnessModel?, ollamaHost?, allowCustomWitnessEndpoint? })
 *   -> { verdict, code, decision, value, rational, interval, swaps,
 *        witness, compiled, elapsedMs }
 * verdict === 'PASS' only when the exact credal solve (lo === hi),
 * counterfactual swaps, and the sealed witness ALL pass.
 * `extract` defaults to makeV4OllamaExtractor({ model, host }).
 * `witnessCheck` defaults to checkV4SemanticEntailment and must be an
 * async (program, X, config) -> sealed receipt function.
 */
export async function verifyV4(input) {
  const started = Date.now();
  const X = ownOpt(input, 'X', undefined);
  const model = ownOpt(input, 'model', V4_EXTRACTOR_MODEL);
  const witnessModel = ownOpt(input, 'witnessModel', V4_WITNESS_MODEL);
  const ollamaHost = ownOpt(input, 'ollamaHost', undefined);
  const maxSwaps = ownOpt(input, 'maxSwaps', 2);
  const allowCustomWitnessEndpoint =
    ownOpt(input, 'allowCustomWitnessEndpoint', false) === true;
  const extract =
    ownOpt(input, 'extract', null) ??
    makeV4OllamaExtractor({ model, host: ollamaHost });
  const witnessCheck = ownOpt(input, 'witnessCheck', null) ?? checkV4SemanticEntailment;

  const refused = (code, rest = {}) => ({
    verdict: 'REFUSED',
    code,
    decision: rest.decision ?? null,
    value: rest.value ?? null,
    rational: rest.value ? R.toString(rest.value) : null,
    interval: rest.interval ?? null,
    swaps: rest.swaps ?? [],
    witness: rest.witness ?? null,
    compiled: rest.compiled ?? null,
    ...(rest.details !== undefined ? { details: rest.details } : {}),
    elapsedMs: Date.now() - started,
  });

  // ---- Substrate 1: extract + compile --------------------------------------
  let compiled;
  try {
    compiled = normalizeExtracted(await extract(X, { slot: 'original' }), X);
  } catch (e) {
    if (e instanceof ParseError) {
      return refused('PARSE_ERROR', { details: { message: e.message } });
    }
    return refused('EXTRACTION_FAILED', { details: { message: e.message } });
  }
  if (compiled.errors.length > 0) {
    return refused('EXTRACTION_FAILED', {
      compiled,
      details: { errors: compiled.errors },
    });
  }
  const prog = compiled.program;

  // ---- Step 2A: exact credal math (zero LLM arithmetic) --------------------
  let decision;
  try {
    decision = solve(prog, null);
  } catch (e) {
    if (e instanceof ParseError) {
      return refused('PARSE_ERROR', { compiled, details: { message: e.message } });
    }
    throw e;
  }
  if (decision.status !== 'ANSWER') {
    return refused(decision.code, {
      compiled,
      decision,
      interval: decision.details?.interval ?? null,
    });
  }
  const value = decision.value;
  const interval = decision.interval ?? { min: value, max: value };
  if (
    !R.eq(interval.min, interval.max) ||
    !R.eq(value, interval.min) ||
    R.cmp(value, R.zero()) < 0 ||
    R.cmp(value, R.one()) > 0
  ) {
    return refused('UNDERDETERMINED_INTERVAL', { compiled, decision, value, interval });
  }

  // ---- Step 2B: counterfactual literal-swap invariance ---------------------
  const { bound, sites } = boundSitesV4(prog, X);
  for (const b of bound) {
    if (!b.site) {
      return refused('UNSTABLE_BINDING', {
        compiled, decision, value, interval,
        details: { detail: `declared ${b.decl.op} value ${R.toString(b.decl.value)} not present in bound span` },
      });
    }
  }
  const keys1 = new Map(
    bound.map((b) => [semanticKey(b.decl, compiled.hypNames, compiled.evNames, X, prog), b])
  );
  const swaps = [];
  // Coherent within-group swaps only (see coherentSwapAssigns): never a
  // cross-group PRIOR<->LIK swap that breaks sum(PRIOR) === 1. With no
  // distinct-value within-group pair (e.g. a single literal) there are
  // zero swaps and verification proceeds straight to the witness.
  const assigns = sites.length >= 2 ? coherentSwapAssigns(bound, sites, maxSwaps) : [];

  for (const assign of assigns) {
    const { X: Xp, remappedSites, remapSpan } = applyPermutation(X, sites, assign);
    let ok = true;
    let detail = null;
    let compiled2 = null;
    try {
      compiled2 = normalizeExtracted(
        await extract(Xp, {
          slot: 'swap',
          remappedSites,
          assign,
          remapSpan,
          declSiteIdx: bound.map((b) => b.siteIdx),
        }),
        Xp
      );
      if (compiled2.errors.length > 0) {
        ok = false;
        detail = `counterfactual compile failed: ${compiled2.errors[0]}`;
      }
    } catch (e) {
      ok = false;
      detail = `extract threw on counterfactual: ${e.message}`;
    }
    if (ok) {
      const prog2 = compiled2.program;
      const num2 = prog2.decls.filter((d) => NUMERIC_OPS.has(d.op));
      // Align hyp/event names between prog and prog2 (exact normSwapName
      // first, then unambiguous token-subset/shared-token bijection) and
      // key prog2 factors by prog's canonical names. Ambiguous or
      // unmatched names fail closed; the covariant value check below is
      // then exactly as strict as before (R.eq on every proposition).
      const num1Decls = bound.map((b) => b.decl);
      const hypNames1 = entityNameMap(
        prog, compiled.hypNames, X, 'HYP', referencedIds(num1Decls, 'hyp')
      );
      const evNames1 = entityNameMap(
        prog, compiled.evNames, X, 'EVENT', referencedIds(num1Decls, 'ev')
      );
      const hypAlign = alignEntityNames(
        hypNames1,
        entityNameMap(prog2, compiled2.hypNames, Xp, 'HYP', referencedIds(num2, 'hyp'))
      );
      const evAlign = alignEntityNames(
        evNames1,
        entityNameMap(prog2, compiled2.evNames, Xp, 'EVENT', referencedIds(num2, 'ev'))
      );
      let keys2 = new Map();
      if (!hypAlign || !evAlign) {
        ok = false;
        detail =
          'hypothesis/event names vanished or ambiguous under literal permutation';
      } else {
        const ctxOf2 = (d) => singlePopulationCtx(prog2) ?? d.ctx;
        const keyOf2 = (d) => {
          const hypName = hypNames1.get(hypAlign.get(d.hyp));
          const evName = d.ev ? evNames1.get(evAlign.get(d.ev)) : '';
          switch (d.op) {
            case 'PRIOR':
              return `PRIOR|${hypName}||${ctxOf2(d)}`;
            case 'LIK':
              return `LIK|${hypName}|${evName}|${d.pol}|${ctxOf2(d)}`;
            case 'POSTERIOR_STATED':
              return `POSTERIOR_STATED|${hypName}|${evName}|${d.pol}|${ctxOf2(d)}`;
            default:
              return `${d.op}|${d.id ?? ''}`;
          }
        };
        keys2 = new Map(num2.map((d) => [keyOf2(d), d]));
        if (keys2.size !== num2.length) {
          ok = false;
          detail = 'duplicate proposition under literal permutation';
        }
      }
      if (ok) {
        for (const k of keys1.keys()) {
          if (!keys2.has(k)) {
            ok = false;
            detail = `proposition ${k} vanished under literal permutation`;
            break;
          }
        }
      }
      if (ok) {
        for (const k of keys2.keys()) {
          if (!keys1.has(k)) {
            ok = false;
            detail = `proposition ${k} appeared under literal permutation`;
            break;
          }
        }
      }
      if (ok) {
        for (const [k, b] of keys1.entries()) {
          const expected = remappedSites[b.siteIdx].value;
          const d2 = keys2.get(k);
          if (!R.eq(d2.value, expected)) {
            ok = false;
            detail =
              `${k}: value ${R.toString(d2.value)} did not track literal ` +
              `permutation (expected ${R.toString(expected)})`;
            break;
          }
        }
      }
      if (ok) {
        for (const d of num2) {
          if (!literalsInSpan(Xp, d.span).some((l) => R.eq(l.value, d.value))) {
            ok = false;
            detail = 'declared value absent from span on counterfactual';
            break;
          }
        }
      }
    }
    swaps.push({ assign, ok, detail });
    if (!ok) {
      return refused('UNSTABLE_BINDING', {
        compiled, decision, value, interval, swaps,
        details: { detail, assign },
      });
    }
  }

  // ---- Substrate 3: sealed cross-model witness -----------------------------
  const witnessHost = ollamaHost ?? DEFAULT_WITNESS_HOST;
  const config = {
    model: witnessModel,
    host: witnessHost,
    allowCustomWitnessEndpoint,
    hypNames: compiled.hypNames,
    evNames: compiled.evNames,
  };
  let witness;
  try {
    witness = await witnessCheck(prog, X, config);
  } catch (e) {
    return refused('ENTAILMENT_WITNESS_UNAVAILABLE', {
      compiled, decision, value, interval, swaps,
      details: { detail: `witness check threw: ${e.message}` },
    });
  }
  const v = verifyV4Witness(witness, prog, X, config);
  if (!v.ok) {
    return refused(v.code, {
      compiled, decision, value, interval, swaps, witness,
      details: { detail: v.detail },
    });
  }

  return {
    verdict: 'PASS',
    code: null,
    decision,
    value,
    rational: value ? R.toString(value) : null,
    interval,
    swaps,
    witness,
    compiled,
    elapsedMs: Date.now() - started,
  };
}
