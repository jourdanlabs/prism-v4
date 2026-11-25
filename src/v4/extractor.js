// prism-v4 :: Substrate 1 — untrusted local extractor + deterministic
// pointer & literal compiler.
//
// The model emits a minimal JSON factor structure (no `partitionSpan`).
// compileV4(json, X) compiles it into the kernel AST enforcing ONLY
// phrasing-independent structural invariants — verbatim span resolution,
// exact literal/value equality, literal-site uniqueness, NON_ASSERTED
// exclusion, COND_INDEP-vs-DET_LINK deduplication, and stratum-preserving
// ctx normalization. No English-grammar ownership regexes, no
// partitionSpan / canonicalLicense / inert-scene choke-points: every
// semantic ownership claim is deferred to the exact credal solver
// (solve(prog, null)), counterfactual literal swaps, and the sealed
// cross-model witness in src/v4/gate.js + src/v4/witness.js.

import { chat } from '../live/ollama.js';
import { resolveSpanAll, pickObsSpan } from '../live/extractor.js';
import { literalsInSpan, parseLiteral } from '../kernel/literals.js';
import { sentenceBounds } from '../invariance/ablation.js';
import { validateProgram } from '../kernel/parser.js';
import * as R from '../kernel/rational.js';

export const V4_EXTRACTOR_MODEL = 'gemma4:26b';

export class ExtractionError extends Error {}

const SYSTEM = `You are a pointer-grounded factor extractor. Read the passage and emit ONLY a JSON object describing its probabilistic factor graph. Rules:
- Extract every numeric figure faithfully as a reduced fraction "n/d" equal to the written figure (46% -> "23/50", 0.46 -> "23/50", three in ten -> "3/10", two thirds -> "2/3").
- "priors" entries {"hyp","value","span","ctx"}: unconditional marginal shares P(H). "liks" entries {"ev","pol","hyp","value","span","ctx"}: forward conditionals P(E|H). "posteriors" entries {"hyp","ev","pol","value","span","ctx"}: direct reverse conditionals P(H|E) as written — NEVER convert a posterior into a prior or likelihood.
- Every "span" MUST be an exact verbatim substring of the passage containing that figure (for numeric declarations) or that claim (for "obs", "condIndep", "detLinks", "nonAsserted"). For "0.34 and 0.62, respectively" or coordinated clauses, include the minimal verbatim substring containing the figure.
- Emit "condIndep" [{"evs":["e0","e1"],"span","ctx"}] ONLY when the passage explicitly states unconditional independence or rate-invariance across ALL hypotheses (e.g. "are conditionally independent", "equally frequent whether or not", "rate is unchanged when", "same rate with vs without"). Never assume independence when omitted, negated, or restricted to one hypothesis. "evs" must have at least 2 distinct events.
- Emit "detLinks" [{"a","rel","b","span","ctx"}] with rel EQUIV, IMPLIES, or MUTEX ONLY when the passage states a deterministic universal link between two events ("a" and "b" are event IDs, never hyp IDs). Never also emit "condIndep" for the same deterministic-link sentence.
- Emit "nonAsserted" [{"span","reason"}] for any figure or premise that is retracted, disputed, unverified, quoted, or hypothetical; reason is one of QUOTED, RETRACTED, HYPOTHETICAL, UNVERIFIED.
- Set "ctx" consistently: use "base" when the passage has a single population; when the passage explicitly compares two or more distinct strata or groups (e.g. "east room" vs "west room", "spring intake" vs "autumn intake", "day shift" vs "night shift", "north wing" vs "south wing"), give EVERY premise AND the query its exact stratum name.
- "ctx" is ONLY for distinct sub-populations or strata compared in the passage (e.g. "east wing" vs "west wing", "day shift" vs "night shift"). NEVER put a hypothesis name or event name into "ctx". If the passage has only ONE population, ALWAYS emit "ctx": "base" on EVERY prior, lik, posterior, obs, condIndep, detLink, and query.
- "obs" entries {"ev","pol","span","ctx"}: what was actually observed in this case. "query": {"target","given":[{"ev","pol"}],"ctx"} — what the question asks; "given" MUST include every observation in "obs". Conversely, EVERY event listed in "query.given" MUST also have a matching entry in "obs" quoting the observation clause — even when the observation is stated inside the question sentence itself (e.g. quote the part of the question sentence that states the drawn/observed item has the event).
- NEVER invent missing base rates, missing likelihoods, or numeric 0/1 values not written in the text.
Schema: {"hyps":[{"id":"h0","name":"...","span":"..."}],"events":[{"id":"e0","name":"...","span":"..."}],"priors":[],"liks":[],"posteriors":[],"obs":[],"condIndep":[],"detLinks":[],"nonAsserted":[],"partition":["h0","h1"],"query":{"target":"h0","given":[{"ev":"e0","pol":"POS"}],"ctx":"base"}}
Use ids h0,h1,... and e0,e1,... in mention order. If the passage does not support a query answer, still emit the factors faithfully — the solver decides.`;

const FEWSHOT = [
  {
    user: `PASSAGE:
Every relay in the trial was either a repaired unit or a new replacement. Refurbished modules accounted for three in ten of the trial, and new replacements made up the rest at 70%. Among repaired units the amber notch appears in 0.46 of cases; among new replacements it appears in 0.10 of cases. One relay was observed with an amber notch. What is the probability it is a repaired unit?`,
    assistant: JSON.stringify({
      hyps: [
        { id: 'h0', name: 'repaired unit', span: 'either a repaired unit or a new replacement' },
        { id: 'h1', name: 'new replacement', span: 'either a repaired unit or a new replacement' },
      ],
      events: [{ id: 'e0', name: 'amber notch', span: 'the amber notch appears' }],
      priors: [
        { hyp: 'h0', value: '3/10', span: 'accounted for three in ten', ctx: 'base' },
        { hyp: 'h1', value: '7/10', span: 'made up the rest at 70%', ctx: 'base' },
      ],
      liks: [
        { ev: 'e0', pol: 'POS', hyp: 'h0', value: '23/50', span: 'amber notch appears in 0.46', ctx: 'base' },
        { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/10', span: 'it appears in 0.10', ctx: 'base' },
      ],
      posteriors: [],
      obs: [{ ev: 'e0', pol: 'POS', span: 'observed with an amber notch', ctx: 'base' }],
      condIndep: [],
      detLinks: [],
      nonAsserted: [],
      partition: ['h0', 'h1'],
      query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
    }),
  },
];

// ---------- small deterministic helpers (phrasing-independent) ----------

export function normCtxV4(label) {
  const t = String(label ?? '').trim();
  if (!t || /^(base|default|main|ctx0)$/i.test(t)) return 'ctx0';
  let slug = t.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  while (slug.startsWith('ctx_')) slug = slug.slice(4);
  if (!slug || slug === '0') return 'ctx0';
  return `ctx_${slug}`;
}

// Generic words that never make a name token distinctive for the
// hypothesis-echo ctx guard (see compileV4).
const CTX_ECHO_STOPWORDS = new Set([
  'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to',
  'unit', 'units', 'item', 'items', 'lot', 'lots',
  'module', 'modules', 'case', 'cases', 'group', 'groups',
]);

function contentTokensOfName(name) {
  return String(name ?? '')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t && !CTX_ECHO_STOPWORDS.has(t));
}

function ctxSlugTokens(normCtx) {
  if (!normCtx || normCtx === 'ctx0') return [];
  return normCtx.replace(/^ctx_/, '').split('_').filter(Boolean);
}

// Query-embedded observation grounding (see compileV4): distinctive
// content stems of a name or sentence — lowercase tokens of length >= 3
// that are not generic stopwords, conservatively singularized.
const QUERY_TOKEN_STOPWORDS = new Set([
  'a', 'an', 'the', 'of', 'in', 'on', 'for', 'to',
  'is', 'are', 'was', 'were',
]);

function stemToken(t) {
  if (t.length >= 5 && /ies$/.test(t)) return `${t.slice(0, -3)}y`;
  if (/(?:s|x|z|ch|sh)es$/.test(t)) return t.slice(0, -2);
  if (/s$/.test(t) && !/(?:ss|us)$/.test(t)) return t.slice(0, -1);
  return t;
}

function distinctiveStems(text) {
  return new Set(
    String(text ?? '')
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter((t) => t.length >= 3 && !QUERY_TOKEN_STOPWORDS.has(t))
      .map(stemToken)
  );
}

// The query sentence: the sentence containing the final "?", or the
// final sentence of X when the passage has no "?".
function querySentenceSpan(X) {
  const qPos = X.lastIndexOf('?');
  let span = null;
  if (qPos > 0) {
    span = sentenceBounds(X, qPos - 1);
  } else if (X.length > 0) {
    let p = X.length - 1;
    while (p > 0 && /\s/.test(X[p])) p--;
    if (p > 0 && (X[p] === '.' || X[p] === '!')) p--;
    while (p > 0 && /\s/.test(X[p])) p--;
    span = sentenceBounds(X, p);
  }
  if (!span) return null;
  let { s, e } = span;
  while (s < e && /\s/.test(X[s])) s++;
  return s < e ? { s, e } : null;
}

function parseValue(v) {
  if (typeof v !== 'string') return null;
  try {
    return R.parse(v);
  } catch {
    return parseLiteral(v);
  }
}

const isTok = (ch) => !!ch && /[\p{L}\p{N}_-]/u.test(ch);

// Narrow a wide entity span to the unique whole-token occurrence of the
// declared name inside it, so X.slice(span) recovers the name exactly.
function narrowToName(X, span, name) {
  if (typeof name !== 'string' || !name) return span;
  const hits = [];
  let i = X.indexOf(name, span.s);
  while (i >= 0 && i + name.length <= span.e) {
    if (!isTok(X[i - 1]) && !isTok(X[i + name.length])) hits.push(i);
    i = X.indexOf(name, i + 1);
  }
  return hits.length === 1 ? { s: hits[0], e: hits[0] + name.length } : span;
}

// Disambiguate duplicate verbatim occurrences of a numeric span:
// prefer a candidate whose span contains a literal equal to the value,
// then one whose sentence names an owner (hyp/event), then the first.
function pickNumericSpan(X, cands, valueRat, ownerNames) {
  if (cands.length <= 1) return cands[0] ?? null;
  let pool = cands;
  if (valueRat) {
    const withLit = pool.filter((c) =>
      literalsInSpan(X, c).some((l) => R.eq(l.value, valueRat)));
    if (withLit.length > 0) pool = withLit;
    if (pool.length === 1) return pool[0];
  }
  const names = (ownerNames ?? []).filter(Boolean).map((n) => n.toLowerCase());
  if (names.length > 0) {
    const owned = pool.filter((c) => {
      const sb = sentenceBounds(X, c.s);
      const sent = X.slice(sb.s, sb.e).toLowerCase();
      return names.some((n) => sent.includes(n));
    });
    if (owned.length > 0) pool = owned;
  }
  return pool[0];
}

const REASON_MAP = new Map([
  ['QUOTED', 'QUOTED'],
  ['RETRACTED', 'RETRACTED'],
  ['HYPOTHETICAL', 'HYPOTHETICAL'],
  ['HEDGED_APPROX', 'HEDGED_APPROX'],
  // the v4 schema adds UNVERIFIED; the kernel reason set has no such code,
  // so it compiles to the nearest kernel code (span coverage is what
  // excludes the figure — the reason string is diagnostic only)
  ['UNVERIFIED', 'HYPOTHETICAL'],
]);

const polOf = (e, what, errors) => {
  const pol = e?.pol ?? 'POS';
  if (pol !== 'POS' && pol !== 'NEG') {
    errors.push(`${what}: bad polarity ${JSON.stringify(e?.pol)}`);
    return 'POS';
  }
  return pol;
};

/**
 * compileV4(json, X) -> { program, errors, hypNames, evNames, bound }
 * Deterministic pointer & literal compiler. Never throws on model output:
 * every structural failure is collected into `errors` (fail-closed —
 * callers must refuse when errors.length > 0).
 */
export function compileV4(j, X) {
  const errors = [];
  const decls = [];
  const hypNames = {};
  const evNames = {};
  if (!j || typeof j !== 'object' || Array.isArray(j)) {
    return {
      program: { decls: [], query: { op: 'ASK_PROB', target: '', negated: false, given: [], ctx: 'ctx0' } },
      errors: ['extraction is not a JSON object'],
      hypNames, evNames, bound: [],
    };
  }
  if (typeof X !== 'string') {
    return {
      program: { decls: [], query: { op: 'ASK_PROB', target: '', negated: false, given: [], ctx: 'ctx0' } },
      errors: ['passage X is not a string'],
      hypNames, evNames, bound: [],
    };
  }

  // ---- HYP / EVENT entities: valid id + non-empty verbatim name ----------
  const resolveEntity = (entry, what) => {
    const cands = resolveSpanAll(X, entry?.span);
    if (cands.length === 0) {
      errors.push(`unresolvable span for ${what}: ${JSON.stringify(entry?.span)}`);
      return { s: 0, e: 1 };
    }
    return narrowToName(X, cands[0], entry?.name);
  };
  const hypIds = [];
  const seenHyp = new Set();
  for (const h of j.hyps ?? []) {
    if (!/^h\d+$/.test(String(h?.id ?? ''))) {
      errors.push(`invalid HYP id: ${JSON.stringify(h?.id)}`);
      continue;
    }
    if (typeof h.name !== 'string' || !h.name.trim()) {
      errors.push(`HYP ${h.id} has an empty name`);
      continue;
    }
    if (seenHyp.has(h.id)) {
      errors.push(`duplicate HYP "${h.id}"`);
      continue;
    }
    seenHyp.add(h.id);
    hypIds.push(h.id);
    hypNames[h.id] = String(h.name);
    decls.push({ op: 'HYP', id: h.id, span: resolveEntity(h, `HYP ${h.id}`) });
  }
  const evIds = [];
  const seenEv = new Set();
  const eventDecls = [];
  for (const e of j.events ?? []) {
    if (!/^e\d+$/.test(String(e?.id ?? ''))) {
      errors.push(`invalid EVENT id: ${JSON.stringify(e?.id)}`);
      continue;
    }
    if (typeof e.name !== 'string' || !e.name.trim()) {
      errors.push(`EVENT ${e.id} has an empty name`);
      continue;
    }
    if (seenEv.has(e.id)) {
      errors.push(`duplicate EVENT "${e.id}"`);
      continue;
    }
    seenEv.add(e.id);
    evIds.push(e.id);
    evNames[e.id] = String(e.name);
    eventDecls.push({ op: 'EVENT', id: e.id, span: resolveEntity(e, `EVENT ${e.id}`) });
  }

  // ---- Hypothesis-echo ctx guard (LIK / POSTERIOR_STATED only) -----------
  // A model sometimes echoes a hypothesis name into a LIK/POSTERIOR ctx
  // ("Alpha", "Alpha lots") while PRIOR/OBS/query carry the shared
  // population. If a non-ctx0 ctx slug contains a distinctive content
  // token of the decl's OWN hypothesis name (and that token is not also
  // in the query ctx), it is an echo, not a stratum: rebind it to the
  // shared PRIOR/query ctx (or ctx0). Genuine strata ("east room",
  // "spring intake") share no token with a hypothesis name, so
  // multi-stratum CONTEXT_MISMATCH traps are untouched.
  const hypContentTokens = new Map(
    Object.entries(hypNames).map(([id, n]) => [id, contentTokensOfName(n)])
  );
  const rawQueryCtx = j.query?.ctx;
  const queryCtxExplicit =
    rawQueryCtx === undefined || rawQueryCtx === null || String(rawQueryCtx).trim() === ''
      ? null
      : normCtxV4(rawQueryCtx);
  const queryCtxTokens = new Set(
    queryCtxExplicit ? ctxSlugTokens(queryCtxExplicit) : []
  );
  const sharedBaseCtx = (() => {
    const baseSet = new Set(
      (Array.isArray(j.priors) ? j.priors : []).map((p) => normCtxV4(p?.ctx))
    );
    if (queryCtxExplicit) baseSet.add(queryCtxExplicit);
    return baseSet.size === 1 ? [...baseSet][0] : 'ctx0';
  })();
  const guardEchoCtx = (normCtx, hypId) => {
    if (!normCtx || normCtx === 'ctx0') return normCtx;
    const slugTokens = ctxSlugTokens(normCtx);
    if (slugTokens.length === 0) return normCtx;
    const ownTokens = hypContentTokens.get(hypId) ?? [];
    const isEcho = ownTokens.some(
      (t) => slugTokens.includes(t) && !queryCtxTokens.has(t)
    );
    return isEcho ? sharedBaseCtx : normCtx;
  };

  // ---- NON_ASSERTED spans (resolved first: they exclude figures) ---------
  const naDecls = [];
  for (const na of j.nonAsserted ?? []) {
    const cands = resolveSpanAll(X, na?.span);
    if (cands.length === 0) {
      errors.push(`unresolvable span for NON_ASSERTED: ${JSON.stringify(na?.span)}`);
      continue;
    }
    const reason = REASON_MAP.get(String(na?.reason ?? 'QUOTED').toUpperCase());
    if (!reason) {
      errors.push(`bad NON_ASSERTED reason ${JSON.stringify(na?.reason)}`);
      continue;
    }
    naDecls.push({ op: 'NON_ASSERTED', span: cands[0], reason });
  }
  const siteCovered = (site) =>
    naDecls.some((na) => site.s >= na.span.s && site.e <= na.span.e);

  // ---- numeric declarations: value + verbatim span + literal + site ------
  const claimedBy = new Map(); // "s:e" -> factor key
  const bound = [];
  const numericDecls = [];
  const addNumeric = (entry, build, factorKey, what, ownerNames) => {
    const v = parseValue(entry?.value);
    if (!v) {
      errors.push(`${what}: unparseable rational value ${JSON.stringify(entry?.value)}`);
      return;
    }
    if (R.cmp(v, R.zero()) < 0 || R.cmp(v, R.one()) > 0) {
      errors.push(`${what}: value ${R.toString(v)} outside [0,1]`);
      return;
    }
    const cands = resolveSpanAll(X, entry?.span);
    if (cands.length === 0) {
      errors.push(`unresolvable span for ${what}: ${JSON.stringify(entry?.span)}`);
      return;
    }
    const span = pickNumericSpan(X, cands, v, ownerNames);
    const matches = literalsInSpan(X, span).filter((l) => R.eq(l.value, v));
    if (matches.length === 0) {
      errors.push(`${what}: no literal equal to ${R.toString(v)} inside its span`);
      return;
    }
    // literal-site uniqueness: claim a distinct unused matching site
    // (a coordinated span holding two equal literals grounds two factors)
    const site =
      matches.find((l) => {
        const owner = claimedBy.get(`${l.s}:${l.e}`);
        return owner === undefined || owner === factorKey;
      }) ?? null;
    if (!site) {
      errors.push(`${what}: shared literal claimed by distinct factors (${factorKey})`);
      return;
    }
    claimedBy.set(`${site.s}:${site.e}`, factorKey);
    const decl = build(v, span);
    // NON_ASSERTED exclusion: a covered figure never reaches the solver
    if (siteCovered(site)) return;
    numericDecls.push(decl);
    bound.push({ decl, site, key: factorKey });
  };

  for (const p of j.priors ?? []) {
    if (!seenHyp.has(p?.hyp)) {
      errors.push(`PRIOR references undeclared HYP ${JSON.stringify(p?.hyp)}`);
      continue;
    }
    const ctx = normCtxV4(p.ctx);
    addNumeric(
      p,
      (v, span) => ({ op: 'PRIOR', hyp: p.hyp, value: v, ctx, span }),
      JSON.stringify(['PRIOR', p.hyp, ctx]),
      `PRIOR ${p.hyp}`,
      [hypNames[p.hyp]]
    );
  }
  for (const l of j.liks ?? []) {
    if (!seenEv.has(l?.ev) || !seenHyp.has(l?.hyp)) {
      errors.push(`LIK references undeclared ids ev=${JSON.stringify(l?.ev)} hyp=${JSON.stringify(l?.hyp)}`);
      continue;
    }
    const ctx = guardEchoCtx(normCtxV4(l.ctx), l.hyp);
    const pol = polOf(l, `LIK ${l.ev}`, errors);
    addNumeric(
      l,
      (v, span) => ({ op: 'LIK', ev: l.ev, pol, hyp: l.hyp, value: v, ctx, span }),
      JSON.stringify(['LIK', l.ev, pol, l.hyp, ctx]),
      `LIK ${l.ev}`,
      [hypNames[l.hyp], evNames[l.ev]]
    );
  }
  for (const p of j.posteriors ?? []) {
    if (!seenHyp.has(p?.hyp) || !seenEv.has(p?.ev)) {
      errors.push(`POSTERIOR references undeclared ids hyp=${JSON.stringify(p?.hyp)} ev=${JSON.stringify(p?.ev)}`);
      continue;
    }
    const ctx = guardEchoCtx(normCtxV4(p.ctx), p.hyp);
    const pol = polOf(p, `POSTERIOR ${p.hyp}`, errors);
    addNumeric(
      p,
      (v, span) => ({ op: 'POSTERIOR_STATED', hyp: p.hyp, ev: p.ev, pol, value: v, ctx, span }),
      JSON.stringify(['POSTERIOR_STATED', p.hyp, p.ev, pol, ctx]),
      `POSTERIOR ${p.hyp}`,
      [hypNames[p.hyp], evNames[p.ev]]
    );
  }

  // ---- OBS: two-pass verbatim resolution (literal-free sentence wins) ----
  const obsEntries = (j.obs ?? []).filter((o) => {
    if (!seenEv.has(o?.ev)) {
      errors.push(`OBS references undeclared EVENT ${JSON.stringify(o?.ev)}`);
      return false;
    }
    return true;
  });
  const qPos = X.lastIndexOf('?');
  const qSpan = qPos > 0 ? sentenceBounds(X, qPos - 1) : null;
  const obsDecls = new Array(obsEntries.length).fill(null);
  {
    const resolvedSents = [];
    obsEntries.forEach((o, i) => {
      const cands = resolveSpanAll(X, o?.span);
      if (cands.length === 1) {
        obsDecls[i] = cands[0];
        resolvedSents.push(sentenceBounds(X, cands[0].s));
      }
    });
    obsEntries.forEach((o, i) => {
      if (obsDecls[i]) return;
      const cands = resolveSpanAll(X, o?.span);
      if (cands.length === 0) {
        errors.push(`unresolvable span for OBS ${o.ev}: ${JSON.stringify(o?.span)}`);
        obsDecls[i] = { s: 0, e: 1 };
        return;
      }
      obsDecls[i] = pickObsSpan(X, cands, resolvedSents, qSpan) ?? cands[0];
      resolvedSents.push(sentenceBounds(X, obsDecls[i].s));
    });
  }
  const obsOut = obsEntries.map((o, i) => ({
    op: 'OBS', ev: o.ev, pol: polOf(o, `OBS ${o.ev}`, errors),
    ctx: normCtxV4(o.ctx), span: obsDecls[i],
  }));

  // ---- DET_LINK / COND_INDEP licenses -------------------------------------
  const detDecls = [];
  const detKeys = []; // {pairKey, sentStart}
  for (const dl of j.detLinks ?? []) {
    if (!seenEv.has(dl?.a) || !seenEv.has(dl?.b)) {
      errors.push(`DET_LINK endpoints are declared events only: ${JSON.stringify(dl?.a)} ${JSON.stringify(dl?.b)}`);
      continue;
    }
    if (dl.a === dl.b) {
      errors.push(`DET_LINK endpoints must be distinct events: ${dl.a}`);
      continue;
    }
    const rel = String(dl.rel ?? '').toUpperCase();
    if (!['EQUIV', 'IMPLIES', 'MUTEX'].includes(rel)) {
      errors.push(`bad DET_LINK rel ${JSON.stringify(dl.rel)}`);
      continue;
    }
    const cands = resolveSpanAll(X, dl?.span);
    if (cands.length === 0) {
      errors.push(`unresolvable span for DET_LINK: ${JSON.stringify(dl?.span)}`);
      continue;
    }
    const span = cands[0];
    detDecls.push({ op: 'DET_LINK', a: dl.a, rel, b: dl.b, ctx: normCtxV4(dl.ctx), span });
    detKeys.push({
      pairKey: [dl.a, dl.b].sort().join('|'),
      sentStart: sentenceBounds(X, span.s).s,
    });
  }
  const ciDecls = [];
  for (const ci of j.condIndep ?? []) {
    const evs = Array.isArray(ci?.evs) ? ci.evs : [];
    if (evs.length < 2 || new Set(evs).size !== evs.length) {
      errors.push(`COND_INDEP needs >=2 distinct events: ${JSON.stringify(ci?.evs)}`);
      continue;
    }
    if (!evs.every((e) => seenEv.has(e))) {
      errors.push(`COND_INDEP references undeclared events: ${JSON.stringify(evs)}`);
      continue;
    }
    const cands = resolveSpanAll(X, ci?.span);
    if (cands.length === 0) {
      errors.push(`unresolvable span for COND_INDEP: ${JSON.stringify(ci?.span)}`);
      continue;
    }
    const span = cands[0];
    // deduplicate a contradictory COND_INDEP on the exact same sentence
    // and event pair as an emitted DET_LINK (deterministic linkage and
    // independence cannot both be licensed by one sentence)
    if (evs.length === 2) {
      const pairKey = [...evs].sort().join('|');
      const sentStart = sentenceBounds(X, span.s).s;
      if (detKeys.some((d) => d.pairKey === pairKey && d.sentStart === sentStart)) {
        continue;
      }
    }
    ciDecls.push({
      op: 'COND_INDEP', evs: [...evs], hyp: '*', ctx: normCtxV4(ci.ctx), span,
    });
  }

  // ---- partition + query ---------------------------------------------------
  const parts = Array.isArray(j.partition) && j.partition.length
    ? j.partition
    : hypIds;
  for (const id of parts) {
    if (!seenHyp.has(id)) errors.push(`partition references undeclared HYP ${JSON.stringify(id)}`);
  }
  if (new Set(parts).size !== parts.length) errors.push('partition has duplicate hyps');

  const q = j.query ?? {};
  const given = [];
  for (const g of q.given ?? []) {
    if (!seenEv.has(g?.ev)) {
      errors.push(`query given references undeclared EVENT ${JSON.stringify(g?.ev)}`);
      continue;
    }
    given.push({ ev: g.ev, pol: polOf(g, 'query given', errors) });
  }
  if (!seenHyp.has(q.target) && !seenEv.has(q.target)) {
    errors.push(`query target is neither HYP nor EVENT: ${JSON.stringify(q.target)}`);
  }
  // query ctx: an absent label inherits the single shared premise ctx;
  // an explicit label (including "base") is normalized as stated, so two
  // distinct stratum labels are never collapsed
  let qCtx;
  if (q.ctx === undefined || q.ctx === null || String(q.ctx).trim() === '') {
    const premiseCtxs = new Set(
      [...numericDecls, ...obsOut, ...ciDecls, ...detDecls].map((d) => d.ctx)
    );
    qCtx = premiseCtxs.size === 1 ? [...premiseCtxs][0] : 'ctx0';
  } else {
    qCtx = normCtxV4(q.ctx);
  }

  // ---- Query-embedded observation grounding ------------------------------
  // A passage may state the observed event inside the question sentence
  // itself ("One reel that smells of vinegar is drawn at random; what is
  // the probability that it is decaying?"). Extractors then put the event
  // in query.given with obs: [] and solve(prog, null) would refuse
  // MISSING_OBSERVATION_WITNESS. For each given with no covering OBS,
  // ground an OBS on the query sentence ONLY when that sentence actually
  // mentions the event by a distinctive content stem of its declared
  // name. A hallucinated given with no textual witness in the query
  // sentence gets no synthesized OBS and still refuses downstream; the
  // witness's O1 check still audits the exact conditioning against X.
  {
    const synthSpan = querySentenceSpan(X);
    const synthStems = synthSpan
      ? distinctiveStems(X.slice(synthSpan.s, synthSpan.e))
      : null;
    for (const g of given) {
      const covered = obsOut.some(
        (o) => o.ev === g.ev && o.pol === g.pol && (o.ctx === qCtx || o.ctx === 'ctx0')
      );
      if (covered || !synthSpan || !synthStems) continue;
      const nameStems = distinctiveStems(evNames[g.ev] ?? '');
      const mentioned = [...nameStems].some((t) => synthStems.has(t));
      if (!mentioned) continue;
      obsOut.push({ op: 'OBS', ev: g.ev, pol: g.pol, ctx: qCtx, span: synthSpan });
    }
  }

  const firstHypSpan = decls.find((d) => d.op === 'HYP')?.span ?? { s: 0, e: 1 };
  const program = {
    decls: [
      ...decls,
      ...(parts.length ? [{ op: 'PART_EXHAUSTIVE', hyps: [...parts], span: firstHypSpan }] : []),
      ...eventDecls,
      ...numericDecls,
      ...obsOut,
      ...naDecls,
      ...ciDecls,
      ...detDecls,
    ],
    query: {
      op: 'ASK_PROB',
      target: q.target ?? '',
      negated: !!q.negated,
      given,
      ctx: qCtx,
    },
  };
  // Single-population canonicalization: if literally every decl and the
  // query share ONE ctx (zero second stratum anywhere), that label is an
  // arbitrary slug for the sole population — canonicalize it to ctx0 so
  // a counterfactual swap emitting "base" still matches. With >=2
  // distinct ctxs nothing collapses and CONTEXT_MISMATCH is untouched.
  {
    const distinctCtxs = new Set();
    for (const d of program.decls) {
      if (d.ctx !== undefined && d.ctx !== null) distinctCtxs.add(d.ctx);
    }
    if (program.query.ctx !== undefined && program.query.ctx !== null) {
      distinctCtxs.add(program.query.ctx);
    }
    if (distinctCtxs.size === 1) {
      for (const d of program.decls) {
        if (d.ctx !== undefined && d.ctx !== null) d.ctx = 'ctx0';
      }
      program.query.ctx = 'ctx0';
    }
  }
  if (errors.length === 0) {
    try {
      validateProgram(program, X.length);
    } catch (e) {
      errors.push(e.message);
    }
  }
  // bound sites in program declaration order (swap engine input)
  const orderedBound = program.decls
    .map((d) => bound.find((b) => b.decl === d))
    .filter(Boolean);
  return { program, errors, hypNames, evNames, bound: orderedBound };
}

function extractJsonObject(text) {
  const start = text.indexOf('{');
  if (start < 0) throw new ExtractionError('no JSON object in model output');
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  throw new ExtractionError('unbalanced JSON in model output');
}

/**
 * makeV4OllamaExtractor({model, host, calls}) -> async (X, info) => program
 * Untrusted extractor (default gemma4:26b, temperature 0, seed 0,
 * think:false, format json). The returned program carries `hypNames`
 * and `evNames` maps for the gate's semantic keys and witness checks.
 */
export function makeV4OllamaExtractor({ model = V4_EXTRACTOR_MODEL, host, calls = null } = {}) {
  return async (X /*, info */) => {
    const { content } = await chat({
      model,
      host,
      numPredict: 2500,
      think: false,
      seed: 0,
      messages: [
        { role: 'system', content: SYSTEM },
        ...FEWSHOT.flatMap((f) => [
          { role: 'user', content: f.user },
          { role: 'assistant', content: f.assistant },
        ]),
        { role: 'user', content: `PASSAGE:\n${X}` },
      ],
    });
    if (calls) calls.push({ X, content });
    let j;
    try {
      j = JSON.parse(extractJsonObject(content));
    } catch (e) {
      throw new ExtractionError(`model JSON parse failed: ${e.message}`);
    }
    const { program, errors, hypNames, evNames } = compileV4(j, X);
    if (errors.length) throw new ExtractionError(errors.join('; '));
    program.hypNames = hypNames;
    program.evNames = evNames;
    return program;
  };
}

