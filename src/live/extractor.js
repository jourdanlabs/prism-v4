// live eval :: constrained factor extraction — drives an open-weights model
// (Ollama) into emitting a JSON factor structure, then compiles it into
// pointer-grounded bytecode with real span offsets resolved against X.
// The model is the UNTRUSTED extractor: the kernel + gate verify it.

import { chat } from './ollama.js';
import { parseLiteral, scanLiterals, literalAtSpanForValue, literalsInSpan } from '../kernel/literals.js';
import { sentenceBounds } from '../invariance/ablation.js';
import {
  ctxLabels, autoSentTags, expandCtxExclude, sentences,
  assertsIndependence, premiseDoubt, DEPENDENCY_WORDS,
} from '../kernel/sentences.js';
import { nameMaps, ownershipCheck, coordinatedArmForSite } from '../invariance/ownership.js';
import * as R from '../kernel/rational.js';

const SYSTEM = `You are a pointer-grounded factor extractor. Read the passage and emit ONLY a JSON object describing its probabilistic factor graph. Rules:
- Every numeric figure you record MUST include "span": the exact verbatim substring of the passage that contains that figure (include a few surrounding words for uniqueness, max ~12 words).
- "value" is a rational string "n/d" equal to the figure AS WRITTEN (e.g. "80%" -> "4/5", "five eighths" -> "5/8", "0.625" -> "5/8", "one in ten" -> "1/10", "three cards in ten" -> "3/10", "7 out of every 20" -> "7/20").
- "priors" entries: {"hyp","value","span"} — unconditional shares/base rates. NEVER emit a prior for a subgroup share that is conditional on an event ("Of the units with a green mark, 7/11 came from Alpha" is P(Alpha | green mark), a posterior — emit it under "posteriors", never under "priors"). One source figure can ground only one factor: never bind the same figure as both a prior and a posterior. Every numeric factor MUST ground an explicit numeric figure written in the passage (percentage, fraction, decimal, ratio, or number word); NEVER synthesize "0" or "1" for qualitative universal prose ("every ...", "none ...", "all ...") that carries no numeric figure.
- "liks" entries: {"ev","pol","hyp","value","span"} — P(evidence pol | hypothesis). pol is "POS" (event occurs) or "NEG". Every "value" MUST ground an explicit numeric figure written in the passage; NEVER synthesize "0" or "1" for qualitative universal prose ("every ...", "none ...", "all ...") with no numeric figure.
- "posteriors" entries: {"hyp","ev","pol","value","span"} — statements of P(hyp | evidence) AS WRITTEN. Use this when the passage gives the conditional in the posterior direction (e.g. "among flagged cases, 80% are Alpha"). Do NOT convert it into a likelihood. Every "value" MUST ground an explicit numeric figure written in the passage; NEVER synthesize "0" or "1" for qualitative prose with no numeric figure.
- "obs" entries: {"ev","pol","span"} — what was actually observed in this case.
- "nonAsserted": [{"span","reason"}] for figures that are retracted, quoted, hypothetical, or hedged-approximate; reason is one of QUOTED, RETRACTED, HYPOTHETICAL, HEDGED_APPROX. NEVER mark an affirmative equal-rate independence statement (e.g. "the buzzer rate is the same for panels with a lamp as for panels without a lamp", "the probability of the buzzer is identical whether the lamp lights or not") as nonAsserted/HYPOTHETICAL — it is an asserted independence license; emit it under "condIndep".
- "condIndep": [{"evs":["e0","e1"],"span"}] — ONLY when the text explicitly asserts the events are independent (e.g. "recorded independently") or states the equal-conditional-rate definition (one event's rate/probability is "the same"/"identical" with vs without the other, or "whether <other> occurs or not"). "evs" MUST contain at least two distinct events (["e0","e1",...]); NEVER emit a 1-event condIndep. Never assume it. Never emit it for negated wording ("not independent", "not the same", "differs" are NOT licenses). The span MUST cover a clause that names or anaphorically refers to BOTH events and asserts their independence.
- "detLinks": [{"a","rel","b","span"}] — deterministic linkage between two events: rel is "EQUIV" (always co-occur — "wired to the same switch", "fire together", "share a relay"), "IMPLIES" (a firing forces b — "whenever a fires, b fires"), or "MUTEX" (never both). "a" and "b" are EVENT ids only (e0, e1, ...) — NEVER pass a hypothesis id (h0, h1, ...) into detLinks. The span MUST cover a clause that names or anaphorically refers to BOTH events and actually expresses that relation — never bind it to unrelated prose. A deterministic implication sentence (e.g. "Every unit that has a star also has a notch") is ONLY a "detLinks" (IMPLIES) entry, NEVER a "condIndep".
- "partition": the exhaustive list of hypothesis ids.
- "partitionSpan": OPTIONAL — include it ONLY by copying one complete sentence verbatim when it is exactly "Every/Each <one unqualified noun> is either <hyp name> or <hyp name>." naming the two partition hyps in either order; otherwise omit it entirely. "one of" or "mutually exclusive" wording does not qualify.
- "hyps" entries: when such an explicit partition sentence appears, use that same sentence as each hyp's "span"; otherwise use the smallest verbatim span naming that hyp.
- Event "name" values: use the short exact noun phrase from the text (e.g. "flag", "flag test", "amber notch"): plain alphabetic/hyphenated words, never a clause, never a paraphrase, and never containing negation or logical words (not, no, never, or, if, unless, except, without, and, but).
- Every PRIOR/LIK "span" must be the smallest verbatim clause that, in its own source clause, contains that one figure with its stated owner (and, for a LIK, its event) exactly as written there. Do not borrow an owner or event from another clause, do not synthesize or reorder a swapped surface form, and never reuse a span that also holds another figure. One exception: when a single sentence coordinates two events' rates under one hypothesis ("... a notch on 1/4 and a stripe on 1/4"), each LIK may quote either its own arm or that whole sentence — but each arm's figure belongs to its own event exactly as written; never swap the arms.
- "query": {"target","given":[{"ev","pol"}],"ctx"} — what the question asks. "given" MUST include every observation you ledgered in "obs".
- "ctx": every prior/lik/posterior/obs/condIndep/detLink AND the query MUST carry a "ctx" naming the population/stratum its figures belong to, using the passage's own words (e.g. "South", "North", "day shift", "1998 manual"). Use "base" when the passage has one undifferentiated population. Premises about a DIFFERENT population than the question keeps are still emitted faithfully with their own ctx — the solver decides.
Schema: {"hyps":[{"id","name","span"}],"events":[{"id","name","span"}],"priors":[],"liks":[],"posteriors":[],"obs":[],"nonAsserted":[],"condIndep":[],"detLinks":[],"partition":[],"query":{}}
Use ids h0,h1,... and e0,e1,... in mention order. If the passage does not support a query answer, still emit the factors faithfully — the solver decides.`;

const FEWSHOT = [
  {
    user: `PASSAGE:
The audit partitions cases into disjoint cohorts. Alpha accounts for 4/5 of cases and Beta for 1/5. When the case is Alpha, the flag test is positive 9/10 of the time; when it is Beta, 1/10. A case was observed: the flag test came back positive. What is the probability the case is Alpha?`,
    assistant: JSON.stringify({
      hyps: [
        { id: 'h0', name: 'Alpha', span: 'Alpha accounts for 4/5' },
        { id: 'h1', name: 'Beta', span: 'Beta for 1/5' },
      ],
      events: [{ id: 'e0', name: 'flag test', span: 'the flag test is positive' }],
      priors: [
        { hyp: 'h0', value: '4/5', span: 'Alpha accounts for 4/5 of cases' },
        { hyp: 'h1', value: '1/5', span: 'Beta for 1/5' },
      ],
      liks: [
        { ev: 'e0', pol: 'POS', hyp: 'h0', value: '9/10', span: 'is Alpha, the flag test is positive 9/10' },
        { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/10', span: 'it is Beta, 1/10' },
      ],
      posteriors: [],
      obs: [{ ev: 'e0', pol: 'POS', span: 'the flag test came back positive' }],
      nonAsserted: [],
      partition: ['h0', 'h1'],
      query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
    }),
  },
  {
    user: `PASSAGE:
The obsolete 1998 manual, since retracted, states the following base rates. Alpha accounts for 3/4; Beta for 1/4. When Alpha, the flag test fires 1/2; when Beta, 1/4. A case was observed: the flag test fired. What is the probability the case is Alpha?`,
    assistant: JSON.stringify({
      hyps: [
        { id: 'h0', name: 'Alpha', span: 'Alpha accounts for 3/4' },
        { id: 'h1', name: 'Beta', span: 'Beta for 1/4' },
      ],
      events: [{ id: 'e0', name: 'flag test', span: 'the flag test fires' }],
      priors: [
        { hyp: 'h0', value: '3/4', span: 'Alpha accounts for 3/4' },
        { hyp: 'h1', value: '1/4', span: 'Beta for 1/4' },
      ],
      liks: [
        { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: 'the flag test fires 1/2' },
        { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: 'when Beta, 1/4' },
      ],
      posteriors: [],
      obs: [{ ev: 'e0', pol: 'POS', span: 'the flag test fired' }],
      nonAsserted: [
        {
          span: 'The obsolete 1998 manual, since retracted, states the following base rates. Alpha accounts for 3/4; Beta for 1/4',
          reason: 'RETRACTED',
        },
      ],
      partition: ['h0', 'h1'],
      query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }] },
    }),
  },
];

export class ExtractionError extends Error {}

// resolve a model-emitted span substring to [s,e) offsets in X.
// tries exact, then whitespace-insensitive, then ellipsis-fragment search.
// Returns ALL candidate spans; callers disambiguate by context.
export function resolveSpan(X, spanText) {
  const cands = resolveSpanAll(X, spanText);
  return cands[0] ?? null;
}

export function resolveSpanAll(X, spanText) {
  if (!spanText || typeof spanText !== 'string') return [];
  const out = [];
  const push = (s, e) => {
    if (!out.some((c) => c.s === s && c.e === e)) out.push({ s, e });
  };
  let i = -1;
  while ((i = X.indexOf(spanText, i + 1)) >= 0) push(i, i + spanText.length);
  if (out.length) return out;
  // ellipsis / "..." spans: resolve each fragment, span covers first..last
  const frags = spanText.split(/\.\.\.|…/).map((f) => f.trim()).filter(Boolean);
  if (frags.length >= 2) {
    const spans = frags.map((f) => resolveSpan(X, f));
    if (spans.every(Boolean)) {
      push(spans[0].s, spans[spans.length - 1].e);
      return out;
    }
  }
  const squeeze = (s) => s.replace(/\s+/g, ' ');
  const needle = squeeze(spanText.trim());
  const hay = squeeze(X);
  i = hay.indexOf(needle);
  if (i >= 0) {
    // map squeezed offset back — approximate: find first word occurrence
    const firstWord = needle.split(' ')[0];
    const orig = X.indexOf(firstWord);
    if (orig >= 0) {
      // walk: find the needle's start word at orig, compute end
      let count = 0;
      const target = needle.length;
      let end = orig;
      while (end < X.length && count < target) {
        if (!/\s/.test(X[end]) || (count > 0 && /\s/.test(needle[count - 1]))) count++;
        end++;
      }
      push(orig, Math.min(X.length, orig + spanText.length));
    }
  }
  return out;
}

// Pick among duplicate span occurrences. Disambiguation order:
//   1. candidate whose sentence names the decl's owner (hypothesis/event)
//   2. candidate whose sentence contains the decl's bound literal value
//   3. first occurrence
// This is what stops two decls emitted with identical span text
// ("and a flash with probability 60%") from both resolving to the FIRST
// occurrence and silently misbinding an owner.
function pickSpan(X, cands, valueRat, ownerName = null) {
  if (cands.length <= 1) return cands[0] ?? null;
  const sentOf = (c) => {
    const sb = sentenceBounds(X, c.s);
    return X.slice(sb.s, sb.e);
  };
  if (ownerName) {
    const nm = ownerName.toLowerCase();
    const owned = cands.filter((c) => sentOf(c).toLowerCase().includes(nm));
    if (owned.length > 0) {
      cands = owned;
      if (cands.length === 1) return cands[0];
    }
  }
  if (valueRat) {
    for (const c of cands) {
      for (const lit of scanLiterals(sentOf(c))) {
        if (R.eq(lit.value, valueRat)) return c;
      }
    }
  }
  return cands[0];
}

// Pick among duplicate OBS quote occurrences. An observation quote
// ("and a stamped lid") can repeat verbatim inside an earlier numeric
// rate sentence; blindly taking the first occurrence would bind the
// OBS into the rate sentence and strand the real observation. Order:
//   1. candidates in a literal-free sentence (an observation quote
//      never jumps into a numeric rate sentence when an exact match
//      exists in a literal-free observation sentence)
//   2. candidates in the same sentence as an already-resolved OBS
//   3. non-query candidates closest before the query sentence
// Every candidate is a verbatim occurrence in X; nothing is synthesized.
export function pickObsSpan(X, cands, resolvedObsSents = [], qSpan = null) {
  if (!cands || cands.length === 0) return null;
  if (cands.length === 1) return cands[0];
  const sentBoundsOf = (c) => sentenceBounds(X, c.s);
  const sentTextOf = (c) => {
    const sb = sentBoundsOf(c);
    return X.slice(sb.s, sb.e);
  };
  let pool = cands;
  const litFree = pool.filter(
    (c) => !scanLiterals(sentTextOf(c)).some((l) => !/^\d+$/.test(l.text))
  );
  if (litFree.length > 0) pool = litFree;
  if (pool.length === 1) return pool[0];
  if (resolvedObsSents && resolvedObsSents.length > 0) {
    const sameSent = pool.filter((c) =>
      resolvedObsSents.some((rs) => c.s >= rs.s && c.s < rs.e)
    );
    if (sameSent.length > 0) pool = sameSent;
  }
  if (pool.length === 1) return pool[0];
  const nonQuery = pool.filter((c) => !sentTextOf(c).includes('?'));
  const pref = nonQuery.length > 0 ? nonQuery : pool;
  if (qSpan) {
    const before = pref
      .filter((c) => c.s < qSpan.s)
      .sort((a, b) => b.s - a.s);
    if (before.length > 0) return before[0];
  }
  return pref[0] ?? cands[0];
}

// Source-level token test: Unicode letters/digits plus '_' and '-'
// continue a token, so a name never matches inside a longer word.
const isTok = (ch) => !!ch && /[\p{L}\p{N}_-]/u.test(ch);

// Exact whole-token occurrences of `name` within X[s,e). Boundaries
// are read from X itself, so a quote that starts or ends inside a
// word cannot manufacture a boundary at its edge. Empty,
// whitespace-only, or whitespace-padded names never match.
function tokenHitsIn(X, s, e, name) {
  if (typeof name !== 'string' || !name.trim() || name.trim() !== name) return [];
  const hits = [];
  let i = X.indexOf(name, s);
  while (i >= 0 && i + name.length <= e) {
    if (!isTok(X[i - 1]) && !isTok(X[i + name.length])) hits.push(i);
    i = X.indexOf(name, i + 1);
  }
  return hits;
}

// Distinct declared names occurring as whole tokens in X[s,e), or
// null when the name list itself is unsafe (empty, whitespace-
// padded, or duplicated names) — callers fail closed on null.
function distinctIn(X, s, e, names) {
  if (!Array.isArray(names) || !names.length) return null;
  if (names.some((n) => typeof n !== 'string' || !n.trim() || n.trim() !== n)) return null;
  if (new Set(names).size !== names.length) return null;
  return names.filter((n) => tokenHitsIn(X, s, e, n).length > 0);
}

// Narrow a HYP/EVENT decl's wide span to its own name, inside the
// resolved EXACT quote only (no global search, no trust in a fuzzy
// or partial occurrence): the name must occur exactly once as a
// whole token there. Otherwise keep the original span unchanged.
function narrowNameSpan(X, span, spanText, name) {
  if (typeof spanText !== 'string' || X.slice(span.s, span.e) !== spanText) return span;
  const hits = tokenHitsIn(X, span.s, span.e, name);
  return hits.length === 1 ? { s: hits[0], e: hits[0] + name.length } : span;
}

// First exact whole-token occurrence of `name` anywhere in X, or null.
// Non-numeric grounding fallback ONLY (HYP / EVENT names): a numeric
// declaration never resolves through here.
function nameOccurrenceSpan(X, name) {
  const hits = tokenHitsIn(X, 0, X.length, name);
  return hits.length ? { s: hits[0], e: hits[0] + name.length } : null;
}

// EVENT canonical-name preference: when the resolved exact quote names the
// event by a hyphenated synonym ("amber-shaped notch") but the declared
// name itself occurs verbatim in X ("amber notch"), ground the EVENT on
// that exact verbatim occurrence so the recurring canonical name is what
// every later witness matches. Only on an exact resolved quote; otherwise
// the ordinary narrow-or-keep result stands.
function canonicalEventSpan(X, wide, spanText, name) {
  const narrowed = narrowNameSpan(X, wide, spanText, name);
  if (typeof spanText !== 'string' || X.slice(wide.s, wide.e) !== spanText) return narrowed;
  if (typeof name === 'string' && name && X.slice(narrowed.s, narrowed.e) !== name) {
    return nameOccurrenceSpan(X, name) ?? narrowed;
  }
  return narrowed;
}

// NON_ASSERTED leading-article snap: a quote that fails exact resolution
// ONLY because its leading article differs ("A ..." for the text's
// "The ...") snaps to the one sentence in X whose post-article body is
// verbatim the quote's post-article body (at least 20 chars; one terminal
// '.' may be omitted). Zero or multiple such sentences fail closed.
function articleSnapSpan(X, txt) {
  if (typeof txt !== 'string') return null;
  const qt = txt.trim();
  const qm = /^(a|an|the)\s+/i.exec(qt);
  if (!qm) return null;
  const stripDot = (s) => (s.endsWith('.') ? s.slice(0, -1) : s);
  const qBody = stripDot(qt.slice(qm[0].length));
  if (qBody.length < 20) return null;
  const cands = [];
  for (const seg of sentences(X)) {
    const t = seg.text.trim();
    const sm = /^(a|an|the)\s+/i.exec(t);
    if (!sm) continue;
    if (sm[1].toLowerCase() === qm[1].toLowerCase()) continue;
    if (stripDot(t.slice(sm[0].length)) !== qBody) continue;
    const s = X.indexOf(t, seg.s);
    if (s >= 0) cands.push({ s, e: s + t.length });
  }
  return cands.length === 1 ? cands[0] : null;
}

// The optional binary partition proof supports only conservative
// single lexical atoms: a Unicode letter, then bounded letters/digits,
// optionally hyphen-joined to further letter-led components. Logical
// connective/quantifier words are never atoms, so a multiword string
// naming a third member can never be swallowed as one hyp name.
const PART_ATOM_SRC = String.raw`\p{L}(?:[\p{L}\p{N}]{0,30})(?:-\p{L}(?:[\p{L}\p{N}]{0,30}))*`;
const PART_BAD_WORDS = new Set([
  'and', 'or', 'either', 'neither', 'nor', 'both', 'not', 'no',
  'every', 'each', 'is', 'are', 'all', 'any', 'one', 'of',
]);
function isPartitionAtom(name) {
  if (typeof name !== 'string' || !name || name.length > 32 || name.trim() !== name) return false;
  if (!new RegExp(`^${PART_ATOM_SRC}$`, 'u').test(name)) return false;
  return name.split('-').every((p) => !PART_BAD_WORDS.has(p.toLowerCase()));
}

// Validate an explicitly supplied partitionSpan. Accepted only when
// its single exact source occurrence IS the whole actual sentence
// containing its first character (trimmed equality; the sentence's
// terminal '.' may be omitted from the quote, but '?'/'!' sentences
// are rejected, never stripped) and that sentence matches the narrow
// grammar "Every/Each <single unqualified noun> is either <name> or
// <name>" naming the two distinct declared partition hyps in either
// order. Fragments, "one of", "mutually exclusive", negated or longer
// sentences return null and the caller fails closed.
// Each permitted atom must additionally be source-grounded in its OWN
// supplied HYP quote: the compiled (narrowed) HYP span for that id
// slices exactly to the asserted name, lies inside the wide span, and
// that wide span is verbatim the id's exact proposed quote (which may
// be the shared exhaustive sentence or a prior sentence naming it).
function partitionSpanOf(X, txt, parts, hypIds, nameOf, hypProof) {
  if (typeof txt !== 'string' || !txt.trim()) return null;
  if (!Array.isArray(parts) || parts.length !== 2 || new Set(parts).size !== 2) return null;
  if (!Array.isArray(hypIds) || hypIds.length !== 2 || new Set(hypIds).size !== 2) return null;
  if (!parts.every((id) => hypIds.includes(id))) return null;
  const names = parts.map((id) => nameOf.get(id));
  if (!names.every(isPartitionAtom)) return null;
  if (new Set(names.map((n) => n.toLowerCase())).size !== 2) return null;
  if (!(hypProof instanceof Map)) return null;
  for (let k = 0; k < 2; k++) {
    const src = hypProof.get(parts[k]);
    const nm = names[k];
    if (!src || typeof src.quote !== 'string' || !src.wide || !src.span) return null;
    if (X.slice(src.wide.s, src.wide.e) !== src.quote) return null;
    if (src.span.s < src.wide.s || src.span.e > src.wide.e) return null;
    if (X.slice(src.span.s, src.span.e) !== nm) return null;
    const own = tokenHitsIn(X, src.wide.s, src.wide.e, nm);
    if (own.length !== 1 || own[0] !== src.span.s) return null;
  }
  const qt = txt.trim();
  const hits = [];
  let i = -1;
  while ((i = X.indexOf(qt, i + 1)) >= 0) hits.push(i);
  if (hits.length !== 1) return null;
  const s = hits[0], e = s + qt.length;
  const sb = sentenceBounds(X, s);
  if (s < sb.s || e > sb.e) return null;
  const sent = X.slice(sb.s, sb.e).trim();
  if (/[?!]/.test(sent)) return null;
  if (sent !== qt && sent !== `${qt}.`) return null;
  const core = sent.endsWith('.') ? sent.slice(0, -1) : sent;
  const m = new RegExp(
    `^(?:Every|Each)\\s+\\p{L}+\\s+is\\s+either\\s+(${PART_ATOM_SRC})\\s+or\\s+(${PART_ATOM_SRC})$`,
    'u',
  ).exec(core);
  if (!m) return null;
  if (!isPartitionAtom(m[1]) || !isPartitionAtom(m[2])) return null;
  const inOrder = m[1] === names[0] && m[2] === names[1];
  const swapped = m[1] === names[1] && m[2] === names[0];
  return inOrder || swapped ? { s, e } : null;
}

// Explicit binary partition sentence auto-anchoring (R6 Fix 3a): when
// the model omits partitionSpan and grounds both HYPs in a later rate
// sentence, the passage's own exhaustive-partition sentence ("All its
// valves are brass or zinc.", "Alpha and Beta are the only workshops
// that serviced the cart.") is the true PART_EXHAUSTIVE anchor. The
// match is a strict anchored binary grammar over one complete,
// comma-free, literal-free, doubt-free, dependency-free sentence
// naming both distinct partition atoms (either order, whole words)
// and no declared event. Only a UNIQUE such sentence anchors; zero or
// several fall back to the first HYP span unchanged. This only moves
// the anchor of an already-emitted PART_EXHAUSTIVE — it never creates
// one and never overrides an explicitly supplied partitionSpan.
function findExplicitBinaryPartitionSpan(X, parts, nameOf, eventNames) {
  if (!Array.isArray(parts) || parts.length !== 2 || new Set(parts).size !== 2) return null;
  const n0 = nameOf.get(parts[0]);
  const n1 = nameOf.get(parts[1]);
  if (!isPartitionAtom(n0) || !isPartitionAtom(n1)) return null;
  if (n0.toLowerCase() === n1.toLowerCase()) return null;
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pairForms = (a, b) => [
    new RegExp(
      `^(?:All|Every|Each)\\s+(?:(?:of\\s+)?(?:the|its|their|these|those|our)\\s+)?` +
      `[a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?\\s+(?:are|is)\\s+(?:either\\s+)?${a}\\s+or\\s+${b}\\.?$`,
      'i'
    ),
    new RegExp(
      `^${a}\\s+and\\s+${b}\\s+are\\s+the\\s+only\\s+[a-z][a-z-]*` +
      `(?:\\s+[a-z][a-z-]*){0,8}\\.?$`,
      'i'
    ),
  ];
  const pats = [...pairForms(esc(n0), esc(n1)), ...pairForms(esc(n1), esc(n0))];
  const hits = [];
  for (const seg of sentences(X)) {
    const lead = seg.text.length - seg.text.trimStart().length;
    const t = seg.text.trim();
    if (!t || /[?!]/.test(t) || /[,;]/.test(t)) continue;
    if (scanLiterals(t).some((l) => !/^\d+$/.test(l.text))) continue;
    if (premiseDoubt(t) || DEPENDENCY_WORDS.test(t)) continue;
    if (/\b(?:relays?|switch(?:es)?|wires?|wiring|circuits?|links?|linkage|couplings?|mechanisms?)\b/i.test(t)) continue;
    if ((eventNames ?? []).some((n) => n && new RegExp(`\\b${esc(n)}\\b`, 'i').test(t))) continue;
    if (!pats.some((p) => p.test(t))) continue;
    hits.push({ s: seg.s + lead, e: seg.s + lead + t.length });
  }
  return hits.length === 1 ? hits[0] : null;
}

// The sole complete numeric literal of the EXACT candidate quote, as
// a span in X, or null. Requires exactly one scanned literal in the
// quote in total (not merely one equal among extra rates), and that
// literal must match an actual full-passage scanned literal in both
// absolute half-open span and exact rational value, so a quote that
// cuts a longer source literal (120%, a decimal, a fraction or a
// ratio) is rejected rather than read as the true source literal.
function sourceLiteralIn(X, c, valueRat, fullLits = scanLiterals(X), localLits = scanLiterals(X.slice(c.s, c.e))) {
  if (!Array.isArray(localLits) || localLits.length !== 1 || !R.eq(localLits[0].value, valueRat)) return null;
  if (!Array.isArray(fullLits)) return null;
  const lit = localLits[0];
  const gs = c.s + lit.s, ge = c.s + lit.e;
  const full = fullLits.find((f) => f.s === gs && f.e === ge);
  if (!full || full.text !== lit.text || !R.eq(full.value, valueRat)) return null;
  return { s: gs, e: ge };
}

// The quote's sole literal IMMEDIATELY governs `name`: between the
// literal's END and the whole name's start only whitespace, "of",
// whitespace may occur (e.g. "20% of Jade"). An earlier percentage
// somewhere in a prefix ending in "of" is not sufficient.
function literalOfName(X, c, valueRat, name, lit = null, fullLits = undefined) {
  const hits = tokenHitsIn(X, c.s, c.e, name);
  if (hits.length !== 1) return false;
  if (!lit) lit = sourceLiteralIn(X, c, valueRat, fullLits);
  if (!lit) return false;
  return /^\s+of\s+$/u.test(X.slice(lit.e, hits[0]));
}

// Duplicate-span disambiguation for LIK decls ONLY and only when the
// emitted span text occurs more than once (all other decls and all
// unique occurrences keep pickSpan unchanged). A candidate is
// eligible only when: it is the EXACT quote (any fuzzy/partial
// candidate fails the whole pick — there is no fuzzy fallback and
// never a default to the first duplicate); the quote itself carries
// exactly one literal equal to the bound value (a number elsewhere
// in the passage never licenses it); the quote lies wholly in the
// sentence containing its first character (c.e <= that sentence's
// end); that sentence names the cited hyp as its only hyp, or — on
// a dual-owner sentence — the quote binds the rate to the cited hyp
// as "<literal> of <hyp>" and names no other hyp; and the
// ';'-clause around the quote names the cited event as its only
// event, so an event mentioned only in a different clause cannot be
// picked. Exactly one candidate must survive.
// Safe residual failures (rejected rather than guessed): duplicate
// LIKs whose clause does not name the event, dual-owner sentences
// whose locally-read rate is not phrased "<literal> of <owner>",
// and two decls claiming the same literal where the quote gives
// neither decl an unambiguous owner.
function pickLikSpan(X, spanText, cands, valueRat, lik) {
  const { hypName, evName, hypNames, evNames } = lik;
  if (!valueRat || typeof spanText !== 'string' || !spanText) return null;
  if (!Array.isArray(hypNames) || !hypNames.includes(hypName)) return null;
  if (!Array.isArray(evNames) || !evNames.includes(evName)) return null;
  if (spanText.includes(';')) return null;
  const onlyLit = scanLiterals(spanText);
  // A quote carries exactly one literal (the ordinary case) or exactly two
  // (a broad coordinated-arms quote); anything else fails closed.
  if (!Array.isArray(onlyLit) || onlyLit.length < 1 || onlyLit.length > 2) return null;
  if (!onlyLit.some((l) => R.eq(l.value, valueRat))) return null;
  const fullLits = scanLiterals(X);
  const eligible = [];
  for (const c of cands) {
    if (X.slice(c.s, c.e) !== spanText) return null;
    // Coordinated two-arm attribution: the candidate's sentence states two
    // events' rates under one hypothesis in "and"-joined arms. Eligible
    // exactly when an arm literal equal to the bound value attributes to
    // this decl's event + hypothesis (distinct sites, even at equal values).
    if (lik.evId && lik.hypId) {
      const sites = literalsInSpan(X, c).filter((l) => R.eq(l.value, valueRat));
      const coordinated = sites.some((site) => {
        const coord = coordinatedArmForSite(X, site, lik.evEntries, lik.hypEntries);
        return coord && coord.evId === lik.evId && coord.hypId === lik.hypId;
      });
      if (coordinated) {
        eligible.push(c);
        continue;
      }
      if (onlyLit.length === 2) continue; // two-literal quotes bind only via arms
    }
    const litSpan = sourceLiteralIn(X, c, valueRat, fullLits, onlyLit);
    if (!litSpan) continue;
    const sb = sentenceBounds(X, c.s);
    if (c.s < sb.s || c.e > sb.e) continue;
    const sentHyps = distinctIn(X, sb.s, sb.e, hypNames);
    const quoteHyps = distinctIn(X, c.s, c.e, hypNames);
    if (!sentHyps || !quoteHyps) continue;
    const sent = X.slice(sb.s, sb.e);
    const before = sent.lastIndexOf(';', Math.max(0, c.s - sb.s - 1));
    const semi = sent.indexOf(';', c.e - sb.s);
    const clauseS = sb.s + before + 1;
    const clauseE = semi < 0 ? sb.e : sb.s + semi;
    const clauseHyps = distinctIn(X, clauseS, clauseE, hypNames);
    if (!clauseHyps) continue;
    const singleOwner = clauseHyps.length === 1 && clauseHyps[0] === hypName;
    const dualOwner = clauseHyps.length > 1 && clauseHyps.includes(hypName)
      && quoteHyps.length === 1 && quoteHyps[0] === hypName
      && literalOfName(X, c, valueRat, hypName, litSpan);
    if (!singleOwner && !dualOwner) continue;
    const clauseEvs = distinctIn(X, clauseS, clauseE, evNames);
    if (clauseEvs && clauseEvs.length === 1 && clauseEvs[0] === evName) eligible.push(c);
  }
  return eligible.length === 1 ? eligible[0] : null;
}

function rat(v) {
  if (typeof v === 'number') {
    // model gave a float — convert to nearest simple rational
    const s = String(v);
    if (/^\d+\.\d+$/.test(s)) {
      const dec = s.split('.')[1];
      return R.make(BigInt(s.replace('.', '')), 10n ** BigInt(dec.length));
    }
    return R.make(BigInt(v), 1n);
  }
  const t = String(v).trim();
  if (/^-?\d+\s*\/\s*\d+$/.test(t)) return R.parse(t.replace(/\s+/g, ''));
  if (/^-?\d+$/.test(t)) return R.make(BigInt(t), 1n);
  const lit = parseLiteral(t);
  if (lit) return lit;
  throw new ExtractionError(`bad rational "${v}"`);
}

// Normalize a model-emitted ctx label into a CtxId. 'base'/missing -> 'ctx0';
// any other label is its own population: 'ctx_<slug>'. Distinct labels must
// NEVER collapse — ctx labels are how distinct populations stay distinct.
function normCtx(label) {
  const t = String(label ?? '').trim();
  if (!t || /^(base|default|main|general|population|ctx0?)$/i.test(t)) return 'ctx0';
  const slug = t.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  return `ctx_${slug}`;
}

// Derive a premise's ctx from its own source sentence when the model omits
// it: the population labels in that sentence — capitalized names AND
// lowercase "<modifier> <population-noun>" phrases like "day shift" or
// "night-shift" (hypothesis and event names excluded) — define the
// population. Same rule derives the query's ctx from the question sentence —
// so premises and query only match when the text grounds them to the same
// population. Lowercase populations can NEVER collapse into 'ctx0'.
function deriveCtx(X, span, excludeLc) {
  if (!span || span.s === 0 && span.e === 1) return 'ctx0';
  const { s, e } = sentenceBounds(X, span.s);
  const sent = X.slice(s, e);
  const ls = [...ctxLabels(sent, excludeLc)].sort();
  return ls.length ? `ctx_${ls.join('_').replace(/\s+/g, '_')}` : 'ctx0';
}

/** Compile the model's JSON factor structure into a program AST. */
export function jsonToProgram(j, X) {
  const decls = [];
  const errors = [];
  const spanOf = (txt, what, valueRat = null, ownerName = null, lik = null) => {
    const cands = resolveSpanAll(X, txt);
    const likDup = !!lik && cands.length > 1;
    const sp = likDup
      ? pickLikSpan(X, txt, cands, valueRat, lik)
      : pickSpan(X, cands, valueRat, ownerName);
    if (!sp) {
      errors.push(likDup
        ? `ambiguous span for ${what}: ${JSON.stringify(txt)}`
        : `unresolvable span for ${what}: ${JSON.stringify(txt)}`);
      return { s: 0, e: 1 };
    }
    return sp;
  };
  // Non-numeric entity spans (HYP / EVENT): exact quote first; on a total
  // verbatim failure, ground on an exact whole-token occurrence of the
  // declared name itself. Numeric spans NEVER come through here — their
  // literal, value and ownership grounding stays exactly as strict.
  const spanOfEntity = (txt, name, what) => {
    const cands = resolveSpanAll(X, txt);
    if (cands.length) return pickSpan(X, cands, null, null);
    const fb = typeof name === 'string' && name ? nameOccurrenceSpan(X, name) : null;
    if (fb) return fb;
    errors.push(`unresolvable span for ${what}: ${JSON.stringify(txt)}`);
    return { s: 0, e: 1 };
  };
  // NON_ASSERTED spans: exact quote first, then the leading-article snap;
  // anything else fails closed through spanOf.
  const spanOfNonAsserted = (txt) => {
    const cands = resolveSpanAll(X, txt);
    if (cands.length) return pickSpan(X, cands, null, null);
    return articleSnapSpan(X, txt) ?? spanOf(txt, 'NON_ASSERTED');
  };
  const nameOf = new Map();
  for (const h of j.hyps ?? []) if (h.id && h.name) nameOf.set(h.id, String(h.name));
  for (const e of j.events ?? []) if (e.id && e.name) nameOf.set(e.id, String(e.name));
  // Raw (unfiltered) name lists: empty/duplicate names must reach
  // distinctIn so ambiguous LIK duplicates fail closed on them.
  const hypNames = (j.hyps ?? []).map((h) => (typeof h.name === 'string' ? h.name : ''));
  const evNames = (j.events ?? []).map((e) => (typeof e.name === 'string' ? e.name : ''));
  // Full names PLUS their individual words: a multi-word hypothesis
  // name ("Alpha press") must also exclude its narrowed head name
  // ("alpha", as extracted by nameMaps) so deriveCtx on a query naming
  // the head alone cannot manufacture a bogus population.
  const excludeLc = expandCtxExclude(
    [...(j.hyps ?? []).map((h) => String(h.name ?? '')),
     ...(j.events ?? []).map((e) => String(e.name ?? ''))].filter(Boolean)
  );
  // A decl's ctx: the model's label if it supplied one, else derived from
  // the decl's own source sentence. A model label is normalized; a missing
  // label is NEVER silently defaulted to the base context — it is derived
  // from the text, so a "South" premise can never collapse into 'ctx0'.
  const ctxOf = (decl, span) => {
    const label = decl && typeof decl.ctx === 'string' ? decl.ctx.trim() : '';
    const normalized = label ? normCtx(label) : 'ctx0';
    // a vacuous/'base' label never overrides the text's own grounding
    return normalized !== 'ctx0' ? normalized : deriveCtx(X, span, excludeLc);
  };

  let firstHypSpan = null;
  const hypProof = new Map();
  for (const h of j.hyps ?? []) {
    const wide = spanOfEntity(h.span, nameOf.get(h.id), `HYP ${h.id}`);
    firstHypSpan ??= wide;
    const narrowed = narrowNameSpan(X, wide, h.span, nameOf.get(h.id));
    hypProof.set(h.id, { quote: h.span, wide, span: narrowed });
    decls.push({ op: 'HYP', id: h.id, span: narrowed });
  }
  const parts = j.partition ?? (j.hyps ?? []).map((h) => h.id);
  if (j.partitionSpan != null) {
    // Supplied but unsupported/invalid grammar fails closed: record
    // the error and emit no PART_EXHAUSTIVE license from it.
    const pSpan = partitionSpanOf(X, j.partitionSpan, parts, (j.hyps ?? []).map((h) => h.id), nameOf, hypProof);
    if (!pSpan) errors.push(`invalid partitionSpan: ${JSON.stringify(j.partitionSpan)}`);
    else decls.push({ op: 'PART_EXHAUSTIVE', hyps: parts, span: pSpan });
  } else if (parts.length) {
    // Field absent: anchor on the passage's unique explicit binary
    // partition sentence when one exists (R6 Fix 3a), else keep the
    // former partition path exactly (the first HYP's own, unnarrowed
    // span); no new license is created either way.
    const autoPartSpan = findExplicitBinaryPartitionSpan(
      X, parts, nameOf,
      (j.events ?? []).map((e) => nameOf.get(e.id)).filter(Boolean)
    );
    decls.push({
      op: 'PART_EXHAUSTIVE', hyps: parts,
      span: autoPartSpan ?? firstHypSpan ?? { s: 0, e: 1 },
    });
  }
  const EVENT_BAD_WORDS = new Set([
    'not', 'no', 'never', 'or', 'if', 'unless', 'except', 'without', 'and', 'but',
  ]);
  // Single-event names carry no word-count cap: a long exact noun phrase
  // (e.g. "catch at the north stop") is a legitimate event. The strict
  // 1-3-word safe-atom guard lives only in coordinated two-arm matching
  // (coordinatedArmForSite), where multi-event ambiguity exists.
  const isSafeEventName = (name) => {
    if (typeof name !== 'string' || !name.trim() || name.trim() !== name) return false;
    const words = name.split(/\s+/);
    if (words.length < 1) return false;
    return words.every(
      (w) => /^[A-Za-z]+(?:-[A-Za-z]+)*$/.test(w) &&
        w.split('-').every((p) => !EVENT_BAD_WORDS.has(p.toLowerCase()))
    );
  };
  for (const e of j.events ?? []) {
    if (!isSafeEventName(nameOf.get(e.id))) {
      errors.push(`unsafe EVENT ${e.id} name: ${JSON.stringify(e.name)}`);
    }
    const wide = spanOfEntity(e.span, nameOf.get(e.id), `EVENT ${e.id}`);
    decls.push({ op: 'EVENT', id: e.id, span: canonicalEventSpan(X, wide, e.span, nameOf.get(e.id)) });
  }
  // OBS spans and the query ctx resolve BEFORE the numeric factors:
  // the unqueried other-stratum qualitative-LIK filter below (R6 Fix
  // 3b) needs the queried population while LIKs compile. Emission
  // order is unchanged — OBS decls are pushed from these spans at
  // their usual place, and the query keeps this same ctx.
  // Two-pass OBS resolution: unambiguous quotes first (their
  // sentences anchor the observation location), then duplicates are
  // disambiguated toward the literal-free observation sentence
  // instead of an earlier verbatim copy inside a rate sentence.
  const obsEntries = j.obs ?? [];
  const qPosPre = X.lastIndexOf('?');
  const qSpanPre = qPosPre > 0
    ? { s: sentenceBounds(X, qPosPre - 1).s, e: sentenceBounds(X, qPosPre - 1).e }
    : null;
  const obsSpans = new Array(obsEntries.length).fill(null);
  {
    const resolvedObsSents = [];
    obsEntries.forEach((o, i) => {
      const cands = resolveSpanAll(X, o.span);
      if (cands.length === 1) {
        obsSpans[i] = cands[0];
        const sb = sentenceBounds(X, cands[0].s);
        resolvedObsSents.push({ s: sb.s, e: sb.e });
      }
    });
    obsEntries.forEach((o, i) => {
      if (obsSpans[i]) return;
      const cands = resolveSpanAll(X, o.span);
      if (cands.length === 0) {
        errors.push(`unresolvable span for OBS ${o.ev}: ${JSON.stringify(o.span)}`);
        obsSpans[i] = { s: 0, e: 1 };
        return;
      }
      obsSpans[i] = pickObsSpan(X, cands, resolvedObsSents, qSpanPre) ?? cands[0];
      const sb = sentenceBounds(X, obsSpans[i].s);
      resolvedObsSents.push({ s: sb.s, e: sb.e });
    });
  }
  // The query's ctx is ALWAYS source-derived from its own sentence —
  // the question's grounding is authoritative; a model-invented label
  // must never name a population the text doesn't. An anaphoric
  // question ("this item", "it") carries no label of its own: its
  // population is the observed item's — fall back to the OBS witness
  // sentences before collapsing to 'ctx0'.
  let qCtx = deriveCtx(X, qSpanPre, excludeLc);
  const qLabels = new Set(
    qSpanPre ? ctxLabels(X.slice(qSpanPre.s, qSpanPre.e), excludeLc) : []
  );
  if (qCtx === 'ctx0') {
    for (const sp of obsSpans) {
      const oc = deriveCtx(X, sp, excludeLc);
      if (oc !== 'ctx0') {
        qCtx = oc;
        const osb = sentenceBounds(X, sp.s);
        for (const l of ctxLabels(X.slice(osb.s, osb.e), excludeLc)) qLabels.add(l);
        break;
      }
    }
  }
  // Events the query path actually references (queried, observed, or
  // part of a stated posterior / license). A literal-free qualitative
  // 0/1 LIK over any other event in a disjoint stratum is filtered
  // below; every referenced event keeps its factors untouched.
  const referencedEvents = new Set();
  if (j.query?.target) referencedEvents.add(j.query.target);
  for (const g of j.query?.given ?? []) if (g?.ev) referencedEvents.add(g.ev);
  for (const o of j.obs ?? []) if (o?.ev) referencedEvents.add(o.ev);
  for (const p of j.posteriors ?? []) if (p?.ev) referencedEvents.add(p.ev);
  for (const ci of j.condIndep ?? []) for (const e of ci?.evs ?? []) referencedEvents.add(e);
  for (const dl of j.detLinks ?? []) {
    if (dl?.a) referencedEvents.add(dl.a);
    if (dl?.b) referencedEvents.add(dl.b);
  }
  // Literal-site ownership across ALL numeric factors: one source figure
  // grounds exactly one factor. A site claimed as both an unconditional
  // PRIOR and a POSTERIOR_STATED (or by two distinct LIKs) is a conflict.
  const numericSiteOwners = new Map();
  const claimSite = (lit, factorKey, what) => {
    if (!lit) return;
    const site = `${lit.s}:${lit.e}`;
    const owner = numericSiteOwners.get(site);
    if (owner !== undefined && owner !== factorKey) {
      errors.push(`${what}: shared literal at source offset ${site} claimed by distinct factors (${owner} vs ${factorKey})`);
    } else {
      numericSiteOwners.set(site, factorKey);
    }
  };
  const evEntries = (j.events ?? []).map((e) => ({ id: e.id, name: nameOf.get(e.id) }));
  const hypEntries = (j.hyps ?? []).map((h) => ({ id: h.id, name: nameOf.get(h.id) }));
  // The literal site a LIK actually binds: inside a broad coordinated
  // quote holding both arms' figures (possibly equal values), the arm
  // attribution picks this decl's own site rather than the first match.
  const attributedLikSite = (span, decl) => {
    const cands = literalsInSpan(X, span).filter((l) => R.eq(l.value, decl.value));
    if (cands.length <= 1) return cands[0] ?? null;
    for (const site of cands) {
      const coord = coordinatedArmForSite(X, site, evEntries, hypEntries);
      if (coord && coord.evId === decl.ev && coord.hypId === decl.hyp) return site;
    }
    return cands[0];
  };
  for (const p of j.priors ?? []) {
    const v = rat(p.value);
    const span = spanOf(p.span, `PRIOR ${p.hyp}`, v, nameOf.get(p.hyp));
    const decl = {
      op: 'PRIOR', hyp: p.hyp, value: v, ctx: ctxOf(p, span), span,
    };
    decls.push(decl);
    claimSite(
      literalAtSpanForValue(X, span, v),
      JSON.stringify(['PRIOR', decl.hyp, decl.ctx ?? null]),
      `PRIOR ${p.hyp}`
    );
  }
  const skippedLikEvs = new Set();
  for (const l of j.liks ?? []) {
    const v = rat(l.value);
    const span = spanOf(l.span, `LIK ${l.ev}`, v, nameOf.get(l.hyp), {
      hypName: nameOf.get(l.hyp), evName: nameOf.get(l.ev), hypNames, evNames,
      evId: l.ev, hypId: l.hyp, evEntries, hypEntries,
    });
    const declCtx = ctxOf(l, span);
    // Unqueried other-stratum qualitative 0/1 LIK filter (R6 Fix 3b):
    // a synthesized endpoint likelihood over an event nothing queries,
    // observes, or licenses, grounded only in a literal-free sentence
    // scoped to a named stratum disjoint from the queried one, is the
    // extractor's rendering of qualitative universal prose about the
    // other stratum. Skip it so that sentence is accounted by the
    // other-stratum excuse instead of failing literal ownership.
    // Every guard failing keeps the LIK (and its refusal) untouched:
    // same stratum, referenced event, any numeric literal, or a
    // single-population passage never filters.
    if ((R.isZero(v) || R.isOne(v)) && !referencedEvents.has(l.ev) &&
        resolveSpanAll(X, l.span).length > 0 &&
        qCtx !== 'ctx0' && declCtx !== 'ctx0' && declCtx !== qCtx) {
      const sb = sentenceBounds(X, span.s);
      const sent = X.slice(sb.s, sb.e);
      const literalFree = !scanLiterals(sent).some((lit) => !/^\d+$/.test(lit.text));
      const sharesLabel = [...ctxLabels(sent, excludeLc)].some((lab) => qLabels.has(lab));
      if (literalFree && !sharesLabel) {
        skippedLikEvs.add(l.ev);
        continue;
      }
    }
    const decl = {
      op: 'LIK', ev: l.ev, pol: l.pol ?? 'POS', hyp: l.hyp,
      value: v, ctx: declCtx, span,
    };
    decls.push(decl);
    claimSite(
      attributedLikSite(span, decl),
      JSON.stringify(['LIK', decl.ev, decl.pol, decl.hyp, decl.ctx ?? null]),
      `LIK ${l.ev}`
    );
  }
  // An EVENT used ONLY by filtered other-stratum LIKs (never queried,
  // observed, licensed, or kept by a surviving LIK) leaves with them.
  if (skippedLikEvs.size > 0) {
    const keptLikEvs = new Set(decls.filter((d) => d.op === 'LIK').map((d) => d.ev));
    for (let i = decls.length - 1; i >= 0; i--) {
      const d = decls[i];
      if (d.op === 'EVENT' && skippedLikEvs.has(d.id) &&
          !referencedEvents.has(d.id) && !keptLikEvs.has(d.id)) {
        decls.splice(i, 1);
      }
    }
  }
  for (const p of j.posteriors ?? []) {
    const v = rat(p.value);
    const span = spanOf(p.span, `POSTERIOR_STATED ${p.hyp}`, v, nameOf.get(p.hyp));
    const decl = {
      op: 'POSTERIOR_STATED', hyp: p.hyp, ev: p.ev, pol: p.pol ?? 'POS',
      value: v, ctx: ctxOf(p, span), span,
    };
    decls.push(decl);
    claimSite(
      literalAtSpanForValue(X, span, v),
      JSON.stringify(['POSTERIOR_STATED', decl.hyp, decl.ev, decl.pol, decl.ctx ?? null]),
      `POSTERIOR_STATED ${p.hyp}`
    );
  }
  // OBS decls emit from the spans pre-resolved above (same two-pass
  // disambiguation, same order, no span ever synthesized).
  obsEntries.forEach((o, i) => {
    const span = obsSpans[i];
    decls.push({
      op: 'OBS', ev: o.ev, pol: o.pol ?? 'POS', ctx: ctxOf(o, span), span,
    });
  });
  for (const na of j.nonAsserted ?? []) {
    decls.push({
      op: 'NON_ASSERTED',
      span: spanOfNonAsserted(na.span),
      reason: na.reason ?? 'QUOTED',
    });
  }
  // Valid DET_LINK spans resolve first (same pickSpan the emission
  // below uses) so a contradictory duplicate COND_INDEP can be told
  // from a genuine license (R6 Fix 2). Resolution here never records
  // errors — emission still validates every entry in full.
  const detIdSet = new Set((j.hyps ?? []).map((h) => h.id));
  const detResolved = [];
  for (const dl of j.detLinks ?? []) {
    if (detIdSet.has(dl.a) || detIdSet.has(dl.b)) continue;
    const rel = String(dl.rel ?? 'EQUIV').toUpperCase();
    if (!['EQUIV', 'IMPLIES', 'MUTEX'].includes(rel)) continue;
    const cands = resolveSpanAll(X, dl.span);
    if (cands.length === 0) continue;
    detResolved.push({ a: dl.a, b: dl.b, span: pickSpan(X, cands, null, null) });
  }
  for (const ci of j.condIndep ?? []) {
    const span = spanOf(ci.span, 'COND_INDEP');
    const evs = ci.evs ?? [];
    // A COND_INDEP over the exact same event pair, in the exact same
    // sentence, as a DET_LINK contradicts that link (a deterministic
    // relation and conditional independence cannot both hold), and
    // the sentence states no independence. Drop the duplicate; the
    // DET_LINK still faces full license + witness verification. A
    // standalone COND_INDEP, or one whose sentence affirmatively
    // asserts independence, is never dropped.
    if (evs.length === 2 && resolveSpanAll(X, ci.span).length > 0) {
      const ciSent = sentenceBounds(X, span.s);
      const duplicate = detResolved.some(
        (dl) =>
          ((dl.a === evs[0] && dl.b === evs[1]) ||
           (dl.a === evs[1] && dl.b === evs[0])) &&
          sentenceBounds(X, dl.span.s).s === ciSent.s
      );
      if (duplicate && !assertsIndependence(X.slice(ciSent.s, ciSent.e))) continue;
    }
    decls.push({
      op: 'COND_INDEP',
      evs,
      hyp: ci.given ?? '*',
      ctx: ctxOf(ci, span), span,
    });
  }
  const hypIds = new Set((j.hyps ?? []).map((h) => h.id));
  for (const dl of j.detLinks ?? []) {
    if (hypIds.has(dl.a) || hypIds.has(dl.b)) {
      errors.push(`DET_LINK endpoints are events only, never hypothesis ids: ${dl.a} ${dl.b}`);
      continue;
    }
    const span = spanOf(dl.span, 'DET_LINK');
    const rel = String(dl.rel ?? 'EQUIV').toUpperCase();
    if (!['EQUIV', 'IMPLIES', 'MUTEX'].includes(rel)) {
      errors.push(`bad DET_LINK rel "${dl.rel}"`);
      continue;
    }
    decls.push({
      op: 'DET_LINK',
      a: dl.a, rel, b: dl.b,
      ctx: ctxOf(dl, span), span,
    });
  }
  const q = j.query ?? {};
  const given = (q.given ?? []).map((g) => ({ ev: g.ev, pol: g.pol ?? 'POS' }));
  // qCtx was derived above from the query sentence (with the OBS
  // witness fallback) so distinct populations stay distinct through
  // compilation; the query keeps exactly that ctx.
  const program = {
    decls,
    query: { op: 'ASK_PROB', target: q.target, negated: !!q.negated, given, ctx: qCtx },
  };
  // the sentence-partition ledger is compiler-owned, never model-owned:
  // every sentence is ledgered by construction, and the kernel re-verifies
  // each tag against the text anyway.
  const usedCtxs = new Set(
    decls.map((d) => d.ctx).filter(Boolean).concat([qCtx])
  );
  for (const t of autoSentTags(X, program, usedCtxs)) decls.push(t);
  if (!errors.length) {
    const { nameById, evNameById } = nameMaps(X, program);
    const likDecls = program.decls.filter((d) => d.op === 'LIK');
    for (const f of ownershipCheck({ X, decls: likDecls, nameById, evNameById, prog: program })) {
      if (f.code !== 'UNSTABLE_BINDING') continue;
      if (/has no name-grounded HYP span|no grammatical owner/.test(f.detail)) continue;
      errors.push(`${f.code}: ${f.detail}`);
    }
  }
  return { program, errors };
}

function extractJsonObject(text) {
  // find the first balanced {...} block
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
 * makeOllamaExtractor({model, host}) -> async (X, info) => program AST
 * This is the untrusted neural extractor under test.
 */
export function makeOllamaExtractor({ model, host, calls = null }) {
  return async (X /*, info */) => {
    const { content } = await chat({
      model,
      host,
      numPredict: 2500,
      think: false, // deterministic extraction — no reasoning-channel burn
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
    const { program, errors } = jsonToProgram(j, X);
    if (errors.length) throw new ExtractionError(errors.join('; '));
    return program;
  };
}
