// invariance :: Independent Cross-Model Semantic Entailment Witness.
//
// The third substrate of ASSAY-grade verification. The symbolic algebra
// (src/kernel/credal.js) can only reason over what the program DECLARES; the
// deterministic text-layer witnesses (sentences.js, ownership.js) can only
// reason over what a finite template can see. Neither can verify the
// semantic bindings that defeated R3–R7 — English tense ("were wired"),
// negated context membership ("is not a night-shift panel"), subset scope
// ("on damaged panels", "inside the locked enclosure"), and cross-sentence
// noun co-reference ("clinic" carried by a FLUFF sentence). Finite JS
// regexes cannot enumerate English; whack-a-mole is over.
//
// This witness puts every semantic binding algebra cannot check to an
// INDEPENDENT model family — Qwen3.8-27B (Alibaba) on local Ollama, a
// different base model than the gemma4 (Google) extractor — as a single
// compact batch natural-language-entailment problem:
//   (a) every DET_LINK (EQUIV/IMPLIES/MUTEX) and COND_INDEP is asserted
//       unconditionally and UNIVERSALLY over the entire queried population,
//       all hypotheses in scope, at the exact time of the queried
//       observation — zero subset/damage/enclosure/testing exceptions, zero
//       modal or reported wrappers, zero tense mismatch;
//   (b) the queried item in OBS/ASK_PROB affirmatively belongs to the exact
//       same population/context as every premise applied to it — not
//       negated, not foreign;
//   (c) no SENT_TAG FLUFF sentence restricts, negates, retracts, or
//       modifies the queried item or any premise.
// The verdict returns as a SHA-256-bound Entailment Witness Receipt. A
// release without a verified receipt is forged prose — never a release.

import { createHash } from 'node:crypto';
import { toProgram, toBytecode } from '../kernel/parser.js';
import {
  sentences,
  sentenceAt,
  eventNameMap,
  sentenceMateriality,
  entityNames,
  contextLabelPolarity,
  findPoisoningPremiseDoubt,
  irrelevantUncheckedSeg,
  irrelevantUnrecordedPriorSeg,
  irrelevantOtherStratumSeg,
  isInventoryOverviewSentence,
  adjacentClauseFindings,
  subClauses,
  CONCESSIVE_MARKER,
  HEDGED_NEVER,
} from '../kernel/sentences.js';
import { solve } from '../kernel/solve.js';
import { ownershipCheck, nameMaps } from './ownership.js';

export const WITNESS_MODEL = 'hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M';
const DEFAULT_HOST = 'http://127.0.0.1:11434';

// Pre-load unhookable witness brand (W1): `#brand in v` is a lexical
// private-slot check resolved by the engine, not a property lookup. It
// cannot be forged by overriding WeakSet.prototype.has,
// Function.prototype.call/bind, Proxy or Reflect.has — before OR after
// module import — and clones/spreads/JSON copies never carry the slot.
// Issuance is capability-gated (Bulma R3 W1): construction requires the
// module-private MINT_CAP symbol, and every issued instance has a null
// prototype, so holding a genuine receipt exposes no constructor and
// mints nothing. The class and its prototype are frozen as well.
const MINT_CAP = Symbol('SealedWitnessReceipt.mint');
class SealedWitnessReceipt {
  #brand = true;
  constructor(fields, cap) {
    if (cap !== MINT_CAP) {
      throw new TypeError('SealedWitnessReceipt cannot be constructed directly');
    }
    Object.assign(this, fields);
    Object.setPrototypeOf(this, null);
    Object.freeze(this);
  }
  static isIssued(v) {
    try {
      return Boolean(v && typeof v === 'object' && #brand in v);
    } catch {
      return false;
    }
  }
}
Object.freeze(SealedWitnessReceipt.prototype);
Object.freeze(SealedWitnessReceipt);

function issueWitness(receipt) {
  for (const check of receipt.checks) Object.freeze(check);
  Object.freeze(receipt.checks);
  return new SealedWitnessReceipt(receipt, MINT_CAP);
}

// Prototype-pollution-immune option reads (W2): a flag counts only as an
// OWN property of the exact options object the caller passed. Inherited
// values (Object.prototype pollution, __proto__ payloads folded in via
// Object.assign) and destructuring-default prototype walks never opt in.
const hasOwn = Object.hasOwn;
function ownOpt(opts, key, fallback) {
  if (!opts || (typeof opts !== 'object' && typeof opts !== 'function')) return fallback;
  if (!hasOwn(opts, key)) return fallback;
  const v = opts[key];
  return v === undefined ? fallback : v;
}

function unapprovedEndpoint(model, host, optIn) {
  return optIn !== true && (model !== WITNESS_MODEL || host !== DEFAULT_HOST);
}

const ENDPOINT_OPT_IN = 'custom witness host or model requires allowCustomWitnessEndpoint: true';

const sha256 = (s) => createHash('sha256').update(s, 'utf8').digest('hex');
export const passageSha256 = (X) => sha256(X);
export const programCanonicalSha256 = (prog) => sha256(toBytecode(prog));

// Programs whose release cannot be certified by algebra + deterministic
// witnesses alone: a ledgered license, a FLUFF sentence, or a bound
// population context are all semantic claims about X that only entailment
// can judge. (The public gate requires the witness on EVERYTHING anyway
// unless explicitly told otherwise — this is the always-on trigger set for
// callers that relax requireEntailmentWitness.)
export function programNeedsEntailment(prog, X = null) {
  for (const d of prog.decls) {
    if (d.op === 'DET_LINK' || d.op === 'COND_INDEP') return true;
    if (d.op === 'SENT_TAG' && d.tag === 'FLUFF') return true;
    if (d.op !== 'SENT_TAG' && d.ctx && d.ctx !== 'ctx0' && d.ctx !== '*') return true;
  }
  if (prog.query?.ctx && prog.query.ctx !== 'ctx0' && prog.query.ctx !== '*') return true;
  // a residual/adjacent material clause inside a premise sentence is a
  // semantic claim no decl ledgers — the witness must judge it even on
  // the relaxed path
  if (typeof X === 'string' && X.length > 0 && adjacentClauseFindings(X, prog).length > 0) {
    return true;
  }
  // a whole-sentence OBS span leaves no uncovered interval and may
  // leave no residual clause piece, yet its full text can still assert
  // a second event's state — only the witness can judge it
  if (typeof X === 'string' && X.length > 0 && obsSentenceChecks(X, prog).length > 0) {
    return true;
  }
  // a reported co-/solo-activation anywhere in the passage can defeat
  // a DET_LINK license without leaving any uncovered interval
  if (typeof X === 'string' && X.length > 0 && licenseCounterexampleFindings(X, prog).length > 0) {
    return true;
  }
  return false;
}

const REL_CLAIM = {
  EQUIV: (a, b) =>
    `"${a}" and "${b}" always have the same state — whenever one occurs the ` +
    `other occurs, and whenever one does not the other does not`,
  IMPLIES: (a, b) => `whenever "${a}" occurs, "${b}" also occurs`,
  MUTEX: (a, b) => `"${a}" and "${b}" never both occur on the same item`,
};

const sentText = (seg) => (seg ? seg.text.trim().replace(/\s+/g, ' ') : null);

const escRe2 = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
// OBS sentences whose full text the witness must judge even when the
// OBS span covers the whole sentence (Bulma R6): the sentence mentions
// a declared event no OBS in it asserts, or carries joint/state
// vocabulary ("both", "signals", "silent", "solo", "only", "neither")
// an OBS decl cannot ledger on its own.
const OBS_SENT_TRIGGER = /\b(?:both|signals?|indicators?|silent|solo|only|neither)\b/i;
function obsSentenceChecks(X, prog) {
  const evNameById = eventNameMap(X, prog);
  const bySeg = new Map();
  for (const d of prog.decls) {
    if (d.op !== 'OBS' || !d.span) continue;
    const seg = sentenceAt(X, d.span.s);
    if (!seg) continue;
    if (!bySeg.has(seg.idx)) bySeg.set(seg.idx, { seg, obs: [] });
    bySeg.get(seg.idx).obs.push(d);
  }
  const out = [];
  for (const { seg, obs } of bySeg.values()) {
    const observed = new Set(obs.map((d) => d.ev));
    const second = [...evNameById.entries()]
      .filter(([id, name]) => !observed.has(id) &&
        new RegExp(`\\b${escRe2(name)}\\b`, 'i').test(seg.text))
      .map(([, name]) => name);
    if (second.length > 0 || OBS_SENT_TRIGGER.test(seg.text)) {
      out.push({ seg, obs, second });
    }
  }
  return out;
}

// Reported occurrences that contradict a DET_LINK license on their face
// (Bulma R7): a co-activation report defeats MUTEX; a solo-activation
// report ("X lit alone", "X opened without the Y", "only the X rang")
// defeats EQUIV, and defeats IMPLIES when the solo event is its
// antecedent. The scan is clause-level over EVERY sentence — prior,
// rate, likelihood, observation, and query alike — because R7 attached
// the report to whichever sentence the interval accounting skipped.
// The license's own clause (any piece its span overlaps) and universal
// claims ("always activate together") are never occurrence reports.
const CE_ACTIVATION =
  /\b(?:fired|firing|lit|lighting|rang|ringing|sounded|sounding|occurred|activated|activating|swung|swinging|ran|running|glowed|glowing|opened|opening|triggered|triggering)\b|\b[a-z]{4,}ed\b/i;
const CE_UNIVERSAL = /\b(?:always|never|cannot|can't|every|whenever|must)\b/i;
const CE_PAIR_NOUN = /\b(?:devices?|signals?|indicators?|outputs?|openings?|events?)\b/i;
// A purely locative relative ("whose vent was beside its shutter") only
// places the two named parts side by side: it reports no occurrence of
// either event, so a reporting verb outside it ("documented") must not
// turn the two located nouns into a co-activation report. The pattern
// is deliberately rigid — relative pronoun, one event name, a copula, a
// locative preposition, the other event name, and nothing else in its
// complement. It is stripped only when that complement is complete:
// bounded by the end of the judged text or by an actual resumption of
// a rate ("and 20% of ..."). A trailing temporal, relative, or other
// clause ("when both activated", "which then fired together") means
// the material is not proven purely locative and is judged in full.
const CE_LOCATIVE_PREP =
  '(?:beside|next\\s+to|adjacent\\s+to|alongside|near|above|below|behind|in\\s+front\\s+of)';
const CE_LOCATIVE_BOUNDARY =
  '(?=\\s*(?:[.!?](?:\\s|$)|$)|\\s*,?\\s+and\\s+\\d+(?:\\.\\d+)?\\s*(?:%|percent\\b))';
function stripPureLocativeRelatives(t, a, b) {
  let out = t;
  for (const [x, y] of [[a, b], [b, a]]) {
    out = out.replace(
      new RegExp(
        `\\bwhose\\s+(?:the\\s+|its\\s+)?${escRe2(x)}\\s+(?:is|was)` +
        `(?:\\s+(?:mounted|installed|located|situated|positioned|fixed))?\\s+` +
        `${CE_LOCATIVE_PREP}\\s+(?:the\\s+|its\\s+|their\\s+)?${escRe2(y)}` +
        `${CE_LOCATIVE_BOUNDARY}`, 'gi'),
      ' ');
  }
  return out;
}
// A collective pair is exempt only as a proved complete, stative
// reporting object at its own occurrence: a reporting verb directly
// governs that "both <pair>" and a compact, syntactically inert
// location NP that terminates at a true rate resumption or at the end
// of the full sentence. The full sentence is used only to prove the
// ending of the same report that overlaps this piece at its original
// source offsets (including endings split from the piece by "and");
// it is never a sentence-wide boolean that exempts another "both"
// elsewhere in the sentence. Each proved report is subtracted, in
// place and at unchanged length, and any remaining "both <pair>" in
// the piece — an action predicate ("powered both ..."), or material
// after the location NP ("pulsing in unison", "aglow") that defeated
// the boundary proof — keeps the existing hard finding. The location
// NP may carry one neutral cardinal-direction premodifier ("west
// door"); no other extra word is allowed in the proved NP.
const CE_REPORT_GOVERNING =
  '(?:documented|recorded|logged|reported|noted|observed)';
const CE_DIRECTION_PREMODIFIER =
  '(?:north|south|east|west|northern|southern|eastern|western|' +
  'northeast|northwest|southeast|southwest|north-east|north-west|' +
  'south-east|south-west|northeastern|northwestern|southeastern|southwestern)';
function stripProvedCompletePairLocatives(t, fullText, tOffsetInFull) {
  const hay = typeof fullText === 'string' && fullText.length >= t.length
    ? fullText : t;
  let offset = Number.isInteger(tOffsetInFull) ? tOffsetInFull : hay.indexOf(t);
  if (offset < 0 || hay.slice(offset, offset + t.length) !== t) {
    const at = hay.indexOf(t);
    offset = at >= 0 ? at : 0;
  }
  const re = new RegExp(
    `\\b${CE_REPORT_GOVERNING}\\s+both\\s+` +
    `(?:devices?|signals?|indicators?|outputs?|openings?|events?)\\s+` +
    `${CE_LOCATIVE_PREP}\\s+(?:the\\s+|its\\s+|their\\s+)?` +
    `(?:${CE_DIRECTION_PREMODIFIER}\\s+)?[a-z][a-z-]*` +
    `${CE_LOCATIVE_BOUNDARY}`, 'gi',
  );
  const chars = t.split('');
  let m;
  while ((m = re.exec(hay)) !== null) {
    const s = Math.max(0, m.index - offset);
    const e = Math.min(t.length, m.index + m[0].length - offset);
    for (let i = s; i < e; i += 1) chars[i] = ' ';
    if (m[0].length === 0) re.lastIndex += 1;
  }
  return chars.join('');
}
// A named part occurring as the subject of its own occurrence verb
// ("its shutter opened"), as opposed to merely being located or named.
const CE_OCCURRENCE_VERB =
  '(?:opened|opens|opening|fired|fires|firing|lit|lights|lighting|rang|rings|ringing|' +
  'sounded|sounds|sounding|occurred|occurs|occurring|activated|activates|activating|' +
  'triggered|triggers|triggering|swung|swings|swinging|ran|runs|running|glowed|glows|glowing)';
function eventOccurs(t, name) {
  return new RegExp(
    `\\b${escRe2(name)}\\s+(?:also\\s+)?(?:(?:was|is)\\s+)?${CE_OCCURRENCE_VERB}\\b`, 'i',
  ).test(t);
}
const CE_TEMPORAL_JOIN =
  /\b(?:while|simultaneously|together)\b|\bat\s+once\b|\bat\s+the\s+same\s+(?:time|moment)\b/i;
const CE_REPORT_CONTEXT =
  /\b(?:documented|recorded|logged|reported|noted|observed|recorder|records?|reports?|logs?)\b/i;
export function licenseCounterexampleFindings(X, prog) {
  const evNameById = eventNameMap(X, prog);
  const names = entityNames(X, prog);
  const links = prog.decls.filter((d) => d.op === 'DET_LINK' && d.span);
  if (links.length === 0) return [];
  const mentions = (t, name) => new RegExp(`\\b${escRe2(name)}\\b`, 'i').test(t);
  const out = [];
  for (const seg of sentences(X)) {
    const pieces = subClauses(seg, names);
    for (const piece of pieces) {
      const t = piece.text;
      if (CE_UNIVERSAL.test(t) || !CE_ACTIVATION.test(t)) continue;
      for (const d of links) {
        if (d.span.s < piece.e && piece.s < d.span.e) continue;
        const a = evNameById.get(d.a);
        const b = evNameById.get(d.b);
        if (!a || !b) continue;
        const judged = d.rel === 'MUTEX' ? stripPureLocativeRelatives(t, a, b) : t;
        let hit = false;
        if (d.rel === 'MUTEX') {
          const pairJudged = stripPureLocativeRelatives(
            stripProvedCompletePairLocatives(t, seg.text, piece.s - seg.s), a, b);
          hit = (mentions(judged, a) && mentions(judged, b)) ||
            (/\bboth\b/i.test(pairJudged) && CE_PAIR_NOUN.test(pairJudged));
        } else {
          const soloOf = (name, other) => {
            const at = judged.toLowerCase().indexOf(name.toLowerCase());
            if (at < 0) return false;
            const rest = judged.slice(at + name.length);
            if (new RegExp(`\\bwithout\\s+(?:the\\s+|its\\s+)?${escRe2(other)}\\b`, 'i').test(rest)) return true;
            if (/\b(?:alone|by itself|on its own|solo)\b/i.test(rest)) return true;
            return new RegExp(`\\bonly\\s+(?:the\\s+)?${escRe2(name)}\\b`, 'i').test(judged);
          };
          hit = d.rel === 'EQUIV' ? (soloOf(a, b) || soloOf(b, a)) : soloOf(a, b);
        }
        if (hit) out.push({ segIdx: seg.idx, piece, rel: d.rel });
      }
    }
    // Sentence-level sweep of an attributed report's whole relative:
    // subClauses may split before "while", leaving one occurring event
    // per piece ("whose vent opened" / "while its shutter opened"), so
    // no single piece carries the joint occurrence. Judge the relative
    // clause itself — starting at "whose", so a likelihood predicate in
    // the surrounding sentence is never conflated with an occurrence
    // inside the relative — and require each named part to govern its
    // own occurrence verb under an explicit temporal join. A purely
    // locative relative fails eventOccurs (and is stripped first), so
    // this fires only on a genuine same-item joint occurrence.
    const whoseAt = seg.text.search(/\bwhose\b/i);
    if (whoseAt >= 0 && pieces.length > 0 &&
        CE_REPORT_CONTEXT.test(seg.text.slice(0, whoseAt))) {
      const relPiece = {
        text: seg.text.slice(whoseAt),
        s: seg.s + whoseAt,
        e: seg.e,
      };
      if (!CE_UNIVERSAL.test(relPiece.text) && CE_TEMPORAL_JOIN.test(relPiece.text)) {
        for (const d of links) {
          if (d.rel !== 'MUTEX') continue;
          if (d.span.s < relPiece.e && relPiece.s < d.span.e) continue;
          const a = evNameById.get(d.a);
          const b = evNameById.get(d.b);
          if (!a || !b) continue;
          const judged = stripPureLocativeRelatives(relPiece.text, a, b);
          if (eventOccurs(judged, a) && eventOccurs(judged, b) &&
              !out.some((f) => f.segIdx === seg.idx && f.rel === d.rel)) {
            out.push({ segIdx: seg.idx, piece: relPiece, rel: d.rel });
          }
        }
      }
    }
  }
  return out;
}

// Sentences the witness must sweep for a license counterexample even
// when no deterministic pattern fires: a concessive clause, a recorded
// -case report, or a past-tense co-/solo-activation report, in any
// sentence of the passage. A license sentence's own present-tense
// universal claim ("always activate together") never triggers.
const CE_REPORT_CASE = /\b(?:documented|recorded|logged)\s+(?:case|instance)\b/i;
const CE_PAST_REPORT =
  /\b(?:fired|firing|lit|lighting|rang|ringing|sounded|sounding|occurred|occurring|activated|activating|swung|swinging|opened|opening|ran|running|glowed|glowing|triggered|triggering)\b/i;
function counterexampleSweepSentences(X, prog) {
  const evNameById = eventNameMap(X, prog);
  const evNames = [...evNameById.values()];
  const out = [];
  for (const seg of sentences(X)) {
    const t = seg.text;
    const namedCount = evNames.filter(
      (n) => new RegExp(`\\b${escRe2(n)}\\b`, 'i').test(t)).length;
    const bothPair = /\bboth\b/i.test(t) && CE_PAIR_NOUN.test(t);
    const activationReport = CE_PAST_REPORT.test(t) &&
      (CE_TEMPORAL_JOIN.test(t) || /\b(?:alone|solo|without)\b/i.test(t)) &&
      (namedCount > 0 || CE_PAIR_NOUN.test(t));
    const assessedReport = CE_REPORT_CONTEXT.test(t) &&
      ((/\b(?:whose|who|which)\b/i.test(t) &&
        (namedCount >= 2 || bothPair ||
          (namedCount >= 1 && CE_TEMPORAL_JOIN.test(t)))) ||
        (namedCount >= 2 && CE_TEMPORAL_JOIN.test(t)));
    if (CONCESSIVE_MARKER.test(t) || CE_REPORT_CASE.test(t) ||
        activationReport || assessedReport) {
      out.push(seg);
    }
  }
  return out;
}

// the noun the OBS/query calls the queried item ("panel", "capsule",
// "item", ...) — named concretely so the witness can tell a subset
// ("Ash panels", "damaged panels") or a foreign population
// ("prototypes", "clinics") from the queried domain itself.
const ITEM_NOUN = /\b(panel|capsule|item|unit|module|device|machine|sensor|cabinet|enclosure|prototype|clinic|crew|ward|bay|line|station)s?\b/gi;
function observedItemNoun(X, prog) {
  const scan = (text) => {
    let last = null;
    for (const m of text.matchAll(ITEM_NOUN)) last = m[1].toLowerCase();
    return last;
  };
  for (const d of prog.decls) {
    if (d.op === 'OBS' && d.span) {
      const n = scan(X.slice(d.span.s, d.span.e));
      if (n) return n;
    }
  }
  const qPos = X.lastIndexOf('?');
  return qPos > 0 ? scan(X.slice(qPos)) : null;
}

// Render every semantic binding the program asserts into a numbered
// entailment claim for the batch witness prompt.
function buildChecks(prog, X) {
  const segs = sentences(X);
  const evNameById = eventNameMap(X, prog);
  const evName = (id) => evNameById.get(id) ?? id;
  const itemNoun = observedItemNoun(X, prog);
  const hypNames = new Map();
  for (const d of prog.decls) {
    if (d.op === 'HYP' && d.span) {
      const m = X.slice(d.span.s, d.span.e).match(/\b([A-Z][a-zA-Z]+)\b/);
      if (m) hypNames.set(d.id, m[1]);
    }
  }
  const checks = [];
  let li = 0;

  // the queried population, stated concretely from the OBS/query text:
  // affirmative context labels the observation assigns to the item
  // ("a night-shift panel"), and populations it affirmatively DENIES.
  const excl = new Set(
    [...evNameById.values(), ...hypNames.values()].map((n) => n.toLowerCase())
  );
  const obsPop = new Set();
  const obsNeg = new Set();
  {
    const texts = [];
    for (const d of prog.decls) {
      if (d.op === 'OBS' && d.span) {
        const seg = sentenceAt(X, d.span.s);
        if (seg) texts.push(seg.text);
      }
    }
    const qp = X.lastIndexOf('?');
    if (qp > 0) {
      const seg = sentenceAt(X, qp);
      if (seg) texts.push(seg.text);
    }
    for (const s of texts) {
      const { affirmed, excluded } = contextLabelPolarity(s, excl);
      for (const label of affirmed) obsPop.add(label);
      for (const label of excluded) obsNeg.add(label);
    }
    for (const label of obsNeg) obsPop.delete(label);
  }
  const popDesc = obsPop.size
    ? `the "${[...obsPop].join('" and "')}" population — the observation affirmatively places the item there`
    : `the ENTIRE unqualified ${itemNoun ?? 'item'} population — the observation states no membership qualifier, so the relation must hold for every ${itemNoun ?? 'item'} without restriction`;
  const negTxt = obsNeg.size
    ? ` The observed item is explicitly DENIED membership in "${[...obsNeg].join('", "')}" — a premise scoped to a denied population FAILS.`
    : '';

  const ctxTxt = (ctx) =>
    ctx && ctx !== 'ctx0' && ctx !== '*'
      ? ` restricted by the program to the "${ctx.replace(/^ctx_/, '').replace(/_/g, ' ')}" population`
      : '';

  // Surface candidate restriction markers from the license sentence as a
  // prompt hint — the witness still judges; a universal phrase ("on all
  // panels", "for every panel") or a manner phrase ("in lockstep") passes,
  // while a genuine subset ("inside the locked enclosure", "on damaged
  // panels") fails when the observation does not affirm the item meets it.
  const RESTRICT_MARKER =
    /\b(on|in|inside|outside|except|excepting|unless|until|during|without|under|within|across|through|if|when|while|where|with)\b/gi;
  const markerTxt = (segText) => {
    const ms = [
      ...new Set([...segText.toLowerCase().matchAll(RESTRICT_MARKER)].map((m) => m[0])),
    ];
    return ms.length
      ? ` Possible restriction marker(s) present: ${ms.map((m) => `"${m}"`).join(', ')} — if ` +
          `any marker limits WHICH items, WHERE, or WHEN the relation holds and the passage ` +
          `does not affirm the observed item satisfies it, FAIL; if the phrase is universal ` +
          `("on all panels", "for every panel", "in lockstep") it does not restrict.`
      : '';
  };

  for (const d of prog.decls) {
    if (d.op === 'DET_LINK' && d.span) {
      const seg = sentenceAt(X, d.span.s);
      const rel = REL_CLAIM[d.rel]?.(evName(d.a), evName(d.b));
      if (!seg || !rel) continue;
      li++;
      checks.push({
        id: `L${li}`,
        text:
          `${`L${li}`} [LICENSE ${d.rel}]: The program asserts that "${sentText(seg)}" ` +
          `establishes, for EVERY ${itemNoun ?? 'item'} in the queried population` +
          `${ctxTxt(d.ctx)} — ${popDesc} — that ${rel} — holding at the exact time ` +
          `of the queried observation, universally across that population, with no ` +
          `subset, exception, damage, enclosure, testing, or time restriction; no ` +
          `modal, hypothetical, or reported wrapper; and no tense mismatch (a ` +
          `past-tense clause does not assert a present relation). A quantified ` +
          `domain that is a SUBSET of the queried population ("on damaged ` +
          `${itemNoun ?? 'item'}s", "on Ash ${itemNoun ?? 'item'}s" when the item is unqualified) or a ` +
          `DIFFERENT noun ("every prototype's", "every clinic's") fails.` +
          (d.rel === 'MUTEX'
            ? ' Accept categorical exclusion ("cannot", "can never", "will never", "must never", or bare "never"). Reject hedges such as "may never", "might never", "could never", and "would never": they do not rule out co-occurrence.'
            : '') +
          ' FAIL if ANY clause anywhere in the passage — including a clause attached to a prior, rate, likelihood, observation, or query sentence — reports an actual exception or counterexample occurrence to this universal rule, however it is attached (concessive, temporal, parenthetical, or coordinate).' +
          negTxt + markerTxt(sentText(seg)),
      });
    }
    if (d.op === 'COND_INDEP' && d.span) {
      const seg = sentenceAt(X, d.span.s);
      if (!seg) continue;
      const evs = d.evs.map(evName).join('", "');
      const hypTxt =
        d.hyp === '*' ? 'every hypothesis in scope' : `hypothesis "${hypNames.get(d.hyp) ?? d.hyp}"`;
      li++;
      checks.push({
        id: `L${li}`,
        text:
          `${`L${li}`} [LICENSE INDEPENDENCE]: The program asserts that "${sentText(seg)}" ` +
          `establishes that "${evs}" are probabilistically independent (knowing one ` +
          `does not change the probability of the other) under ${hypTxt}, for every ` +
          `${itemNoun ?? 'item'} in the queried population${ctxTxt(d.ctx)} — unconditionally, with no ` +
          `subset, exception, modal, or tense restriction. FAIL if ANY clause anywhere ` +
          `in the passage — including a clause attached to a prior, rate, likelihood, ` +
          `observation, or query sentence — reports that the events are linked, forced ` +
          `to co-occur, or barred from co-occurring, however it is attached.` +
          markerTxt(sentText(seg)),
      });
    }
  }

  // (b) population/context membership of the queried item.
  const obsSents = [];
  for (const d of prog.decls) {
    if (d.op === 'OBS' && d.span) {
      const t = sentText(sentenceAt(X, d.span.s));
      if (t && !obsSents.includes(t)) obsSents.push(t);
    }
  }
  const qPos = X.lastIndexOf('?');
  let qSent = qPos > 0 ? sentText(sentenceAt(X, qPos)) : null;
  if (!qSent) {
    const names = entityNames(X, prog);
    for (const s of segs) {
      if (sentenceMateriality(s, names).kind === 'query') qSent = sentText(s);
    }
  }
  const premiseCtxs = [
    ...new Set(
      prog.decls
        .filter((d) => d.op !== 'SENT_TAG' && d.ctx && d.ctx !== 'ctx0' && d.ctx !== '*')
        .map((d) => d.ctx.replace(/^ctx_/, '').replace(/_/g, ' '))
    ),
  ];
  li = 0;
  checks.push({
    id: 'M1',
    text:
      `M1 [POPULATION]: The item described in ${obsSents.map((s) => `"${s}"`).join(' and ') || 'the observation'} ` +
      `${qSent ? `and asked about in "${qSent}" ` : ''}affirmatively belongs to the ` +
      `exact same population and context (shift, site, cohort, stratum, condition) ` +
      `as every premise applied to it — the queried population is ${popDesc}` +
      (premiseCtxs.length ? ` — i.e. the "${premiseCtxs.join('", "')}" population` : '') +
      ` — it is NOT negated or excluded from that population (e.g. "is not a ` +
      `night-shift panel", "is outside the night shift", "non-night-shift"), and ` +
      `no premise quantifies over a different or foreign population — a rule about ` +
      `"${itemNoun && itemNoun !== 'panel' ? 'panels' : 'prototypes'}", "clinics", "crews", "units" or any noun other than the queried ` +
      `"${itemNoun ?? 'item'}s" cannot serve this query unless the passage explicitly equates them.` +
      negTxt,
  });

  // (c) FLUFF sentences are inert: nothing they say may restrict the item
  // or any premise. One proven exception: a sentence the deterministic
  // layer has already proved only remarks that an unobserved, unqueried
  // physical attribute was not inspected (irrelevantUncheckedSeg: zero
  // probability literals, zero queried/witness-path events, zero
  // data/claim/retraction nouns, concrete non-anaphoric subject, no
  // load-bearing span in it) is not submitted as an unqualified FLUFF
  // check — judged without the queried-event scope, "hasn't been
  // checked" reads as an evidence restriction even though it cannot
  // restrict the distribution or observation of any queried event.
  // Every other FLUFF sentence, and any unchecked remark touching a
  // queried event or a load-bearing premise, is still checked in full.
  for (const d of prog.decls) {
    if (d.op !== 'SENT_TAG' || d.tag !== 'FLUFF') continue;
    const seg = segs[d.idx];
    if (!seg) continue;
    if (irrelevantUncheckedSeg(X, prog, seg)) continue;
    // Scoped FLUFF checks: three deterministically-proved sentence
    // classes stay witness-checked, but under the exact scope the
    // deterministic layer proved — an unrecorded-count remark in a
    // zero-prior program, a qualitative sentence scoped to an
    // unqueried stratum, and an inventory-overview sentence. Judged
    // unscoped, each reads as a restriction it cannot be; judged
    // scoped, any real contradiction of a declared premise still
    // fails the strict AND.
    if (irrelevantUnrecordedPriorSeg(X, prog, seg)) {
      checks.push({
        id: `F${d.idx}`,
        text:
          `F${d.idx} [INERT SENTENCE — UNRECORDED OVERALL COUNT]: Sentence ${d.idx} of the passage, ` +
          `"${sentText(seg)}", notes that no overall marginal count was kept. ` +
          `The program declares NO marginal priors and uses only direct conditional/posterior statements, so this sentence is TRUE only if it does not contradict, retract, or dispute any declared conditional/posterior premise or the queried observation.`,
      });
      continue;
    }
    if (irrelevantOtherStratumSeg(X, prog, seg)) {
      const qCtxName = prog.query.ctx.replace(/^ctx_/, '').replace(/_/g, ' ');
      checks.push({
        id: `F${d.idx}`,
        text:
          `F${d.idx} [INERT SENTENCE — OTHER STRATUM]: Sentence ${d.idx} of the passage, ` +
          `"${sentText(seg)}", is scoped to a different population stratum than the queried stratum ("${qCtxName}"). ` +
          `TRUE only if it does not contradict, retract, or modify any premise applied to "${qCtxName}" or claim that the queried item belongs to that other stratum.`,
      });
      continue;
    }
    if (isInventoryOverviewSentence(seg.text, entityNames(X, prog))) {
      checks.push({
        id: `F${d.idx}`,
        text:
          `F${d.idx} [INERT SENTENCE — INVENTORY OVERVIEW]: Sentence ${d.idx} of the passage, ` +
          `"${sentText(seg)}", is an introductory inventory-overview sentence stating that separate inventories/records are kept. ` +
          `TRUE only if it does not contradict, merge, retract, or restrict any premise or the queried item's population.`,
      });
      continue;
    }
    checks.push({
      id: `F${d.idx}`,
      text:
        `F${d.idx} [INERT SENTENCE]: Sentence ${d.idx} of the passage, ` +
        `"${sentText(seg)}", is ledgered as irrelevant filler. TRUE only if it does ` +
        `not restrict, negate, retract, or qualify the queried item, the ` +
        `observed evidence, or any premise (including a claim that nobody verified, checked, or confirmed it), and introduces no population or ` +
        `condition the premises depend on.`,
    });
  }

  // (c2) Full OBS sentences (Bulma R6): when an OBS span covers its
  // whole sentence, span-interval subtraction sees nothing uncovered,
  // so any OBS sentence mentioning a declared event no OBS in it
  // asserts — or carrying both/signals/indicators/silent/solo/only/
  // neither vocabulary — is put to the witness as a whole sentence.
  let oi = 0;
  for (const oc of obsSentenceChecks(X, prog)) {
    oi++;
    const ledgered = oc.obs.map((d) => `"${evName(d.ev)}" ${d.pol}`).join(', ');
    checks.push({
      id: `O${oi}`,
      text:
        `O${oi} [OBSERVATION SENTENCE]: The entire observation sentence ` +
        `"${sentText(oc.seg)}" is ledgered only as the observation ${ledgered}. ` +
        `For each clause, identify its subject and referents, predicate or state, declared event (if any), time, and polarity, in full sentence and passage context, before matching it to that ledger. ` +
        `A name, pronoun, or physical-location reference (being beside, near, or at a device or indicator) alone does not assert that device's state. ` +
        `TRUE only if everything the full sentence asserts is exactly that ` +
        `observation (plus inert time/place/scenery framing). FAIL if it also ` +
        `states, explicitly or by coreference or a shared or coordinated subject or predicate, any additional, contradictory, or differently timed occurrence or state of ${oc.second.length ? `"${oc.second.join('", "')}" or ` : ''}` +
        `any other declared event, that both events are (or are not) active ` +
        `together, that an event is silent/idle, or that any event occurred ` +
        `solo/only/alone at any time — none of that is captured by the ` +
        `observation decl, however it is attached to the sentence, unless that same subject, state, time, and polarity is exactly ledgered above. ` +
        `A correctly ledgered negative observation and inert scenery do not fail merely for being negative. FAIL if any asserted occurrence or condition is unaccounted for, or if a referent, subject, scope, time, or polarity is unresolved or ambiguous; an ambiguous reading does not establish an exact match. Multiple occurrences pass only if every one is ledgered above.`,
    });
  }

  // (d) Residual/adjacent clauses INSIDE a premise or observation
  // sentence (Bulma R5). A clause that shares its sentence with a
  // grounded decl but is anchored by no decl of its own — parenthetical,
  // comma-absolute, concessive/gerund, or coordinate sister clause — is
  // judged explicitly and on its own text, not through its host
  // sentence's license or observation. It passes only if everything it
  // asserts is captured by a program decl or is semantically inert: a
  // separate-time occurrence of an event, a stated contrary state of
  // the queried or a linked event, or any unledgered restriction,
  // exception, or counterexample to a licensed relation FAILS.
  const declSummary = (d) =>
    [d.op, d.id, d.ev, d.pol, d.a && d.b ? `${d.a} ${d.rel ?? ''} ${d.b}` : null,
     Array.isArray(d.evs) ? `(${d.evs.join(' ')})` : null, d.hyp ? `GIVEN ${d.hyp}` : null]
      .filter(Boolean).join(' ');
  // (d2) Counterexample sweep (Bulma R7): every sentence carrying a
  // concessive clause or a co-/solo-activation report is asked, as a
  // whole sentence, whether ANY clause in it — including one attached
  // to a prior or rate sentence — reports an exception to a licensed
  // universal rule. Emitted only when the program actually licenses a
  // relation the report could defeat.
  const licenses = prog.decls.filter((d) => d.op === 'DET_LINK' || d.op === 'COND_INDEP');
  if (licenses.length > 0) {
    const licenseList = licenses.map((d) => {
      if (d.op !== 'DET_LINK') return `independence of "${d.evs.map(evName).join('", "')}"`;
      const quoted = d.span ? X.slice(d.span.s, d.span.e) : '';
      return `${d.rel} between "${evName(d.a)}" and "${evName(d.b)}"` +
        (quoted ? `, stated in the passage as ${JSON.stringify(quoted)}` : '');
    }).join('; ');
    let ci = 0;
    for (const seg of counterexampleSweepSentences(X, prog)) {
      ci++;
      checks.push({
        id: `C${ci}`,
        text:
          `C${ci} [COUNTEREXAMPLE SWEEP]: Passage sentence ${seg.idx} ("${sentText(seg)}") ` +
          `is selected for a whole-sentence counterexample check; this selection does ` +
          `not by itself assert that a counterexample occurs in it. TRUE only if NO ` +
          `clause in this sentence reports an actual occurrence that is an exception or ` +
          `counterexample to any licensed universal rule (${licenseList}). For each ` +
          `binary rule, first extract from its quoted passage wording the actual ` +
          `event states or predicate it links or makes mutually exclusive, then ` +
          `judge only those event states: component names may denote objects, and ` +
          `physical co-presence of objects is not the linked event unless the ` +
          `passage defines presence itself as that event. Compare what the reported ` +
          `occurrence actually asserts about EACH such event state, for the same ` +
          `item at the same time — both events ` +
          `occurring together defeats a mutual-exclusion rule; one event occurring ` +
          `alone, on its own, or without the other defeats an equivalence rule and ` +
          `defeats an implication when the solo event is its antecedent; a stated ` +
          `linkage or forced/barred co-occurrence defeats an independence license. ` +
          `FAIL if any clause anywhere in this sentence reports such an occurrence at ` +
          `any time, however it is attached to its host sentence. ` +
          `Identify each alleged counterexample by its grammatical subject, the ` +
          `event occurrence asserted, the item it concerns, and its time, ` +
          `separately from any source or reporting verb and from a relative ` +
          `clause describing only a physical arrangement; mere location or ` +
          `mention of the named items alone does not assert that either event ` +
          `occurred, while actual or ambiguous activation, unledgered material, ` +
          `mixed extra assertions, or an explicit exception still make this FALSE; ` +
          `if the asserted state of either rule event is unclear or genuinely ` +
          `contradicts the quoted rule, this is also FALSE.`,
      });
    }
  }

  let ri = 0;
  for (const f of adjacentClauseFindings(X, prog)) {
    ri++;
    const clause = f.piece.text.trim().replace(/\s+/g, ' ');
    const host = sentText(segs[f.segIdx]);
    const overlapTxt = f.overlapping.length
      ? f.overlapping.map(declSummary).join('; ')
      : 'none';
    checks.push({
      id: `R${ri}`,
      text:
        `R${ri} [RESIDUAL CLAUSE]: The clause "${clause}" sits inside passage sentence ` +
        `${f.segIdx} ("${host}") but no program decl is anchored in it (decls merely ` +
        `overlapping it: ${overlapTxt}). TRUE only if every assertion this clause makes ` +
        `on its own — about "${[...evNameById.values()].join('", "')}", the hypotheses, ` +
        `or the observed item — is fully captured by a program decl, or the clause is ` +
        `purely inert scenery. FAIL if it states a separate occurrence or a contrary ` +
        `state of any declared event at any time, gives a counterexample to a licensed ` +
        `relation, or adds any restriction, exception, or condition no decl ledgers, ` +
        `however it is attached to the host sentence (parentheses, a comma, a ` +
        `coordinator, or a concessive/gerund phrase).`,
    });
  }
  return checks;
}

function buildPrompt(checks, X) {
  return `You are an independent semantic entailment verifier for a Bayesian extraction pipeline. Verify, from the PASSAGE TEXT ONLY, that every ledgered semantic binding below is unconditionally entailed. Be strict: a single subset, exception, negation, tense shift, modality, or foreign-population reference fails that check. Do not give the program the benefit of the doubt.
Domain semantics: in this pipeline, PRESENT-tense deterministic shared-device wiring ("are wired/connected to the same switch/relay/bus/circuit", "share a relay") IS an unconditional assertion that the two outputs always co-activate — accept it as the claimed EQUIV relation. Its past-tense, hedged, exception-carrying, or scoped forms still fail.
Read the entire passage when judging each check. A later statement that a premise was withdrawn, disputed, unverified, unchecked, or unconfirmed defeats the affected check.

PASSAGE:
${X}

CHECKS:
${checks.map((c) => c.text).join('\n')}

Reply with ONLY a single-line compact JSON object, no markdown, no whitespace:
{"checks":[{"id":"<id>","entailed":true,"reason":"<short>"}],"entailed":true,"code":null,"reason":"<short>"}
- "checks" MUST contain one entry per check id listed above (${checks.map((c) => c.id).join(', ')}). Keep each "reason" under 12 words.
- A check's "entailed" is true only when the passage affirmatively and unconditionally supports it.
- Top-level "entailed" is true ONLY if every check is true.
- If any check fails, "code" is one of: SCOPE_EXCEPTION, TENSE_MISMATCH, MODAL_WRAPPER, NEGATED_MEMBERSHIP, FOREIGN_POPULATION, FLUFF_RESTRICTION, NOT_ENTAILED; else null.`;
}

function extractJsonObject(text) {
  const start = text.indexOf('{');
  if (start < 0) return null;
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/**
 * checkSemanticEntailment(program, X, {model, host, allowCustomWitnessEndpoint, timeoutMs, numPredict})
 *   -> Entailment Witness Receipt
 * { witnessModel, witnessHost, passageSha256, programCanonicalSha256, entailed, code,
 *   reason, checks:[{id,entailed,reason}] }
 * One compact batch call per program. Never throws — transport or malformed
 * output returns entailed:false code ENTAILMENT_WITNESS_UNAVAILABLE so every
 * failure mode stays fail-closed.
 */
export async function checkSemanticEntailment(program, X, opts = {}) {
  const model = ownOpt(opts, 'model', WITNESS_MODEL);
  const host = ownOpt(opts, 'host', DEFAULT_HOST);
  const allowCustomWitnessEndpoint = ownOpt(opts, 'allowCustomWitnessEndpoint', false);
  const timeoutMs = ownOpt(opts, 'timeoutMs', 120000);
  const numPredict = ownOpt(opts, 'numPredict', 1200);
  const prog = toProgram(program, X.length);
  const base = {
    witnessModel: model,
    witnessHost: host,
    passageSha256: passageSha256(X),
    programCanonicalSha256: programCanonicalSha256(prog),
    checks: [],
  };
  const unavailable = (reason) => issueWitness({
    ...base,
    entailed: false,
    code: 'ENTAILMENT_WITNESS_UNAVAILABLE',
    reason,
  });
  if (unapprovedEndpoint(model, host, allowCustomWitnessEndpoint)) return unavailable(ENDPOINT_OPT_IN);
  const doubt = findPoisoningPremiseDoubt(X, prog);
  if (doubt) {
    return issueWitness({
      ...base,
      entailed: false,
      code: 'SCOPED_OR_RETRACTED_PREMISE',
      reason: `premise doubt or retraction: "${doubt.marker}"`,
    });
  }
  if (prog.decls.some((d) =>
    d.op === 'DET_LINK' && d.rel === 'MUTEX' && d.span &&
    HEDGED_NEVER.test(sentenceAt(X, d.span.s)?.text ?? '')
  )) {
    return issueWitness({
      ...base,
      entailed: false,
      code: 'MODAL_WRAPPER',
      reason: 'a hedged never does not rule out co-occurrence',
    });
  }
  // Deterministic residual-clause witness (Bulma R5): a material clause
  // inside a premise/observation sentence that no decl is anchored in
  // is unentailed BY CONSTRUCTION — it is still emitted to the model as
  // an explicit R-check by buildChecks, but the model cannot pardon
  // content the program never ledgered: the deterministic finding is
  // AND-ed into the final verdict below even under an endpoint that
  // would approve everything.
  const residual = adjacentClauseFindings(X, prog);
  // Deterministic license-counterexample pre-check (Bulma R7): a
  // reported co-/solo-activation that contradicts a DET_LINK on its
  // face is unentailed by construction — still swept to the model as
  // an explicit C-check, but AND-ed into the verdict below so an
  // approving endpoint cannot pardon it.
  const counterex = licenseCounterexampleFindings(X, prog);
  const claims = buildChecks(prog, X);
  let content;
  try {
    const res = await fetch(`${host}/api/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      signal: AbortSignal.timeout(timeoutMs),
      body: JSON.stringify({
        model,
        stream: false,
        think: false,
        format: 'json',
        messages: [{ role: 'user', content: buildPrompt(claims, X) }],
        options: { temperature: 0, seed: 0, num_predict: numPredict },
      }),
    });
    if (!res.ok) return unavailable(`ollama ${res.status}: ${(await res.text()).slice(0, 200)}`);
    content = (await res.json()).message?.content ?? '';
  } catch (e) {
    return unavailable(`entailment witness call failed: ${e.message}`);
  }
  let parsed = null;
  try {
    const obj = extractJsonObject(content);
    parsed = obj ? JSON.parse(obj) : null;
  } catch {
    parsed = null;
  }
  if (!parsed || typeof parsed !== 'object' || typeof parsed.entailed !== 'boolean') {
    return unavailable(`malformed witness output: ${content.slice(0, 160)}`);
  }
  const byId = new Map();
  const outChecks = [];
  if (Array.isArray(parsed.checks)) {
    for (const c of parsed.checks) {
      if (c && typeof c.id === 'string') {
        byId.set(c.id, !byId.has(c.id) && c.entailed === true);
        outChecks.push({ id: c.id, entailed: c.entailed === true, reason: String(c.reason ?? '') });
      }
    }
  }
  // strict AND: top-level entailed AND every emitted check id entailed
  // AND no deterministic residual-clause finding. A check id the model
  // dropped is not a pass — fail closed.
  const perCheck = claims.every((c) => byId.get(c.id) === true);
  let entailed = parsed.entailed === true && perCheck;
  let refusalCode = typeof parsed.code === 'string' ? parsed.code : 'NOT_ENTAILED';
  let reason = String(parsed.reason ?? '');
  let finalChecks = outChecks;
  if (residual.length > 0) {
    entailed = false;
    refusalCode = 'RESIDUAL_CLAUSE_NOT_ENTAILED';
    reason =
      `material residual clause inside sentence ${residual[0].segIdx} is anchored by ` +
      `no decl: "${residual[0].piece.text.trim().replace(/\s+/g, ' ').slice(0, 90)}"`;
    finalChecks = residual.map((f, i) => ({
      id: `R${i + 1}`,
      entailed: false,
      reason: `unledgered clause in sentence ${f.segIdx}`,
    })).concat(outChecks.filter((c) => !/^R\d+$/.test(c.id)));
  } else if (counterex.length > 0) {
    entailed = false;
    refusalCode = 'LICENSE_COUNTEREXAMPLE';
    reason =
      `reported occurrence in sentence ${counterex[0].segIdx} contradicts the ` +
      `${counterex[0].rel} license: "${counterex[0].piece.text.trim().replace(/\s+/g, ' ').slice(0, 90)}"`;
    finalChecks = counterex.map((f, i) => ({
      id: `C${i + 1}`,
      entailed: false,
      reason: `counterexample to ${f.rel} in sentence ${f.segIdx}`,
    })).concat(outChecks.filter((c) => !/^C\d+$/.test(c.id)));
  }
  return issueWitness({
    ...base,
    entailed,
    code: entailed ? null : refusalCode,
    reason,
    checks: finalChecks,
  });
}

/**
 * verifyEntailmentWitness(witness, prog, X, config) -> {ok:true} | {ok:false, code, detail}
 * Only the original receipt issued in this process is authenticated. Its
 * SHA-256 bindings restrict it to this passage and canonical program.
 */
export function verifyEntailmentWitness(witness, prog, X, opts = {}) {
  const model = ownOpt(opts, 'model', WITNESS_MODEL);
  const host = ownOpt(opts, 'host', DEFAULT_HOST);
  const allowCustomWitnessEndpoint = ownOpt(opts, 'allowCustomWitnessEndpoint', false);
  if (unapprovedEndpoint(model, host, allowCustomWitnessEndpoint)) {
    return { ok: false, code: 'ENTAILMENT_WITNESS_UNAVAILABLE', detail: ENDPOINT_OPT_IN };
  }
  if (!witness || typeof witness !== 'object') {
    return { ok: false, code: 'ENTAILMENT_WITNESS_UNAVAILABLE', detail: 'no entailment witness supplied' };
  }
  if (!SealedWitnessReceipt.isIssued(witness)) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_UNAUTHENTICATED',
      detail: 'witness was not issued by checkSemanticEntailment in this process',
    };
  }
  if (witness.witnessModel !== model) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: `witness issued by "${witness.witnessModel}", expected "${model}"`,
    };
  }
  if (witness.witnessHost !== host) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: 'witness host does not match the configured endpoint',
    };
  }
  if (witness.passageSha256 !== passageSha256(X)) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: 'witness passageSha256 does not bind this passage',
    };
  }
  if (witness.programCanonicalSha256 !== programCanonicalSha256(prog)) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: 'witness programCanonicalSha256 does not bind this program',
    };
  }
  if (witness.entailed !== true) {
    return {
      ok: false,
      code: witness.code === 'ENTAILMENT_WITNESS_UNAVAILABLE' ? witness.code : 'SEMANTIC_NOT_ENTAILED',
      detail:
        `independent entailment witness did not certify the program ` +
        `(${witness.code ?? 'NOT_ENTAILED'}${witness.reason ? `: ${witness.reason}` : ''})`,
    };
  }
  return { ok: true };
}

/**
 * solveWithWitness(program, X, opts) -> decision + entailmentWitness
 * The fail-closed public solve: the credal kernel decides, then an ANSWER
 * may only release behind a verified independent entailment witness.
 * opts:
 *   offlineSyntheticBench === true  — pure offline synthetic unit bench:
 *                                     no model call, no requirement.
 *   requireEntailmentWitness === false — relax the always-on rule to
 *                                        programNeedsEntailment(prog).
 *   entailmentWitness               — a precomputed receipt (re-verified).
 *   model, host, timeoutMs          — witness call configuration.
 *   allowCustomWitnessEndpoint      — opt in to a nondefault host or model.
 */
export async function solveWithWitness(program, X, opts = {}) {
  const prog = toProgram(program, X.length);
  const decision = solve(prog, X);
  if (decision.status === 'ANSWER') {
    const { nameById, evNameById } = nameMaps(X, prog);
    const ownershipFailures = ownershipCheck({
      X,
      decls: prog.decls,
      nameById,
      evNameById,
      prog,
    }).filter((f) => f.code === 'UNSTABLE_BINDING');
    if (ownershipFailures.length > 0) {
      return {
        status: 'REFUSED',
        code: 'UNSTABLE_BINDING',
        details: {
          detail: ownershipFailures[0].detail,
          ownershipFailures,
        },
        entailmentWitness: null,
      };
    }
  }
  if (decision.status !== 'ANSWER' || ownOpt(opts, 'offlineSyntheticBench', false) === true) {
    return { ...decision, entailmentWitness: null };
  }
  if (ownOpt(opts, 'requireEntailmentWitness', true) === false && !programNeedsEntailment(prog, X)) {
    return { ...decision, entailmentWitness: null };
  }
  // forward only own, sanitized fields — never the caller's raw object
  const config = {
    model: ownOpt(opts, 'model', WITNESS_MODEL),
    host: ownOpt(opts, 'host', DEFAULT_HOST),
    allowCustomWitnessEndpoint: ownOpt(opts, 'allowCustomWitnessEndpoint', false) === true,
    timeoutMs: ownOpt(opts, 'timeoutMs', 120000),
    numPredict: ownOpt(opts, 'numPredict', 1200),
  };
  const supplied = ownOpt(opts, 'entailmentWitness', null);
  const witness =
    supplied ??
    (await checkSemanticEntailment(prog, X, config));
  const v = verifyEntailmentWitness(witness, prog, X, config);
  if (!v.ok) {
    return {
      status: 'REFUSED',
      code: v.code,
      details: {
        detail: v.detail,
        witnessCode: witness?.code ?? null,
        reason: witness?.reason ?? null,
      },
      entailmentWitness: witness,
    };
  }
  return { ...decision, entailmentWitness: witness };
}
