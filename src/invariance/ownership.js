// invariance-gate :: source-grounded OWNERSHIP WITNESS.
//
// Numeric invariance (values track literal sites across permutations) cannot
// see a PERSISTENT misbinding: an extractor that systematically assigns the
// wrong owner to every site produces a self-consistent program whose wrong
// answer releases on the original AND every counterfactual. This witness
// re-derives each bound literal's grammatical owner from the source text —
// via clause/constituent span containment and governor-verb proximity /
// direction — and refuses when the declared HypId is not that owner.
//
// Deterministic reference implementation for Phase-0. The production head
// is neural; this covers the clause forms the corpus and gate probes use:
//   "<name> accounts for <lit>"          (subject-verb owner)
//   "<lit> of cases are <name>"          (copula-after owner)
//   "<lit> of <name> <plural-noun>"      (possessive-of owner)
//   "<lit> (are|is|was|were) <name>"     (copula-after owner)
//   "<name> <lit>" / "<name> for <lit>"  (adjacency owner)
//   "the <lit> share (belongs to|for) <name>"  (possessive owner)
//   "when|under|given|among|for <name>"  (conditioner — LIK only)
//   "<name>, <offset-subclause>, <gov> <lit>"  (main-clause subject)
//
// It also supplies the two direction-witness helpers: a literal whose clause
// asserts P(hyp | filter) may not be bound to LIK, and a literal whose
// clause asserts P(ev | hyp) or P(hyp) may not be bound to POSTERIOR_STATED.

import * as R from '../kernel/rational.js';
import { literalsInSpan, scanLiterals } from '../kernel/literals.js';
import { sentenceBounds } from './ablation.js';
import {
  ctxLabels, contextLabelPolarity, findPremiseDoubt, findPoisoningPremiseDoubt,
  premiseDoubt, inflectedNameRe,
} from '../kernel/sentences.js';

const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);
// premise decls whose span must live in the query's population — numeric
// factors AND the linkage/independence licenses that authorize them
const POPULATION_OPS = new Set([...NUMERIC_OPS, 'JOINT_LIK', 'COND_INDEP', 'DET_LINK']);
const RESTRICTED_OPS = new Set(['DET_LINK', 'COND_INDEP', 'LIK']);

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Segment a sentence into clause constituents: maximal runs bounded by
// (), commas, semicolons, colons, em/en dashes. Parenthesized text forms a
// deeper-depth constituent of its own.
function constituents(sent) {
  const segs = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < sent.length; i++) {
    const c = sent[i];
    if (c === '(' || c === ')' || c === ',' || c === ';' || c === ':' || c === '\u2014' || c === '\u2013') {
      if (i > start) segs.push({ s: start, e: i, depth });
      if (c === '(') depth++;
      else if (c === ')') depth = Math.max(0, depth - 1);
      start = i + 1;
    }
  }
  if (sent.length > start) segs.push({ s: start, e: sent.length, depth });
  return segs;
}

const SUBORDINATE = /^\s*(?:compared to|unlike|versus|vs\.?|which|where(?:as)?|while|though|although|if|when|as)\b/i;

// The hypothesis names a clause attributes a literal to, by governor rules.
// Returns the FIRST matching rule's name, or null when indeterminate.
function ownerOfLiteral({ X, site, nameAlt, kind }) {
  const N = `\\b(${nameAlt})\\b`;
  const sb = sentenceBounds(X, site.s);
  const sent = X.slice(sb.s, sb.e);
  const off = sb.s;
  const segs = constituents(sent);
  const rel = site.s - off;
  const litLen = site.e - site.s;
  const loc =
    segs.find((g) => rel >= g.s && rel + litLen <= g.e) ??
    segs.find((g) => rel >= g.s && rel < g.e);
  const localSeg = loc ?? { s: 0, e: sent.length, depth: 0 };
  const local = sent.slice(localSeg.s, localSeg.e);
  const litRel = rel - localSeg.s;
  const before = local.slice(0, litRel);
  const after = local.slice(litRel + litLen);
  const match = (re, str) => {
    const m = str.match(re);
    return m ? m[1] : null;
  };
  const SUBJ_GOV =
    'made\\s+up|makes?\\s+up|accounts?\\s+for|accounted\\s+for|represents?|represented|compris\\w+|constitut\\w+|holds?|held|is|was|were|are|had|has|have|got|shows?|showed|displays?|yields?|produces?|hits?|reads?';
  const COPULA = 'are|is|were|was|compris\\w+|constitut\\w+|makes?\\s+up';
  const ART = '(?:\\s+(?:a|an|the|about|roughly|approximately|around|nearly|some|exactly|just))*';

  // R1: "... share belongs to <name>" / "belongs to <name>"
  let m = match(new RegExp(`\\bbelongs?\\s+to\\s+${N}`, 'i'), local);
  if (m) return m;
  // R2: "<lit> share (for|of|belongs to) <name>"
  m = match(new RegExp(`\\bshare\\s+(?:for|of|to|went\\s+to|goes\\s+to)\\s+${N}`, 'i'), local);
  if (m) return m;
  // R3: "<name> <subject-governor> <lit>" — owner precedes the verb
  m = match(new RegExp(`${N}\\s+(?:${SUBJ_GOV})${ART}\\s*$`, 'i'), before);
  if (m) return m;
  // R3b: active production/origin verb — "<name> made <lit>",
  // "<name> made the other <lit>" ("Alpha made two thirds of them",
  // "Beta made the other third"). The literal immediately follows the
  // verb phrase at the end of `before`.
  m = match(
    new RegExp(
      `${N}\\s+(?:made|makes|produced|produces|printed|prints|built|builds|supplied|supplies|repaired|repairs)\\s+(?:the\\s+(?:other|remaining)\\s+)?$`,
      'i'
    ),
    before
  );
  if (m) return m;
  // R4: "<lit> of <name>" — possessive-of owner ("nine tenths of Cedar pallets")
  m = match(new RegExp(`^\\s*of\\s+${N}`, 'i'), after);
  if (m) return m;
  // R4b: pre-nominal / possessive — "<lit> [of] [the] <name> <noun>"
  // ("half the Alpha panels", "a quarter of the Beta panels",
  // "one quarter of the Alpha units"). Requires a following noun word
  // so a bare trailing name keeps falling through to the copula rules.
  m = match(
    new RegExp(`^\\s*(?:of\\s+)?(?:the\\s+)?${N}\\s+[a-z][a-z-]*\\b`, 'i'),
    after
  );
  if (m) return m;
  // R4b2: possessive chain — "<lit> of [the <place>'s | its | their]
  // <name> <noun>" ("one quarter of the upper bay's copper units", "half
  // of its tin units"). The possessive frames the population; the name
  // still directly pre-modifies the counted noun.
  m = match(
    new RegExp(
      `^\\s*of\\s+(?:(?:the\\s+)?[a-z]+(?:\\s+[a-z]+)?['’]s|its|their)\\s+${N}\\s+[a-z][a-z-]*\\b`,
      'i'
    ),
    after
  );
  if (m) return m;
  // R4c: passive/origin prepositional phrase — "<lit> ... (printed on|
  // came off|came from|on|off|from|in|at) [the] <name> ..." ("half printed
  // on the Alpha press", "half off the Beta press", "Half the units came
  // off the Alpha line", "Half the cloth units in a bin came off the
  // Alpha line", "Half of the vessels in a crate were fired in the Alpha
  // kiln", "Two thirds of the lots were cast at the Alpha foundry"). At
  // most ten plain words may intervene; an intervening word is never a
  // hypothesis name, and [a-z]+ words never cross a [,;.] clause boundary.
  m = match(
    new RegExp(
      `^\\s*(?:(?:are|were|is|was)\\s+)?(?:(?!(?:${nameAlt})\\b)[a-z]+\\s+){0,10}?(?:printed\\s+on|came\\s+off|came\\s+from|on|off|from|in|at)\\s+(?:the\\s+)?${N}`,
      'i'
    ),
    after
  );
  if (m) return m;
  // R4d: passive-agent / elided-passive owner — "<lit> [(were|was|are|
  // is)] [<verb>] by <name>" ("seven elevenths were repaired by Alpha",
  // "four elevenths by Beta"). Anchored at the literal with at most a
  // copula plus one plain verb word intervening (never a hypothesis
  // name), so in a coordinated pair each literal takes its own nearest
  // agent and a wrong-owner binding still refuses. Direction guards
  // are separate: an event-conditioned "by <name>" share bound as an
  // unconditional PRIOR still refuses as a subgroup share below.
  m = match(
    new RegExp(
      `^\\s*(?:(?:were|was|are|is)\\s+)?(?:(?!(?:${nameAlt})\\b)[a-z]+\\s+)?by\\s+${N}`,
      'i'
    ),
    after
  );
  if (m) return m;
  // R5: "<lit> ... (are|is|was|were) <name>" — copula-after owner
  // ("80% of that pile is Flint", "40% of cases are Alpha", "20% is Moss")
  m = match(new RegExp(`^[^.;]{0,40}?\\b(?:${COPULA})\\s+(?:each\\s+)?${N}\\b`, 'i'), after);
  if (m) return m;
  // R6: "<lit> <name>" — immediate name-after adjacency ("40% Ash", "60% Fir")
  m = match(new RegExp(`^\\s*${N}`, 'i'), after);
  if (m) return m;
  // R7: "<name> <lit>" / "<name> for <lit>" — name-before adjacency
  // ("Teal 75%", "Jade 80%", "Basalt for two thirds")
  m = match(new RegExp(`${N}\\s*(?:,|:)?\\s*(?:for\\s+)?$`, 'i'), before);
  if (m) return m;
  // R8 (LIK only): conditioning governor — nearest "when|under|given|among|
  // for <hypName>" left of the literal in the sentence.
  if (kind === 'LIK') {
    const condRe = new RegExp(
      `\\b(?:when|under|given|among|for|if)\\s+(?:the\\s+\\w+\\s+(?:is|was|were)\\s+)?${N}\\b`,
      'gi'
    );
    let last = null;
    let cm;
    const leftText = sent.slice(0, rel);
    while ((cm = condRe.exec(leftText)) !== null) last = cm[1];
    if (last) return last;
  }
  // R9: exactly one hypothesis name inside the local constituent.
  {
    const inLocal = [];
    for (const nm of allNameMatches(sent.slice(localSeg.s, localSeg.e), nameAlt)) {
      inLocal.push(nm);
    }
    if (inLocal.length === 1) return inLocal[0];
  }
  // R10: main-clause subject — a depth-0 hypothesis name outside the local
  // constituent, not inside a subordinate/appositive clause and not a name
  // that owns a different literal. Nearest one to the left wins.
  {
    const litSpans = scanLiterals(sent).map((l) => [l.s, l.e]);
    const cands = [];
    for (const nm of allNameMatches(sent, nameAlt, true)) {
      const seg = segs.find((g) => nm.pos >= g.s && nm.pos < g.e);
      if (!seg || seg.depth !== 0) continue;
      if (seg.s === localSeg.s && seg.e === localSeg.e) continue;
      const segText = sent.slice(seg.s, seg.e);
      if (SUBORDINATE.test(segText)) continue;
      // a name in a constituent carrying its own literal owns that literal
      if (litSpans.some(([ls, le]) => ls >= seg.s && le <= seg.e)) continue;
      cands.push(nm);
    }
    const left = cands.filter((c) => c.pos < rel);
    if (left.length > 0) return left[left.length - 1].name;
    if (cands.length === 1) return cands[0].name;
  }
  return null;
}

// All occurrences of the declared names in `text`; withPos returns {name,pos}.
function allNameMatches(text, nameAlt, withPos = false) {
  const re = new RegExp(`\\b(${nameAlt})\\b`, 'gi');
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push(withPos ? { name: m[1], pos: m.index } : m[1]);
  }
  return out;
}

function semanticKey(d) {
  if (d.op === 'PRIOR') return `PRIOR|${d.hyp}|${d.ctx}`;
  if (d.op === 'LIK') return `LIK|${d.ev}|${d.pol}|${d.hyp}|${d.ctx}`;
  if (d.op === 'POSTERIOR_STATED') return `POST|${d.hyp}|${d.ev}|${d.pol}|${d.ctx}`;
  return `${d.op}|${d.id ?? ''}`;
}

const EVENT_PREDICATE = /\b(?:is|was|are|were|reads?|shows?|displays?|yields?|registers?|tests?|sounds?|lights?|turns?)\b/i;
const INCIDENTAL_PREP = /(?:\b(?:beside|next\s+to|adjacent\s+to|near(?:\s+to)?|behind|in\s+front\s+of|above|below|under|over|by|at|on|with|without|across\s+from|opposite(?:\s+to)?|(?:to\s+the\s+)?(?:left|right)\s+of|between)\s+(?:the\s+|a\s+|an\s+)?(?:[\w'-]+\s+)*)$/i;
const EVENT_AMBIGUOUS = '\u0000ambiguous-event-owner';
const PREPOSITION =
  'next\\s+to|adjacent\\s+to|in\\s+front\\s+of|across\\s+from|beside|between|behind|above|below|under|over|near|opposite|without|with|from|for|by|at|on|in|of|to';
const PURE_COORD_GAP =
  /^\s*(?:,\s*)?(?:and|or)\s+(?:(?:the|a|an)\s+)?(?:[\w'-]+\s+)*$/i;

// ---------------------------------------------------------------------------
// Unambiguous bare head-noun anaphora: an event introduced with a modifier
// ("tiny pit", "small crack", "red lamp") may later be referred to by its
// bare head noun alone ("A pit showed up ...", "The crack appeared ...").
// The bare head matches ONLY when no other declared event shares that head
// noun (so with "red lamp" and "blue lamp" both declared, a bare "lamp"
// stays ambiguous and grounds nothing) and the head noun is not a generic
// container/classifier word.
// ---------------------------------------------------------------------------
const GENERIC_HEAD_STOP = new Set([
  'test', 'rate', 'unit', 'item', 'case', 'mark', 'sign', 'check',
  'value', 'part', 'side', 'type', 'kind', 'group', 'set', 'box',
  'bag', 'tin', 'lot',
]);

const eventHeadOf = (name) => {
  const toks = String(name ?? '').trim().split(/[\s-]+/).filter(Boolean);
  return toks.length ? toks[toks.length - 1] : null;
};

// id -> head noun for every event whose bare head may stand for it.
function bareHeadById(evNameById) {
  const entries = [...(evNameById?.entries() ?? [])].filter(([, nm]) => nm);
  const headCount = new Map();
  for (const [, nm] of entries) {
    const h = eventHeadOf(nm)?.toLowerCase();
    if (h) headCount.set(h, (headCount.get(h) ?? 0) + 1);
  }
  const out = new Map();
  for (const [id, nm] of entries) {
    const toks = String(nm).trim().split(/[\s-]+/).filter(Boolean);
    if (toks.length < 2) continue; // no modifier: the name is already bare
    const head = toks[toks.length - 1];
    if (!/^[A-Za-z]{2,}$/.test(head)) continue;
    if (GENERIC_HEAD_STOP.has(head.toLowerCase())) continue;
    if (headCount.get(head.toLowerCase()) !== 1) continue; // shared head: ambiguous
    out.set(id, head);
  }
  return out;
}

// Does `text` mention event `id` — by full name, or (when unambiguous) by
// its bare head noun under conservative inflection?
function eventMentioned(text, id, evNameById) {
  const nm = evNameById?.get(id);
  if (!nm) return false;
  if (inflectedNameRe(nm).test(text)) return true;
  const head = bareHeadById(evNameById).get(id);
  return !!head && inflectedNameRe(head).test(text);
}

// Local event attribution for a literal when its sentence names two or
// more events. Returns the event id the literal is predicated of, or null
// when only one event is named or ownership is ambiguous. Conservative:
// an event that merely locates the subject ("the lamp beside the buzzer
// is ...") is not the subject event, while events predicated in separate
// constituents keep their own literals.
function eventOwnerOfLiteral({ X, site, evNameById }) {
  const entries = [...(evNameById?.entries() ?? [])].filter(([, nm]) => nm);
  if (entries.length < 2) return null;
  const sb = sentenceBounds(X, site.s);
  const sent = X.slice(sb.s, sb.e);
  const rel = site.s - sb.s;
  const litLen = site.e - site.s;
  const mentions = [];
  for (const [id, nm] of entries) {
    const re = inflectedNameRe(nm, 'gi');
    let m;
    while ((m = re.exec(sent)) !== null) {
      mentions.push({ id, pos: m.index, end: m.index + m[1].length });
    }
  }
  // Bare head-noun mentions for unambiguously-headed events, dropping any
  // that merely re-count the head inside a full-name mention.
  for (const [id, head] of bareHeadById(evNameById)) {
    const re = inflectedNameRe(head, 'gi');
    let m;
    while ((m = re.exec(sent)) !== null) {
      const pos = m.index;
      const end = pos + m[1].length;
      if (mentions.some((f) => pos < f.end && end > f.pos)) continue;
      mentions.push({ id, pos, end });
    }
  }
  mentions.sort((a, b) => a.pos - b.pos);
  if (new Set(mentions.map((m) => m.id)).size < 2) return null;
  // Coordinated rate phrases ("<event> ... <lit> of ... and <lit> of ...")
  // inherit the first literal's subject event, including an appositive
  // subject in its own comma constituent. Event mentions after the first
  // literal, e.g. in a relative/locative modifier, do not supply it; a new
  // subject or predicate after "and" breaks the coordination, since "and"
  // must immediately precede this literal.
  {
    const lits = scanLiterals(sent);
    const idx = lits.findIndex((l) => l.s === rel);
    const prev = idx > 0 ? lits[idx - 1] : null;
    const gap = prev ? sent.slice(prev.e, rel) : '';
    if (
      prev &&
      /\band\s*$/i.test(gap) &&
      !/[.;:]/.test(gap) &&
      /^\s+of\b/i.test(sent.slice(prev.e)) &&
      /^\s+of\b/i.test(sent.slice(rel + litLen))
    ) {
      const heads = mentions.filter(
        (m) => m.pos < prev.s && !INCIDENTAL_PREP.test(sent.slice(0, m.pos))
      );
      const head = heads[heads.length - 1];
      if (
        head &&
        new Set(heads.map((m) => m.id)).size === 1 &&
        /[\w'-]+/.test(sent.slice(head.end, prev.s))
      ) {
        return head.id;
      }
    }
  }
  const segs = constituents(sent);
  let seg =
    segs.find((g) => rel >= g.s && rel + litLen <= g.e) ??
    segs.find((g) => rel >= g.s && rel < g.e) ??
    { s: 0, e: sent.length, depth: 0 };
  let inSeg = mentions.filter((m) => m.pos >= seg.s && m.pos < seg.e);
  if (new Set(inSeg.map((m) => m.id)).size === 0) {
    seg = { s: 0, e: sent.length, depth: 0 };
    inSeg = mentions;
  }
  if (new Set(inSeg.map((m) => m.id)).size === 1) return inSeg[0].id;
  const segText = sent.slice(seg.s, seg.e);
  const litPos = rel - seg.s;
  const local = inSeg.map((m) => ({ id: m.id, pos: m.pos - seg.s, end: m.end - seg.s }));
  const before = local.filter((m) => m.pos < litPos);
  const incidental = (m) => INCIDENTAL_PREP.test(segText.slice(0, m.pos));
  if (before.length) {
    const prevLitEnd = scanLiterals(segText).reduce(
      (mx, l) => (l.e <= litPos ? Math.max(mx, l.e) : mx),
      0
    );
    const scoped = before.filter((m) => m.pos >= prevLitEnd);
    if (!scoped.length) return EVENT_AMBIGUOUS;
    const predRe = new RegExp(EVENT_PREDICATE.source, 'gi');
    let predPos = -1;
    let pm;
    const preLit = segText.slice(prevLitEnd, litPos);
    while ((pm = predRe.exec(preLit)) !== null) predPos = prevLitEnd + pm.index;
    if (predPos < 0) {
      // Transitive object-rate clauses put the predicate before the event:
      // "<subject> <verb> the <event> with probability <lit>", with verb
      // ellipsis in a coordinated second arm ("and the <event> with
      // probability <lit>"). Accept only a single event directly governed
      // as that object and immediately followed by the rate tail; a
      // prepositional/locative object or a second scoped event withholds.
      const RATE_TAIL = /^\s+with\s+(?:a\s+)?probability(?:\s+of)?\s*$/i;
      const isPrepWord = (w) => new RegExp(`^(?:${PREPOSITION})$`, 'i').test(w);
      const objectHead = (from, mention, litStart) => {
        if (!RATE_TAIL.test(segText.slice(mention.end, litStart))) return null;
        const words = [...segText.slice(from, mention.pos).matchAll(/[\w'-]+/gi)].map((t) =>
          t[0].toLowerCase()
        );
        let det = -1;
        for (let i = words.length - 1; i >= 0; i--) {
          if (words[i] === 'the' || words[i] === 'a' || words[i] === 'an') {
            det = i;
            break;
          }
        }
        if (det < 1) return null;
        if (words.slice(det + 1).some((w) => isPrepWord(w) || w === 'and' || w === 'or')) return null;
        return { gov: words[det - 1], govIdx: det - 1, det, words };
      };
      const verbHeaded = (h) =>
        h !== null &&
        h.govIdx >= 1 &&
        !isPrepWord(h.gov) &&
        !/^(?:and|or|but|with|without|probability|positive|negative)$/i.test(h.gov) &&
        !EVENT_PREDICATE.test(h.gov) &&
        !isPrepWord(h.words[h.govIdx - 1]) &&
        !/^(?:the|a|an)$/i.test(h.words[h.govIdx - 1]);
      if (new Set(scoped.map((m) => m.id)).size !== 1) return EVENT_AMBIGUOUS;
      const cand = scoped[scoped.length - 1];
      if (incidental(cand)) return EVENT_AMBIGUOUS;
      const head = objectHead(prevLitEnd, cand, litPos);
      if (!head) return EVENT_AMBIGUOUS;
      if (head.gov === 'and' || head.gov === 'or') {
        if (head.det !== 1 || prevLitEnd <= 0) return EVENT_AMBIGUOUS;
        const lits = scanLiterals(segText).filter((l) => l.e <= litPos);
        const prevIdx = lits.findIndex((l) => l.e === prevLitEnd);
        if (prevIdx < 0) return EVENT_AMBIGUOUS;
        const prevLit = lits[prevIdx];
        const prevStart = prevIdx > 0 ? lits[prevIdx - 1].e : 0;
        const prevMentions = local.filter((m) => m.pos >= prevStart && m.pos < prevLit.s);
        if (new Set(prevMentions.map((m) => m.id)).size !== 1) return EVENT_AMBIGUOUS;
        const prevCand = prevMentions[prevMentions.length - 1];
        if (prevCand.id === cand.id || incidental(prevCand)) return EVENT_AMBIGUOUS;
        return verbHeaded(objectHead(prevStart, prevCand, prevLit.s))
          ? cand.id
          : EVENT_AMBIGUOUS;
      }
      return verbHeaded(head) ? cand.id : EVENT_AMBIGUOUS;
    }
    const subject = scoped.filter((m) => m.pos < predPos);
    if (!subject.length) return EVENT_AMBIGUOUS;
    const frontedPP = (m) => {
      const prefix = segText.slice(prevLitEnd, m.pos).trim();
      return new RegExp(
        `^(?:${PREPOSITION})\\s+(?:(?:the|a|an)\\s+)?(?:(?!the\\b|a\\b|an\\b)[\\w'-]+\\s+)*$`,
        'i'
      ).test(prefix);
    };
    if (new Set(subject.map((m) => m.id)).size === 1) {
      return frontedPP(subject[0]) ? EVENT_AMBIGUOUS : subject[0].id;
    }
    let heads = subject.slice();
    let advanced = true;
    while (advanced && heads.length > 1) {
      advanced = false;
      for (let i = 1; i < heads.length; i++) {
        const gap = segText.slice(heads[i - 1].end, heads[i].pos);
        const newClause =
          EVENT_PREDICATE.test(gap) &&
          /\b(?:and|or|but)\s+(?:(?:the|a|an)\s+)?(?:[\w'-]+\s+)*$/i.test(gap);
        const afterFronted =
          i === 1 &&
          frontedPP(heads[0]) &&
          !new RegExp(`\\b(?:${PREPOSITION})\\b`, 'i').test(gap) &&
          !/\b(?:and|or)\b/i.test(gap) &&
          !/\b[\w'-]+(?:ed|ing)\b/i.test(gap);
        if (newClause || afterFronted) {
          heads = heads.slice(i);
          advanced = true;
          break;
        }
      }
    }
    if (new Set(heads.map((m) => m.id)).size === 1) return heads[0].id;
    if (frontedPP(heads[0])) return EVENT_AMBIGUOUS;
    const firstGap = segText.slice(heads[0].end, heads[1].pos);
    if (PURE_COORD_GAP.test(firstGap)) return EVENT_AMBIGUOUS;
    const modifierIntroduced =
      new RegExp(`\\b(?:${PREPOSITION})\\b`, 'i').test(firstGap) ||
      /\b(?:that|which|who|whose)\b/i.test(firstGap) ||
      /\b[\w'-]+(?:ed|ing)\b/i.test(firstGap) ||
      EVENT_PREDICATE.test(firstGap);
    return modifierIntroduced ? heads[0].id : EVENT_AMBIGUOUS;
  }
  for (let i = 0; i < local.length; i++) {
    const m = local[i];
    if (i > 0 && incidental(m)) continue;
    if (EVENT_PREDICATE.test(segText.slice(m.end))) return m.id;
  }
  const heads = local.filter((m) => !incidental(m));
  return new Set(heads.map((m) => m.id)).size === 1 ? heads[0].id : EVENT_AMBIGUOUS;
}

// Leading literal site whose clause supplies the event inherited by a
// coordinated trailing rate, mirroring the coordinated block above.
function coordinatedRateAnchor({ X, site, evNameById }) {
  const entries = [...(evNameById?.entries() ?? [])].filter(([, nm]) => nm);
  if (entries.length < 2) return null;
  const sb = sentenceBounds(X, site.s);
  const sent = X.slice(sb.s, sb.e);
  const rel = site.s - sb.s;
  const litLen = site.e - site.s;
  const lits = scanLiterals(sent);
  const idx = lits.findIndex((l) => l.s === rel);
  const prev = idx > 0 ? lits[idx - 1] : null;
  if (!prev) return null;
  const gap = sent.slice(prev.e, rel);
  if (
    !/\band\s*$/i.test(gap) ||
    /[.;:]/.test(gap) ||
    !/^\s+of\b/i.test(sent.slice(prev.e)) ||
    !/^\s+of\b/i.test(sent.slice(rel + litLen))
  ) {
    return null;
  }
  const mentions = [];
  for (const [id, nm] of entries) {
    const re = inflectedNameRe(nm, 'gi');
    let m;
    while ((m = re.exec(sent)) !== null) {
      mentions.push({ id, pos: m.index, end: m.index + m[1].length });
    }
  }
  mentions.sort((a, b) => a.pos - b.pos);
  const heads = mentions.filter(
    (m) => m.pos < prev.s && !INCIDENTAL_PREP.test(sent.slice(0, m.pos))
  );
  const head = heads[heads.length - 1];
  if (
    !head ||
    new Set(heads.map((m) => m.id)).size !== 1 ||
    !/[\w'-]+/.test(sent.slice(head.end, prev.s))
  ) {
    return null;
  }
  return {
    eventId: head.id,
    site: { s: sb.s + prev.s, e: sb.s + prev.e, text: sent.slice(prev.s, prev.e) },
  };
}

// ---------------------------------------------------------------------------
// General coordinated two-arm rate grounding. A single-hypothesis sentence
// may state two distinct events' likelihoods in two arms joined by "and"
// ("Among Alpha panels, one in four carry a notch and one in five a
// stripe.", "On Alpha units a notch turns up on 1/4 and a stripe on 1/5.").
// No verb template is assumed: the witness is purely structural — exactly
// one hypothesis named in the sentence, exactly two distinct declared
// events, exactly two probability literals, a coordinating "and" between
// the literal sites, each arm holding exactly one event and one literal,
// and zero negation/doubt/hedging markers. Returns the arm attribution of
// `site` (its event, the sentence's hypothesis, and the sibling site+event
// the matching declaration must also establish), or null.
// evEntries/hypEntries: [{id, name}].
// ---------------------------------------------------------------------------
const COORD_NEG =
  /\b(?:not|no|never|n't|without|except|unless|neither|nor|maybe|perhaps|possibly|unknown|uncertain|might|may|could)\b/i;

// Coordinated matching keeps the strict safe-atom guard: only 1-3-word
// plain alphabetic/hyphenated event names with no negation/logical word
// may take part in two-arm attribution (single-event extraction has no
// such cap — see jsonToProgram in the live extractor).
const COORD_BAD_WORDS = new Set([
  'not', 'no', 'never', 'or', 'if', 'unless', 'except', 'without', 'and', 'but',
]);
function isCoordSafeEventName(name) {
  if (typeof name !== 'string' || !name.trim() || name.trim() !== name) return false;
  const words = name.split(/\s+/);
  if (words.length < 1 || words.length > 3) return false;
  return words.every(
    (w) => /^[A-Za-z]+(?:-[A-Za-z]+)*$/.test(w) &&
      w.split('-').every((p) => !COORD_BAD_WORDS.has(p.toLowerCase()))
  );
}

export function coordinatedArmForSite(X, site, evEntries, hypEntries) {
  const evs = (evEntries ?? []).filter((e) => e && e.name && isCoordSafeEventName(e.name));
  const hyps = (hypEntries ?? []).filter((h) => h && h.name);
  if (evs.length < 2 || hyps.length < 1 || !site) return null;
  const sb = sentenceBounds(X, site.s);
  const sent = X.slice(sb.s, sb.e);
  if (COORD_NEG.test(sent) || premiseDoubt(sent)) return null;
  const lits = scanLiterals(sent).filter((l) => !/^\d+$/.test(l.text));
  if (lits.length !== 2) return null;
  const relS = site.s - sb.s;
  const relE = site.e - sb.s;
  const ownIdx = lits.findIndex((l) => l.s === relS && l.e === relE);
  if (ownIdx < 0) return null;
  const hypHits = hyps.filter((h) =>
    new RegExp(`\\b${esc(h.name)}\\b`, 'i').test(sent)
  );
  if (hypHits.length !== 1) return null;
  const mentionsOf = (text) => {
    const found = new Set();
    for (const e of evs) {
      if (inflectedNameRe(e.name).test(text)) found.add(e.id);
    }
    return found;
  };
  if (mentionsOf(sent).size !== 2) return null;
  const [litA, litB] = lits;
  for (const m of sent.matchAll(/\band\b/gi)) {
    const andS = m.index;
    const andE = andS + m[0].length;
    if (andS < litA.e || andE > litB.s) continue;
    const armEv = [mentionsOf(sent.slice(0, andS)), mentionsOf(sent.slice(andE))];
    if (armEv[0].size !== 1 || armEv[1].size !== 1) continue;
    const [ev0] = armEv[0];
    const [ev1] = armEv[1];
    if (ev0 === ev1) continue;
    const armEvs = [ev0, ev1];
    const sibIdx = 1 - ownIdx;
    return {
      evId: armEvs[ownIdx],
      hypId: hypHits[0].id,
      hypName: hypHits[0].name,
      siblingEvId: armEvs[sibIdx],
      siblingSite: {
        s: sb.s + lits[sibIdx].s,
        e: sb.s + lits[sibIdx].e,
        text: lits[sibIdx].text,
      },
    };
  }
  return null;
}

// English function words are never hypothesis names, even standing alone
// in a narrowed span.
const HYP_NAME_STOP = new Set([
  'the', 'all', 'are', 'or', 'of', 'and', 'every', 'each', 'a', 'an',
  'is', 'was', 'were',
]);

// Extract the hypothesis name grounded by a HYP decl's span. A span already
// narrowed to one clean alphabetic word IS the name, whatever its case
// ("copper", "tin", "Alpha") — lowercase hypotheses ground exactly like
// capitalized ones. A wider span keeps the proper-noun rule: its first
// capitalized token.
export function hypNameFromSpan(X, span) {
  const t = X.slice(span.s, span.e);
  const word = t.trim();
  if (/^[A-Za-z]+(?:-[A-Za-z]+)*$/.test(word) &&
      word.split('-').every((p) => !HYP_NAME_STOP.has(p.toLowerCase()))) {
    return word;
  }
  const m = t.match(/\b([A-Z][a-zA-Z]+)\b/);
  return m && !HYP_NAME_STOP.has(m[1].toLowerCase()) ? m[1] : null;
}

// ---------------------------------------------------------------------------
// ownershipCheck({X, decls, nameById, evNameById})
// For every numeric decl with a bound literal site, verify that the literal's
// grammatical owner in X is the decl's hypothesis — and that the clause's
// conditional direction matches the decl's operator. Returns a list of
// binding or premise-assertion failures (empty = witness held).
// ---------------------------------------------------------------------------
export function ownershipCheck({ X, decls, nameById, evNameById, prog = null }) {
  const failures = [];
  const doubt = prog ? findPoisoningPremiseDoubt(X, prog) : findPremiseDoubt(X);
  if (doubt) {
    failures.push({
      code: 'SCOPED_OR_RETRACTED_PREMISE',
      detail: `source qualifies the premise's assertion: "${doubt.marker}"`,
      span: doubt.span,
    });
  }
  const names = [...new Set([...nameById.values()])].filter(Boolean);
  if (names.length === 0) {
    for (const d of decls) {
      if (!NUMERIC_OPS.has(d.op)) continue;
      if (!d.span && d.op !== 'LIK') continue;
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail: `ownership witness: ${d.op}|${d.hyp} has no name-grounded HYP span`,
        span: d.span,
      });
    }
    return failures;
  }
  const nameAlt = names
    .slice()
    .sort((a, b) => b.length - a.length)
    .map(esc)
    .join('|');
  const nameToId = new Map();
  for (const [id, nm] of nameById) {
    if (nm && !nameToId.has(nm.toLowerCase())) nameToId.set(nm.toLowerCase(), id);
  }
  const selectedSites = [];
  for (const d of decls) {
    if (!NUMERIC_OPS.has(d.op)) continue;
    if (!d.span) {
      if (d.op === 'LIK') {
        failures.push({
          code: 'UNSTABLE_BINDING',
          detail: `ownership witness: LIK ${d.ev ?? '(missing event)'}|${d.hyp} has no span grounding its literal`,
          span: d.span,
        });
      }
      continue;
    }
    const lits = literalsInSpan(X, d.span);
    const hypName = nameById.get(d.hyp);
    const key = `${d.op}|${d.hyp}`;
    if (!hypName) {
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail: `ownership witness: ${key} has no name-grounded HYP span`,
        span: d.span,
      });
      continue;
    }
    const matching = lits.filter((l) => R.eq(l.value, d.value));
    if (matching.length === 0) {
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail: `ownership witness: no literal equal to the declared value for ${key} in its span`,
        span: d.span,
      });
      continue;
    }
    const declEvName =
      d.op === 'LIK' || d.op === 'POSTERIOR_STATED' ? evNameById?.get(d.ev) : null;
    if ((d.op === 'LIK' || d.op === 'POSTERIOR_STATED') && !declEvName) {
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail:
          `event witness: ${d.op} binds ${d.ev ?? '(missing event)'} ` +
          'but has no name-grounded EVENT span',
        span: d.span,
      });
      continue;
    }
    const evEntries = [...(evNameById?.entries() ?? [])].map(([id, name]) => ({ id, name }));
    const hypEntries = [...nameById.entries()].map(([id, name]) => ({ id, name }));
    const candidates = matching.map((candidateSite) => {
      const coord = d.op === 'LIK'
        ? coordinatedArmForSite(X, candidateSite, evEntries, hypEntries)
        : null;
      let owner = ownerOfLiteral({ X, site: candidateSite, nameAlt, kind: d.op });
      let ownerId = owner ? nameToId.get(owner.toLowerCase()) : undefined;
      if (coord) {
        owner = coord.hypName;
        ownerId = coord.hypId;
      }
      const ownerOk =
        !!owner &&
        (ownerId === d.hyp ||
          (ownerId === undefined && owner.toLowerCase() === hypName.toLowerCase()));
      const csb = sentenceBounds(X, candidateSite.s);
      const csent = X.slice(csb.s, csb.e);
      const evOwner = coord
        ? coord.evId
        : declEvName
          ? eventOwnerOfLiteral({ X, site: candidateSite, evNameById })
          : null;
      const coordAnchor = declEvName
        ? coordinatedRateAnchor({ X, site: candidateSite, evNameById })
        : null;
      const eventPresent =
        !declEvName || eventMentioned(csent, d.ev, evNameById);
      const eventOk =
        eventPresent && evOwner !== EVENT_AMBIGUOUS && (!evOwner || evOwner === d.ev);
      // Coordinated arms assert positive occurrence rates: a NEG binding
      // against a coordinated arm literal is a polarity mismatch.
      const polOk = !coord || d.pol === 'POS';
      return {
        site: candidateSite,
        owner,
        ownerId,
        evOwner,
        coord,
        coordAnchor,
        eventPresent,
        compatible: ownerOk && eventOk && polOk,
      };
    });
    const compatibleBySite = new Map();
    for (const c of candidates) {
      if (c.compatible) compatibleBySite.set(`${c.site.s}:${c.site.e}`, c);
    }
    const compatible = [...compatibleBySite.values()];
    if (compatible.length > 1) {
      failures.push({
        code: 'UNSTABLE_BINDING',
        detail: `ownership witness: multiple equal literals in the span for ${key} are compatible; the bound site is ambiguous`,
        span: d.span,
      });
      continue;
    }
    if (compatible.length === 0) {
      const c = candidates[0];
      const diag = c.site;
      let detail;
      if (!c.owner) {
        detail =
          `ownership witness: no grammatical owner for literal "${diag.text}" ` +
          `at [${diag.s},${diag.e}] bound to ${d.hyp} ("${hypName}")`;
      } else if (c.ownerId !== undefined && c.ownerId !== d.hyp) {
        detail =
          `ownership witness: literal "${diag.text}" at [${diag.s},${diag.e}] is ` +
          `grammatically owned by "${c.owner}" (${c.ownerId}) but bound to ${d.hyp} ("${hypName}")`;
      } else if (c.ownerId === undefined && c.owner.toLowerCase() !== hypName.toLowerCase()) {
        detail = `ownership witness: literal "${diag.text}" owned by "${c.owner}" but bound to ${d.hyp} ("${hypName}")`;
      } else if (!c.eventPresent) {
        detail =
          `event witness: ${d.op} binds ${d.ev} ("${declEvName}") but the name ` +
          'never occurs in the literal\'s sentence';
      } else if (c.evOwner === EVENT_AMBIGUOUS) {
        detail =
          `event witness: ${d.op} binds ${d.ev} ("${declEvName}") but literal "${diag.text}" ` +
          `at [${diag.s},${diag.e}] has multiple competing events and no established grammatical head`;
      } else if (c.evOwner && c.evOwner !== d.ev) {
        detail =
          `event witness: ${d.op} binds ${d.ev} ("${declEvName}") but literal "${diag.text}" ` +
          `at [${diag.s},${diag.e}] is locally attributed to ${c.evOwner} ("${evNameById.get(c.evOwner)}")`;
      } else {
        detail = `ownership witness: literal "${diag.text}" at [${diag.s},${diag.e}] is not established for ${key}`;
      }
      failures.push({ code: 'UNSTABLE_BINDING', detail, span: d.span });
      continue;
    }
    const site = compatible[0].site;
    selectedSites.push({
      decl: d,
      key: semanticKey(d),
      site,
      coord: compatible[0].coord ?? null,
      anchor:
        compatible[0].coordAnchor?.eventId === d.ev
          ? compatible[0].coordAnchor
          : null,
    });
    // ---- direction witnesses -------------------------------------------
    const sb = sentenceBounds(X, site.s);
    const sent = X.slice(sb.s, sb.e);
    const hypEsc = esc(hypName);
    if (d.op === 'LIK') {
      // A LIK bound where the clause asserts P(hyp|filter): "<lit> ... are/
      // is <hypname>" — the literal is a population share, not a likelihood.
      const postDir = new RegExp(
        `${esc(site.text)}[^.;]{0,40}?\\b(?:are|is|were|was|compris\\w+|constitut\\w+|makes?\\s+up)\\s+(?:each\\s+)?\\b${hypEsc}\\b`,
        'i'
      );
      if (postDir.test(sent.slice(site.s - sb.s))) {
        failures.push({
          code: 'UNSTABLE_BINDING',
          detail:
            `direction witness: ${d.op} ${d.ev}|${d.hyp} bound at "${site.text}" ` +
            `inside a P(hyp|ev) clause — the literal asserts a population share, not a likelihood`,
          span: d.span,
        });
        continue;
      }
      // Active-voice reversal: "<hypname> accounts for <lit>" inside a
      // sentence whose counted population is event-scoped ("the collection
      // of cards that carry the <ev>", "items bearing the <ev>") asserts
      // P(hyp|ev) — the hypothesis is the subject of the share, not the
      // conditioner of the event.
      const evNm = evNameById.get(d.ev);
      if (evNm) {
        const evEsc = esc(evNm.toLowerCase());
        const evScoped = new RegExp(
          `\\b(?:that|which|who)\\s+(?:[a-z'-]+\\s+){0,4}${evEsc}\\b` +
            `|\\b${evEsc}[\\s-](?:bearing|carrying|carried|marked|tagged|positive|negative|lit|unlit)\\b`,
          'i'
        );
        const hypSubj = new RegExp(
          `\\b${hypEsc}\\s+(?:accounts?\\s+for|accounted\\s+for|makes?\\s+up|made\\s+up|compris\\w+|constitut\\w+|represents?|represented|forms?|formed)\\s+(?:a\\s+|an\\s+|the\\s+)?(?:share\\s+of\\s+|proportion\\s+of\\s+)?${esc(site.text)}`,
          'i'
        );
        if (evScoped.test(sent) && hypSubj.test(sent)) {
          failures.push({
            code: 'UNSTABLE_BINDING',
            detail:
              `direction witness: ${d.op} ${d.ev}|${d.hyp} bound at "${site.text}" ` +
              `inside an event-scoped population share — "${hypName}" is the subject of the ` +
              'percentage, so the clause asserts P(hyp|ev), not P(ev|hyp)',
            span: d.span,
          });
          continue;
        }
      }
    }
    if (d.op === 'POSTERIOR_STATED') {
      // A POST bound where the clause asserts P(ev|hyp) or P(hyp):
      //   "when|under|given|among|for <hypname> ... <lit>"  (conditioner)
      //   "<lit> of <hypname> <plural>"                    (possessive lik)
      //   "<hypname> <subject-governor> <lit>"             (prior share)
      const condDir = new RegExp(
        `\\b(?:when|under|given|among|for|if)\\s+(?:the\\s+\\w+\\s+(?:is|was|were)\\s+)?\\b${hypEsc}\\b`,
        'i'
      );
      const ofDir = new RegExp(
        `${esc(site.text)}\\s+of\\s+\\b${hypEsc}\\b`,
        'i'
      );
      const subjDir = new RegExp(
        `\\b${hypEsc}\\s+(?:made\\s+up|makes?\\s+up|accounts?\\s+for|accounted\\s+for|represents?|holds?|compris\\w+|constitut\\w+|is|was|were|are|had|has|have|shows?|shares?)\\s*(?:a\\s+|an\\s+|the\\s+)?${esc(site.text)}`,
        'i'
      );
      const left = sent.slice(0, site.s - sb.s);
      if (
        condDir.test(left) ||
        ofDir.test(sent.slice(site.s - sb.s)) ||
        subjDir.test(sent)
      ) {
        failures.push({
          code: 'UNSTABLE_BINDING',
          detail:
            `direction witness: ${d.op} ${d.hyp}|${d.ev} bound at "${site.text}" ` +
            'inside a prior/likelihood-direction clause — the text does not state P(hyp|ev) there',
          span: d.span,
        });
        continue;
      }
    }
    if (d.op === 'PRIOR') {
      // An unconditional PRIOR bound where the clause asserts P(hyp|ev):
      // the counted population is restricted by a declared event before
      // the literal ("Of the units with a green mark, ...") and the
      // hypothesis takes the literal as its share of that subgroup
      // ("7/11 came from Alpha", "80% are Alpha"). That is a
      // stated posterior, never a base rate.
      const left = sent.slice(0, site.s - sb.s);
      const after = sent.slice(site.s - sb.s);
      let eventRestricted = false;
      for (const evNm of evNameById?.values() ?? []) {
        if (!evNm) continue;
        const m = inflectedNameRe(evNm).exec(left);
        if (!m) continue;
        const window = left.slice(Math.max(0, m.index - 60), m.index);
        if (/\b(?:of|among|within|with|that|which|bearing|carrying)\b[^.;]*$/i.test(window) ||
            /^\s*(?:of|among|within)\b/i.test(left)) {
          eventRestricted = true;
          break;
        }
      }
      const hypShare = new RegExp(
        `\\b(?:are|is|were|was|by|belongs?\\s+to|went\\s+to|goes\\s+to)\\b[^.;]{0,40}?\\b${hypEsc}\\b`,
        'i'
      );
      if (eventRestricted && hypShare.test(after)) {
        failures.push({
          code: 'UNSTABLE_BINDING',
          detail:
            `direction witness: PRIOR ${d.hyp} bound at "${site.text}" inside an ` +
            'event-conditioned P(hyp|ev) clause — the literal is a subgroup share, not an unconditional base rate',
          span: d.span,
        });
        continue;
      }
    }
  }
  const keysBySite = new Map();
  for (const sel of selectedSites) {
    const k = `${sel.site.s}:${sel.site.e}`;
    if (!keysBySite.has(k)) {
      keysBySite.set(k, { site: sel.site, span: sel.decl.span, keys: new Set() });
    }
    keysBySite.get(k).keys.add(sel.key);
  }
  for (const entry of keysBySite.values()) {
    if (entry.keys.size < 2) continue;
    failures.push({
      code: 'UNSTABLE_BINDING',
      detail:
        `ownership witness: literal "${entry.site.text}" at [${entry.site.s},${entry.site.e}] ` +
        `is reused for distinct factors ${[...entry.keys].join(', ')}`,
      span: entry.span,
    });
  }
  for (const sel of selectedSites) {
    if (!sel.anchor) continue;
    const anchored = selectedSites.some(
      (o) =>
        o !== sel &&
        o.site.s === sel.anchor.site.s &&
        o.site.e === sel.anchor.site.e &&
        o.decl.ev === sel.decl.ev
    );
    if (anchored) continue;
    failures.push({
      code: 'UNSTABLE_BINDING',
      detail:
        `event witness: coordinated literal "${sel.site.text}" at [${sel.site.s},${sel.site.e}] ` +
        `inherits ${sel.decl.ev} from leading literal "${sel.anchor.site.text}" ` +
        `at [${sel.anchor.site.s},${sel.anchor.site.e}], but that leading site is not ` +
        `established by a matching declaration for ${sel.decl.ev}`,
      span: sel.decl.span,
    });
  }
  // Coordinated two-arm completeness: an arm literal is only grounded as
  // one half of the coordinated pair — the sibling arm must be established
  // by a matching POS declaration for the sibling event, under the same
  // hypothesis and context, at the sibling literal's own site.
  for (const sel of selectedSites) {
    if (!sel.coord) continue;
    const sib = sel.coord.siblingSite;
    const established = selectedSites.some(
      (o) =>
        o !== sel &&
        o.site.s === sib.s &&
        o.site.e === sib.e &&
        o.decl.ev === sel.coord.siblingEvId &&
        o.decl.hyp === sel.decl.hyp &&
        o.decl.pol === 'POS' &&
        o.decl.ctx === sel.decl.ctx
    );
    if (established) continue;
    failures.push({
      code: 'UNSTABLE_BINDING',
      detail:
        `event witness: coordinated literal "${sel.site.text}" at [${sel.site.s},${sel.site.e}] ` +
        `is one arm of a coordinated pair, but sibling literal "${sib.text}" ` +
        `at [${sib.s},${sib.e}] is not established by a matching POS declaration ` +
        `for ${sel.coord.siblingEvId} under ${sel.decl.hyp}`,
      span: sel.decl.span,
    });
  }
  return failures;
}

// Build the id -> name maps from a program's HYP / EVENT decl spans.
export function nameMaps(X, prog) {
  const nameById = new Map();
  const evNameById = new Map();
  for (const d of prog.decls) {
    if (d.op === 'HYP' && d.span) {
      const nm = hypNameFromSpan(X, d.span);
      if (nm) nameById.set(d.id, nm);
    }
    if (d.op === 'EVENT' && d.span) {
      const t = X.slice(d.span.s, d.span.e).trim();
      if (t) evNameById.set(d.id, t);
    }
  }
  return { nameById, evNameById };
}

// ---------------------------------------------------------------------------
// ctxWitness: distinct-population source grounding.
// Sentences that name a population ("In South, ...", "figures come from
// North", "a South intake") label their premises with that population. A
// needed premise whose sentence is labeled ONLY with populations absent
// from the question's own labels cannot serve the query — refusing as
// CONTEXT_MISMATCH even when the extractor collapsed the CtxIds.
// Population labels come from the shared kernel-side ctxLabels:
// capitalized names AND lowercase "<modifier> <population-noun>" phrases
// ("day shift" and "night shift" name different populations even though
// neither word is capitalized).
// Returns failures [{code:'CONTEXT_MISMATCH', detail, span}].
// ---------------------------------------------------------------------------
export function ctxWitnessCheck({ X, decision, excludeLc }) {
  if (!decision || decision.status !== 'ANSWER') return [];
  const qPos = X.lastIndexOf('?');
  if (qPos <= 0) return [];
  const qb = sentenceBounds(X, qPos - 1);
  const { affirmed: qLabels, excluded: qNeg } = contextLabelPolarity(X.slice(qb.s, qb.e), excludeLc);
  // An anaphoric question ("this item", "it") carries no population label of
  // its own — its population is the observed item's. Fall back to the OBS
  // witness sentences before deciding the question is unlabeled.
  const useObservedLabels = qLabels.size === 0;
  for (const d of decision.witnesses ?? []) {
    if (d.op !== 'OBS' || !d.span) continue;
    const sb = sentenceBounds(X, d.span.s);
    const { affirmed, excluded } = contextLabelPolarity(X.slice(sb.s, sb.e), excludeLc);
    if (useObservedLabels) for (const label of affirmed) qLabels.add(label);
    for (const label of excluded) qNeg.add(label);
  }
  for (const label of qNeg) qLabels.delete(label);
  const failures = [];
  for (const d of decision.witnesses ?? []) {
    if (!POPULATION_OPS.has(d.op) || !d.span) continue;
    const lits = literalsInSpan(X, d.span);
    const site =
      lits.find((l) => d.value && R.eq(l.value, d.value)) ??
      lits[0] ??
      { s: d.span.s, e: d.span.e };
    const sb = sentenceBounds(X, site.s);
    const pLabels = ctxLabels(X.slice(sb.s, sb.e), excludeLc);
    if (pLabels.size === 0) continue;
    const denied = [...pLabels].some((l) => qNeg.has(l));
    const disjoint = qLabels.size > 0
      ? [...pLabels].every((l) => !qLabels.has(l))
      : RESTRICTED_OPS.has(d.op);
    if (denied || disjoint) {
      failures.push({
        code: 'CONTEXT_MISMATCH',
        detail: denied
          ? `ctx witness: ${d.op} premise at [${site.s},${site.e}] belongs to ` +
            `population {${[...pLabels].join(',')}} — affirmatively denied by the query/observation`
          : `ctx witness: ${d.op} premise at [${site.s},${site.e}] belongs to ` +
            `population {${[...pLabels].join(',')}} — ` +
            (qLabels.size > 0 ? `outside the query's {${[...qLabels].join(',')}}` :
              'the query/observation carries no population label'),
        span: d.span,
      });
    }
  }
  return failures;
}
