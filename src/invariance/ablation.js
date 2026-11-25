// invariance-gate :: Slot-3 clause-isolation context ablation.
// The ASSERTED head scores whether a span's content is genuinely asserted
// given its full context vs. read in isolation. In production this head is
// neural; the gate ships a deterministic reference scorer for the Phase-0
// spike. A span suppressed by surrounding context ("the obsolete manual
// states...", "preliminary figures", retracted quotes) is refused as
// SCOPED_OR_RETRACTED_PREMISE.

const SCOPE_MARKERS = [
  /\bobsolete\b/i,
  /\boutdated\b/i,
  /\bsuperseded\b/i,
  /\bretracted\b/i,
  /\bpreliminary\b/i,
  /\bdiscarded\b/i,
  /\bwithdrawn\b/i,
  /\bhypothetic\w*\b/i,
  /\bif we assume\b/i,
  /\bfor (?:the sake of )?argument\b/i,
  /\bsuppose(?:dly)?\b/i,
  /\balleged\w*\b/i,
  /\brumou?red\b/i,
  /\bunverified\b/i,
  /\bclaimed\b/i,
  /\baccording to the (?:old|retracted|obsolete|outdated|discredited)\b/i,
  /\bignore (?:the|those|any)\b/i,
  /\bno longer (?:valid|applies|holds)\b/i,
  /\brough\b/i,
  /\broughly\b/i, // hedged approximation
  /\babout\b/i, // hedged approximation — only when directly modifying a figure
  /\bapproximat\w*\b/i,
  /\bballpark\b/i,
  /\btentative\b/i,
  /\bquoted\b/i,
  /\bwas later (?:corrected|revised|retracted)\b/i,
];

// Backward-governing suppression markers: these appear in the sentence
// FOLLOWING a premise and retract/suppress what was just stated.
// ("The counts ... were struck from the record", "they were a drill".)
const BACKWARD_MARKERS = [
  /\bstruck from the record\b/i,
  /\bstricken\b/i,
  /\brescinded\b/i,
  /\bannulled\b/i,
  /\bvoided\b/i,
  /\bredacted\b/i,
  /\bretracted\b/i,
  /\bwithdrawn\b/i,
  /\brecalled\b/i,
  /\bnever (?:valid|counted|real|official)\b/i,
  /\bno longer (?:valid|applies|holds|counts|operative)\b/i,
  /\b(?:was|were|is|are) (?:only |just |merely )?(?:preliminary|a drill|a dry run|a practice|a simulation|a tabletop exercise|hypothetical|illustrative|an? exercise|for (?:the sake of )?argument|made up|fabricated)\b/i,
  /\btabletop exercise\b/i,
  /\bdry run\b/i,
  /\bpractice run\b/i,
  /\bnot (?:to be|meant to be) (?:used|trusted|relied on)\b/i,
  /\bdisregard\b/i,
  /\bignore (?:the|those|any|them)\b/i,
  /\bunverified\b/i,
  // structural back-references: a following sentence that points BACK at
  // the premise just stated and invalidates, mutates or supersedes it
  /\bno longer\b/i,
  /\b(?:anymore|any longer)\b/i,
  /^\s*(?:(?:however|later|in fact|subsequently|afterwards?|meanwhile|then|now|eventually)\s*,?\s+)?(?:that|this|it|they|both|neither)\b[^.!?]{0,60}?\b(?:wrong|false|inaccurate|mistaken|incorrect|erroneous|outdated|obsolete|debunked|disproved|disproven|refuted|overturned|denied|contested|disputed|questionable|invalid|unreliable|superseded|removed|rewired|disconnected|replaced|otherwise)\b/i,
  /\b(?:claims?|statements?|reports?|findings?|assumptions?|premises?|assertions?|conclusions?|results?|memos?|rules?|guidance|guidelines?|manuals?|information|figures?|relays?|switch(?:es)?|wires?|wiring|circuits?|links?|linkage|couplings?|connections?|setups?|arrangements?)\b[^.!?]{0,60}?\b(?:wrong|false|inaccurate|mistaken|incorrect|erroneous|outdated|obsolete|debunked|disproved|disproven|refuted|overturned|denied|contested|disputed|questionable|invalid|unreliable|superseded|removed|rewired|disconnected|replaced|reconfigured)\b/i,
  /\b(?:removed|rewired|disconnected|replaced|reconfigured|rerouted|detached|unplugged)\b[^.!?]{0,40}?\b(?:relays?|switch(?:es)?|wires?|wiring|circuits?|links?|linkage|couplings?|connections?|setups?|arrangements?)\b/i,
  /\b(?:doubts?|doubted|den(?:y|ies|ied)|disprov\w+|refut\w+|contest\w*|disput\w*|questioned|challeng\w+|rejected|dismiss\w*)\s+(?:\w+ly\s+)?(?:it|this|that|them)\b/i,
  /\b(?:called|brought|thrown|cast)\s+into\s+question\b/i,
  /\b(?:found|showed|shown|said|says|proved|concluded)\s+otherwise\b/i,
  /\bturned out\b[^.!?]{0,40}?\b(?:wrong|false|mistaken|incorrect|otherwise|not)\b/i,
  /\b(?:debunked|disproved|disproven|refuted|overturned|superseded|outdated|obsolete|inaccurate|contested|disputed|rewired|disconnected)\b/i,
  /\b(?:was|were|is|are)\s+(?:an?\s+)?(?:error|mistake)\b/i,
  /\b(?:held|holds|valid|applied|true)\s+only\s+until\b/i,
  /\b(?:treat|read|use|view|regard)\b[^.!?]{0,50}?\bwith\s+caution\b/i,
  /\bbeliev\w+\b[^.!?]{0,40}?\b(?:anymore|any longer|no longer)\b/i,
  // editorial / preceding-line suppression (Bulma R3 F2): a following
  // sentence that strikes, disregards or re-sources the line just stated,
  // or verblessly scopes it to a prototype/demonstration, governs backward
  /^\s*(?:strike|delete|remove|disregard|ignore|omit|drop|scratch|expunge)\b[^.!?]{0,50}?\b(?:preceding|previous|prior|above|earlier|last|sentence|line|statement|logbook|record)\b/i,
  /\b(?:preceding|previous)\s+(?:sentence|line|statement|claim)\b/i,
  /\b(?:sentence|line)\s+(?:immediately\s+)?(?:above|below)\b/i,
  /\b(?:copied|taken|lifted|transcribed|reproduced)\b[^.!?]{0,60}?\b(?:fiction|fictional|novel|story|textbook)\b/i,
  /\bwork of fiction\b/i,
  /^\s*only\s+(?:in|on|during|for|under|at)\b/i,
  /\b(?:only\s+(?:the\s+|in\s+the\s+)?(?:\w+\s+){0,2}?(?:prototype|demonstration|exhibition|classroom|mockup|showroom)|(?:limits?|limited|restrict\w*|confined)\b[^.!?]{0,50}?\b(?:prototype|demonstration|exhibition|exhibit|classroom|decommissioned|mockup|showroom))\b/i,
  /\b(?:exhibit|label|placard|catalog(?:ue)?|illustration)\b[^.!?]{0,60}?\b(?:limits?|limited|restrict\w*|only|decommissioned)\b/i,
];

// Conservative: "about"/"roughly" only count when immediately preceding a
// numeric literal inside the same clause — handled by proximity check below.

// A '.' between digits ("68.75%", "0.625") is a decimal point, not a
// sentence terminator.
const isTerminator = (text, i) =>
  text[i] === '!' ||
  text[i] === '?' ||
  text[i] === '\n' ||
  (text[i] === '.' && !(i > 0 && i + 1 < text.length && /\d/.test(text[i - 1]) && /\d/.test(text[i + 1])));

export function sentenceBounds(text, pos) {
  let s = 0;
  let e = text.length;
  for (let i = pos; i >= 0; i--) {
    if (isTerminator(text, i)) {
      s = i + 1;
      break;
    }
  }
  for (let i = pos; i < text.length; i++) {
    if (isTerminator(text, i)) {
      e = i + 1;
      break;
    }
  }
  return { s, e };
}

// The sentence immediately preceding offset `s`, skipping whitespace and the
// trailing terminator. Returns null when `s` starts the document or a fresh
// paragraph (a blank line breaks government: the '\n' boundary makes the
// "previous sentence" an empty segment).
function prevSentenceBounds(context, s) {
  let p = s - 1;
  while (p >= 0 && /\s/.test(context[p])) p--;
  if (p < 0) return null;
  const end = p + 1;
  let start = 0;
  for (let i = p - 1; i >= 0; i--) {
    if ('.!?\n'.includes(context[i]) && context[i] !== '.') {
      start = i + 1;
      break;
    }
    if (context[i] === '.' && !(i > 0 && i + 1 < context.length && /\d/.test(context[i - 1]) && /\d/.test(context[i + 1]))) {
      start = i + 1;
      break;
    }
  }
  const sent = context.slice(start, end);
  if (!sent.trim()) return null;
  return { s: start, e: end };
}

// Does the sentence containing `span` carry a scoping marker — or does the
// immediately preceding sentence carry one? The second case covers governing
// prefixes ("The obsolete 1998 manual, since retracted, states the following
// base rates." → the figures sentence is scoped by the prefix).
// Does the sentence containing `span` carry a scoping marker — does the
// immediately preceding sentence govern it — does a FOLLOWING sentence
// retract it — or does the span sit inside a verbatim quotation?
export function spanScoped(context, span) {
  const { s, e } = sentenceBounds(context, span.s);
  const sentEnd = Math.max(e, span.e);
  const sentence = context.slice(s, sentEnd);
  const hits = [];
  const scan = (sent, base, checkProximity) => {
    for (const re of SCOPE_MARKERS) {
      const m = re.exec(sent);
      if (m) {
        // "about"/"roughly" must sit within ~24 chars before the span to count
        if (checkProximity && (String(re).includes('about') || String(re).includes('roughly'))) {
          const absPos = base + m.index;
          if (span.s - (absPos + m[0].length) > 24 || absPos > span.e) continue;
        }
        hits.push({ marker: m[0], offset: base + m.index });
      }
    }
  };
  scan(sentence, s, true);
  // verbatim-quoted figures are reported speech, not assertions
  if (hits.length === 0) {
    const lq = sentence.lastIndexOf('"', span.s - s);
    const rq = sentence.indexOf('"', span.e - s);
    const lqc = sentence.lastIndexOf('\u201C', span.s - s);
    const rqc = sentence.indexOf('\u201D', span.e - s);
    if ((lq >= 0 && rq >= 0) || (lqc >= 0 && rqc >= 0)) {
      hits.push({ marker: 'quoted-string', offset: span.s });
    }
  }
  if (hits.length === 0) {
    const prev = prevSentenceBounds(context, s);
    if (prev) scan(context.slice(prev.s, prev.e), prev.s, false);
  }
  // backward-governing suppression: a following sentence that retracts what
  // was just stated ("were struck from the record", "that was a drill").
  if (hits.length === 0) {
    const next = nextSentenceBounds(context, sentEnd);
    if (next) {
      const nsent = context.slice(next.s, next.e);
      for (const re of BACKWARD_MARKERS) {
        const m = re.exec(nsent);
        if (m) hits.push({ marker: `post:${m[0]}`, offset: next.s + m.index });
      }
    }
  }
  return hits.length > 0 ? { scoped: true, sentence: [s, sentEnd], hits } : { scoped: false };
}

// The sentence immediately following offset `e`, skipping whitespace.
function nextSentenceBounds(context, e) {
  let p = e;
  while (p < context.length && /\s/.test(context[p])) p++;
  if (p >= context.length) return null;
  const end = sentenceBounds(context, p).e;
  return { s: p, e: end };
}

/**
 * Deterministic reference ASSERTED head.
 *   assertedHead(excerpt, context) -> [0,1]
 * `context` may be the full source X (assertion judged in-context) or the
 * excerpt itself (isolation run). Returns high when asserted, low when the
 * governing sentence carries a scoping marker.
 */
export function heuristicAssertedHead(excerpt, context) {
  const isolated = context === excerpt;
  if (isolated) {
    // an excerpt that flags itself ("this retracted figure...") still counts
    const idx = 0;
    const probe = spanScoped(excerpt, { s: idx, e: Math.max(1, excerpt.length) });
    return probe.scoped ? 0.35 : 0.95;
  }
  // locate the excerpt in context; evaluate every occurrence's sentence.
  const occurrences = [];
  let from = 0;
  while (true) {
    const i = context.indexOf(excerpt, from);
    if (i === -1) break;
    occurrences.push(i);
    from = i + 1;
    if (occurrences.length > 64) break;
  }
  if (occurrences.length === 0) return 0.5;
  // take the occurrence with the highest assertion score — the extractor may
  // bind any of them; suppression requires ALL occurrences suppressed? No:
  // conservative gate — if any occurrence is cleanly asserted the premise is
  // defensible; if all occurrences are scoped, the premise is suppressed.
  let best = 0;
  let anyScoped = false;
  for (const i of occurrences) {
    const probe = spanScoped(context, { s: i, e: i + excerpt.length });
    if (probe.scoped) anyScoped = true;
    best = Math.max(best, probe.scoped ? 0.1 : 0.95);
  }
  return anyScoped && best <= 0.1 ? 0.1 : best;
}

/**
 * ablationCheck({X, spans, assertedHead, delta})
 * spans: premise spans whose assertion is load-bearing.
 * Returns {ok:true} or {ok:false, span, suppression, reason}.
 */
export function ablationCheck({ X, spans, assertedHead = heuristicAssertedHead, delta = 0.4 }) {
  for (const span of spans) {
    if (!span) continue;
    const excerpt = X.slice(span.s, span.e);
    const aIso = assertedHead(excerpt, excerpt);
    const aFull = assertedHead(excerpt, X);
    const suppression = aIso - aFull;
    if (suppression > delta) {
      return { ok: false, span, suppression, aIso, aFull, excerpt };
    }
  }
  return { ok: true };
}
