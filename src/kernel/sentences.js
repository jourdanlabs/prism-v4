// prism-kernel :: exhaustive sentence partitioning (PRISM V2 ARCH-SPEC §3).
// Every input passage X segments into indexed sentences S_0..S_{m-1}; a
// Program's SENT_TAG decls must account for 100% of indices, and every
// material sentence must be grounded by a premise decl. A material sentence
// the program silently drops — e.g. "the lamp and buzzer are wired to the
// same switch" mentioning two declared events — is dropped evidence and
// refuses as UNACCOUNTED_MATERIAL_SENTENCE. This check is text-layer but
// model-free: the gate and kernel both run it, and neither trusts the
// extractor's tags.

import { scanLiterals } from './literals.js';
import * as R from './rational.js';

export const SENT_TAGS = new Set([
  'PREMISE',
  'QUERY',
  'CONTEXT_SCOPE',
  'NON_ASSERTED',
  'FLUFF',
]);

// Segment containing offset `pos` (decl span start), or null outside text.
export function sentenceAt(X, pos) {
  return sentences(X).find((seg) => pos >= seg.s && pos < seg.e) ?? null;
}

// A '.' between digits ("68.75%", "0.625") is a decimal point, not a
// sentence terminator. '\n', '!' and '?' always terminate.
const isTerminator = (text, i) =>
  text[i] === '!' ||
  text[i] === '?' ||
  text[i] === '\n' ||
  (text[i] === '.' && !(i > 0 && i + 1 < text.length && /\d/.test(text[i - 1]) && /\d/.test(text[i + 1])));

// Segment X into sentences [{idx, s, e, text}]. Trailing whitespace after the
// last terminator attaches to the last segment; an empty X yields [].
export function sentences(X) {
  const out = [];
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    if (isTerminator(X, i)) {
      out.push({ idx: out.length, s, e: i + 1, text: X.slice(s, i + 1) });
      s = i + 1;
    }
  }
  if (s < X.length) {
    out.push({ idx: out.length, s, e: X.length, text: X.slice(s) });
  }
  // blank segments (empty lines) are not sentences — drop them and re-index
  // so idx is always the contiguous position 0..m-1 the ledger checker uses
  return out
    .filter((seg) => seg.text.trim().length > 0 || out.length === 1)
    .map((seg, idx) => ({ ...seg, idx }));
}

// Hypothesis name from a HYP decl span (mirrors ownership.js
// hypNameFromSpan — kept kernel-local to avoid dep direction). A span
// narrowed to one clean alphabetic word IS the name in any case
// ("copper", "tin", "Alpha"); a wider span yields its first capitalized
// token. English function words are never names.
const HYP_NAME_STOP = new Set([
  'the', 'all', 'are', 'or', 'of', 'and', 'every', 'each', 'a', 'an',
  'is', 'was', 'were',
]);
function nameFromSpan(X, span) {
  const t = X.slice(span.s, span.e);
  const word = t.trim();
  if (/^[A-Za-z]+(?:-[A-Za-z]+)*$/.test(word) &&
      word.split('-').every((p) => !HYP_NAME_STOP.has(p.toLowerCase()))) {
    return word;
  }
  const m = t.match(/\b([A-Z][a-zA-Z]+)\b/);
  return m && !HYP_NAME_STOP.has(m[1].toLowerCase()) ? m[1] : null;
}

// Declared entity names: HYP names + EVENT span text (trimmed).
export function entityNames(X, prog) {
  const names = [];
  for (const d of prog.decls) {
    if (d.op === 'HYP' && d.span) {
      const n = nameFromSpan(X, d.span);
      if (n) names.push(n);
    }
    if (d.op === 'EVENT' && d.span) {
      const t = X.slice(d.span.s, d.span.e).trim();
      if (t.length >= 2) names.push(t);
    }
  }
  return names;
}

const wordRe = (name) => {
  const esc = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`\\b${esc}\\b`, 'i');
};

// Conservative English singular/plural inflections of a head noun: the
// regular s / es / ies<->y alternations only. "box<->boxes",
// "tally<->tallies", "notch<->notches", "panel<->panels".
function headVariants(head) {
  const out = new Set([head]);
  const lc = head.toLowerCase();
  if (lc.length < 2) return [...out];
  if (/[^aeiou]y$/.test(lc)) out.add(`${head.slice(0, -1)}ies`);
  if (/ies$/.test(lc)) out.add(`${head.slice(0, -3)}y`);
  if (/(?:s|x|z|ch|sh)$/.test(lc)) out.add(`${head}es`);
  if (/(?:ses|xes|zes|ches|shes)$/.test(lc)) out.add(head.slice(0, -2));
  if (!/s$/.test(lc)) out.add(`${head}s`);
  if (/[^s]s$/.test(lc) && !/ies$/.test(lc) && !/(?:ses|xes|zes|ches|shes)$/.test(lc)) {
    out.add(head.slice(0, -1));
  }
  return [...out];
}

// Conservative leading-verb inflections of a simple alphabetic verb:
// -ing / -s / -es / -ed, silent-e drop ("fire" -> "firing", "fires",
// "fired"), final-consonant doubling on a short stressed CVC ("trip" ->
// "tripping"), -y -> -ies, and the one common irregular past "caught".
const IRREGULAR_PAST = new Map([['catch', 'caught']]);
function verbVariants(verb) {
  const out = new Set([verb]);
  const lc = verb.toLowerCase();
  if (lc.length < 2 || !/^[a-z]+$/.test(lc)) return [...out];
  const irregular = IRREGULAR_PAST.get(lc);
  if (irregular) out.add(irregular);
  if (/[^aeiou]e$/.test(lc)) {
    out.add(`${verb.slice(0, -1)}ing`);
    out.add(`${verb}s`);
    out.add(`${verb}d`);
  } else if (/[^aeiou]y$/.test(lc)) {
    out.add(`${verb}ing`);
    out.add(`${verb.slice(0, -1)}ies`);
    out.add(`${verb.slice(0, -1)}ied`);
  } else if (/^[bcdfghjklmnpqrstvwxz]+[aeiou][bcdfgklmnprstz]$/.test(lc) && lc.length <= 5) {
    const dbl = verb + verb.slice(-1);
    out.add(`${dbl}ing`);
    out.add(`${verb}s`);
    out.add(`${dbl}ed`);
  } else {
    out.add(`${verb}ing`);
    out.add(`${verb}s`);
    if (/(?:s|x|z|ch|sh)$/.test(lc)) out.add(`${verb}es`);
    out.add(`${verb}ed`);
  }
  return [...out];
}

const VERB_PHRASE_LINK = new Set([
  'a', 'an', 'the', 'at', 'on', 'in', 'of', 'for', 'to', 'by', 'with',
  'from', 'over', 'under', 'up', 'out', 'off',
  'before', 'after', 'past', 'through', 'into', 'onto', 'across',
  'around', 'against', 'below', 'above', 'near', 'during', 'between',
  'beyond', 'within', 'behind', 'beside', 'until', 'without',
]);

// Surface forms of a (possibly multi-word) event name under conservative
// inflection of its head noun — the final alphabetic token (after the last
// hyphen of the final word). "amber notch" -> also "amber notches". A
// multi-word VERB phrase ("catch at the north stop" — a leading alphabetic
// verb followed by an article/preposition-linked complement) ALSO inflects
// its leading verb, so "catching / catches / caught at the north stop"
// match the declared base form.
export function inflectedNameVariants(name) {
  if (typeof name !== 'string' || !name.trim()) return [];
  const t = name.trim();
  const cut = Math.max(t.lastIndexOf(' '), t.lastIndexOf('-'));
  const prefix = cut >= 0 ? t.slice(0, cut + 1) : '';
  const head = t.slice(cut + 1);
  if (!/^[A-Za-z]+$/.test(head)) return [name];
  const nouns = headVariants(head).map((h) => prefix + h);
  const words = t.split(/\s+/);
  if (words.length < 2 || !/^[A-Za-z]{2,}$/.test(words[0])) return nouns;
  if (!words.slice(1).some((w) => VERB_PHRASE_LINK.has(w.toLowerCase()))) return nouns;
  const rest = t.slice(words[0].length);
  const out = new Set(nouns);
  for (const v of verbVariants(words[0])) {
    if (v === words[0]) continue;
    out.add(v + rest);
    // leading-verb inflection combined with head-noun plurality
    const infl = v + rest;
    const cut2 = Math.max(infl.lastIndexOf(' '), infl.lastIndexOf('-'));
    for (const h of headVariants(head)) out.add(infl.slice(0, cut2 + 1) + h);
  }
  return [...out];
}

// Whole-token regex over every conservative inflection of `name`. Strict
// \b edges: an inflection never matches inside a longer word. Words of a
// multi-word name may be joined by spaces or hyphens interchangeably, so
// "amber tag" matches "amber-tag" and "amber tags" matches "amber-tags".
export function inflectedNameRe(name, flags = 'i') {
  const alts = inflectedNameVariants(name)
    .slice()
    .sort((a, b) => b.length - a.length)
    .map((v) => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/[\s-]+/g, '[\\s-]+'));
  return new RegExp(`\\b(${alts.join('|')})\\b`, flags);
}

// Dependency / linkage / scope vocabulary: a sentence that mentions a
// declared entity under one of these is asserting structure the program must
// ledger (DET_LINK, COND_INDEP, JOINT_LIK, CONTEXT_SCOPE) — never FLUFF.
export const DEPENDENCY_WORDS =
  /\bwired\b|\bhard-?wired\b|\bswitch\b|\blinked\b|\blinkage\b|\bcoupled\b|\bganged\b|\bslaved\b|\blockstep\b|\bin tandem\b|\bin unison\b|\btogether\b|\bsimultaneous\w*\b|\bco-?occur\w*\b|\bindependen\w*\b|\bdependen\w*\b|\bcorrelat\w*\b|\bconfound\w*\b|\bcaus\w+\b|\btrigger\w*\b|\bdriv\w+\b|\bactivate\w*\b|\bsame\b|\bmutex\b|\bexclusiv\w*\b|\bdisjoint\b|\bwhenever\b|\bimplies\b|\balways\b|\bnever\b|\brelays?\b|\bcircuits?\b|\byok\w+\b|\bdaisy-?chain\w*\b|\bshar\w+\s+a\b|\bpreclud\w+\b|\bincompatible\b/i;

// Plural anaphoric co-reference: "the two outputs share a relay", "these
// outputs activate together", "both fire" — the named entities lived in
// earlier sentences; this sentence still asserts a relation among them.
export const ANAPHORA =
  /\b(?:the\s+two|these|those|both|the\s+pair|the\s+duo|they|the\s+two\s+of\s+them)\b/i;

// Is a probability-like literal present? Bare integer runs are counts/labels
// and exempt (same rule as UNBOUND_LITERAL coverage).
const probLikeLiteral = (text) => scanLiterals(text).some((l) => !/^\d+$/.test(l.text));

// Query vocabulary: marks the ASK_PROB sentence even when phrased as an
// imperative ("Estimate the probability of Alpha given the reading.") rather
// than with a question mark. A sentence bearing a numeric literal is checked
// for literals first, so premise sentences can never be classified here.
const QUESTION_VOCAB =
  /\b(what|how likely|estimate|probability|probable|odds|chance|likelihood)\b/i;

const VERIFIED = '(?:check(?:s|ed|ing)?|verif(?:y|ies|ied|ying)|confirm(?:s|ed|ing)?|validat(?:e|es|ed|ing)|test(?:s|ed|ing)?|establish(?:es|ed|ing)?|review(?:s|ed|ing)?|substantiat(?:e|es|ed|ing)|proven|corroborated|authenticated)(?![\\w-])';
const UNVERIFIED = '(?:unverified|unconfirmed|unchecked|untested|unvalidated|unsubstantiated|uncertain|unproven|disputed|in dispute)';
const DOUBT_MARKERS = [
  new RegExp(`\\b(?:nobody|no[- ]one|no person|none of (?:us|them|the [\\w-]+))\\b[^,;.!?]*?\\b${VERIFIED}`, 'i'),
  new RegExp(`\\b(?:not|never|\\w+n['’]t)\\s+(?:(?:yet|ever|actually|officially|independently|personally|fully|formally|been|be)\\s+){0,4}${VERIFIED}`, 'i'),
  /\b(?:no|without)\s+(?:independent\s+|on-site\s+)?(?:checks?|verification|confirmation|validation|testing|corroboration|evidence (?:that|for))\b/i,
  /\b(?:verification|confirmation|validation)\s+(?:is|remains)\s+(?:still\s+)?pending\b/i,
  new RegExp(`(?:^\\s*|\\b(?:is|are|was|were|been|remains?|stays?|still|as yet)\\s+)(?:(?:still|currently|entirely|as yet|largely)\\s+)*${UNVERIFIED}\\b`, 'i'),
  new RegExp(`\\b(?:has|have|is|are)\\s+yet to be\\s+${VERIFIED}`, 'i'),
  /\b(?:is|are|was|were|been|got)\s+(?:(?:later|since|subsequently|formally|officially)\s+)*(?:retracted|withdrawn|rescinded|voided|disavowed|stricken|struck from the record)\b/i,
  /\b(?:was|were)\s+(?:later|subsequently)\s+corrected\b/i,
  /\b(?:retract|withdraw|rescind|disavow)\w*\s+(?:(?:a|that|the|this|its|their|our)\s+)?(?:claim|finding|statement|report|conclusion|assertion|result)s?\b/i,
  /\b(?:no longer (?:applies|holds|valid|operative)|(?:casts?|raises?) doubt on|(?:that|this|the) (?:claim|finding|statement|report|conclusion|assertion|result) is questionable)\b/i,
];

// --- structural discourse back-reference / retraction guard (R3 F2) -------
// A closed phrase list overfits the wording it was written against. These
// patterns key on STRUCTURE instead: a bare propositional/anaphoric subject
// ("That ...", "This ...", "It ...") or a prior-discourse / mechanism noun
// combined with an invalidation, mutation, denial or supersession
// predicate. The subject must be the pronoun itself (immediately followed
// by verb-class material within the predicate window) so adjectival uses
// ("This cabinet was inspected ...", "That information ..." is caught by
// the discourse-noun arm instead) keep their concrete reading, and an
// expletive outer frame ("It is possible/believed/likely that ...") is
// explicitly excluded — those sentences fail as unlicensed licenses, not
// as retracted premises.
const DISCOURSE_NOUN =
  '(?:claims?|statements?|reports?|findings?|assumptions?|premises?|assertions?|conclusions?|results?|memos?|rules?|guidance|guidelines?|manuals?|information|figures?|data)';
const MECHANISM_NOUN =
  '(?:relays?|switch(?:es)?|wires?|wiring|circuits?|links?|linkage|couplings?|connections?|setups?|arrangements?)';
const INVALID_PRED =
  '(?:wrong|false|inaccurate|mistaken|incorrect|erroneous|outdated|obsolete|debunked|disproved|disproven|refuted|overturned|denied|contested|disputed|questionable|invalid|unreliable|superseded|retracted|withdrawn|rescinded|voided|an error|a mistake|in error|otherwise)';
const MUTATION_PRED =
  '(?:removed|rewired|disconnected|replaced|reconfigured|rerouted|detached|unplugged|scrapped|dismantled|taken out|swapped out|stripped out|ripped out|torn out)';
const EPISTEMIC_VERB =
  '(?:doubts?|doubted|doubting|den(?:y|ies|ied)|disprov\\w+|refut\\w+|contest\\w*|disput\\w*|questions?|questioned|challeng\\w+|rejects?|rejected|dismiss\\w*)';
const BARE_SUBJ = '(?:that|this|it|they|both|neither)';
const SENT_LEAD =
  "^\\s*[\"'“”‘’]?\\s*(?:(?:however|later|in fact|subsequently|afterwards?|meanwhile|then|now|eventually|in the end|as it turned out|on inspection|on review|in retrospect)\\s*,?\\s+)?";
const EXPLETIVE_THAT =
  /^\s*["'“”‘’]?\s*it\s+(?:is|was|seems|appears|can(?:not|'t)\s+be|cannot\s+be)\s+(?:not\s+)?(?:possible|believed|likely|unlikely|rumou?red|thought|known|said|reported|certain|true|ruled\s+out)\b[^.!?;]{0,24}?\bthat\b/i;

const STRUCTURAL_DOUBT = [
  // bare propositional subject + invalidation / supersession predicate
  new RegExp(`${SENT_LEAD}${BARE_SUBJ}\\b[^.!?;]{0,60}?\\b${INVALID_PRED}\\b`, 'i'),
  new RegExp(`${SENT_LEAD}${BARE_SUBJ}\\b[^.!?;]{0,50}?\\b(?:no longer|anymore|any longer)\\b`, 'i'),
  new RegExp(`${SENT_LEAD}${BARE_SUBJ}\\b[^.!?;]{0,50}?\\b${MUTATION_PRED}\\b`, 'i'),
  // prior-discourse / mechanism noun + invalidation or mutation (either order)
  new RegExp(`\\b(?:${DISCOURSE_NOUN}|${MECHANISM_NOUN})\\b[^.!?;]{0,60}?\\b${INVALID_PRED}\\b`, 'i'),
  new RegExp(`\\b${INVALID_PRED}\\b[^.!?;]{0,40}?\\b(?:in|of|by|from)\\s+(?:the\\s+)?(?:${DISCOURSE_NOUN}|${MECHANISM_NOUN})\\b`, 'i'),
  new RegExp(`\\b${MECHANISM_NOUN}\\b[^.!?;]{0,50}?\\b${MUTATION_PRED}\\b`, 'i'),
  new RegExp(`\\b${MUTATION_PRED}\\b[^.!?;]{0,40}?\\b${MECHANISM_NOUN}\\b`, 'i'),
  // epistemic / mutation verb over an anaphoric or discourse object
  new RegExp(`\\b${EPISTEMIC_VERB}\\s+(?:\\w+ly\\s+)?(?:it|this|that|them|the\\s+${DISCOURSE_NOUN}|that\\s+${DISCOURSE_NOUN}|this\\s+${DISCOURSE_NOUN})\\b`, 'i'),
  new RegExp(`\\b${MUTATION_PRED}\\s+(?:(?:\\w+)\\s+){0,3}?(?:them|it|this|that)\\b`, 'i'),
  /\b(?:called|brought|thrown|cast)\s+into\s+question\b/i,
  /\b(?:found|showed|shown|said|says|proved|concluded)\s+otherwise\b/i,
  // belief negation / belief supersession
  /\b(?:nobody|no[- ]one|no person|none of (?:us|them|the [\w-]+))\b[^.!?;]{0,40}?\bbeliev\w+\b/i,
  /\bbeliev\w+\b[^.!?;]{0,40}?\b(?:anymore|any longer|no longer)\b/i,
  // temporal supersession of a prior arrangement
  new RegExp(`\\b(?:has|have|had)\\s+since\\s+(?:been\\s+)?(?:${MUTATION_PRED}|${INVALID_PRED})\\b`, 'i'),
  /\b(?:held|holds|valid|applied|true)\s+only\s+until\b/i,
  /\bcontrary\s+to\s+(?:the\s+)?(?:earlier|previous|original|stated|reported)\b/i,
  // explicit caution over a prior claim
  /\b(?:treat|read|use|view|regard)\b[^.!?;]{0,50}?\bwith\s+caution\b/i,
  // editorial action over the preceding text (Bulma R3 F2): an instruction
  // or report that strikes, deletes, disregards or re-sources the sentence
  // just stated withdraws that premise as surely as a retraction verb does
  /^\s*["'“”‘’]?\s*(?:strike|delete|remove|disregard|ignore|omit|drop|scratch|expunge)\b[^.!?;]{0,50}?\b(?:preceding|previous|prior|above|earlier|last|sentence|line|statement|logbook|record)\b/i,
  /\b(?:preceding|previous)\s+(?:sentence|line|statement|claim)\b/i,
  /\b(?:sentence|line)\s+(?:immediately\s+)?(?:above|below)\b/i,
  /\b(?:disregard|ignore|strike|delete|omit)\b[^.!?;]{0,40}?\b(?:above|preceding|previous|sentence|line)\b/i,
  /\b(?:copied|taken|lifted|transcribed|reproduced)\b[^.!?;]{0,60}?\b(?:fiction|fictional|novel|story|textbook)\b/i,
  /\bwork of fiction\b/i,
  // scope-limiting metalanguage: the prior description is confined to a
  // prototype, demonstration, exhibit or retired stock
  /\b(?:only\s+(?:the\s+|in\s+the\s+)?(?:\w+\s+){0,2}?(?:prototype|demonstration|exhibition|classroom|mockup|showroom)|(?:limits?|limited|restrict\w*|confined)\b[^.!?;]{0,50}?\b(?:prototype|demonstration|exhibition|exhibit|classroom|decommissioned|mockup|showroom))\b/i,
  /\b(?:exhibit|label|placard|catalog(?:ue)?|illustration)\b[^.!?;]{0,60}?\b(?:limits?|limited|restrict\w*|only|decommissioned)\b/i,
];

// A verbless scope fragment ("Only during the showroom trial.") still
// scopes the premise it follows. No finite verb and an "only <prep>" lead
// make it a restriction on the preceding claim, never scene filler.
const FINITE_VERB_TOK =
  /\b(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|should|shall|may|might|must|runs?|ran|stands?|stood|sits?|sat|hangs?|hung|lights?|lit|sounds?|sounded|rings?|rang|flashes?|flashed|fires?|fired|activates?|activated|occurs?|occurred|holds?|held|applies|applied|went|came|fell|kept|blew|shone)\b/i;
function scopeFragmentDoubt(text) {
  const m = text.match(/^\s*["'“”‘’]?\s*only\s+(?:in|on|during|for|under|at)\b[^.!?;]*/i);
  if (!m) return null;
  if (FINITE_VERB_TOK.test(m[0])) return null;
  return m[0].trim();
}

export function premiseDoubt(text) {
  for (const marker of DOUBT_MARKERS) {
    const match = marker.exec(text);
    if (match) return match[0];
  }
  if (EXPLETIVE_THAT.test(text)) return null;
  for (const marker of STRUCTURAL_DOUBT) {
    const match = marker.exec(text);
    if (match) return match[0];
  }
  return scopeFragmentDoubt(text);
}

// --- positive-evidence inert-scene whitelist (Bulma R3 F2) ----------------
// FLUFF is no longer the residue of a trigger blacklist. A name-free
// sentence is inert ONLY when it positively matches a compact structural
// grammar of static scene statements, in a single clause:
//   * static physical attribute   "The cabinets are painted grey."
//   * material / composition      "The counter is made of polished birch."
//   * object placement            "The cabinet stands beside a stool."
//   * possession of a fixture     "The binder had a blue cover."
//   * completed mundane handling  "The notes were transcribed after rounds."
//   * background organisation     "The line audit grouped units by origin."
//   * plain item count            "There are 12 spare nozzles in the shed."
// The subject must be a concrete determiner/count noun phrase (never a
// pronoun, an imperative, or a verbless fragment), the verb must fall in
// one of the inert classes above, and the tail must partition into noun
// phrases and prepositional phrases. Disqualifiers — temporal/frequency
// adverbs, editorial metalanguage, behaviour/activation/timer/output
// vocabulary, epistemic or evaluative predicates, negation, modals,
// mutation participles and anaphoric pronouns — void the evidence no
// matter the frame. Anything short of this proof is MATERIAL.
const INERT_FREQ_TOK = new Set(
  ('sometimes occasionally yesterday every always never alone apart frequently rarely seldom often daily weekly monthly nightly hourly yearly whenever occasional since later anymore subsequently afterwards until anymore').split(' ')
);
const INERT_EDITORIAL_TOK = new Set(
  ('sentence sentences preceding strike disregard copied fiction fictional novel textbook prototype prototypes demonstration demonstrations exhibit exhibits exhibition description descriptions stock only above editor logbook catalog catalogue illustration restricted limiting limits decommissioned mockup').split(' ')
);
const INERT_BEHAVIOR_TOK = new Set(
  ('behavior behaviour activation activations timer timers output outputs firing firings solo occurrence occurrences').split(' ')
);
const INERT_MODAL_NEG_TOK = new Set(
  ('may might could would should shall must can cannot will not never no actually').split(' ')
);
const INERT_MUTATION_TOK = new Set(
  ('removed rewired disconnected replaced reconfigured rerouted detached unplugged scrapped dismantled swapped').split(' ')
);
const INERT_BEHAVIOR_VERB_TOK = new Set(
  ('fires fired activates activated triggers triggered occurs occurred happening happens happened shows showed indicates indicated suggests suggested states stated describes described mirrors mirrored duplicates duplicated matches matched drives drove feeds fed powers powered').split(' ')
);
// Coupling vocabulary: a scene statement never needs to say two things
// act as a pair, jointly, separately-as-a-pair or through a shared feed.
const INERT_COUPLING_TOK = new Set(
  ('pair duo both together separate joint combined shared share shares sharing').split(' ')
);
const MECH_HEAD_TOK = new Set(
  ('relay relays switch switches wire wires wiring circuit circuits link links linkage coupling couplings connection connections controller controllers manifold').split(' ')
);
const TAIL_ADVERB_TOK = new Set(['separately', 'alphabetically', 'neatly', 'tidily']);
const WEEKDAY_TOK = new Set(
  ('monday tuesday wednesday thursday friday saturday sunday').split(' ')
);
const NUMWORD_TOK = new Set(
  ('one two three four five six seven eight nine ten eleven twelve').split(' ')
);
const COPULA_TOK = new Set(['is', 'are', 'was', 'were']);
const AUX_STOP_TOK = new Set(
  ('is are was were has have had be been being does do did').split(' ')
);
const PLACE_VERB_TOK = new Set(
  ('stands stood sits sat hangs hung lies lay rests rested').split(' ')
);
const IRREG_PAST_TOK = new Set(
  ('ran took split had made kept left sent built went came gave saw held').split(' ')
);
const IRREG_PART_TOK = new Set(['made', 'built', 'kept', 'left', 'sent', 'held']);
const PREP_TOK = new Set(
  ('in on at by for with across through into beside near before after over under from to of').split(' ')
);
const OBJ_DET_TOK = new Set(['the', 'a', 'an', 'each', 'its', 'all']);
const PRONOUN_TOK = new Set(
  ('it them they he she we you him her us this that these those').split(' ')
);
const HEAD_BAN_TOK = new Set(
  ('claim claims statement statements report reports finding findings assumption assumptions premise premises assertion assertions conclusion conclusions result results memo memos rule rules guidance guideline guidelines manual manuals information figure figures data behavior behaviour activation activations output outputs timer timers sentence sentences line lines description descriptions').split(' ')
);
const INERT_INVALID_RE = new RegExp(`\\b${INVALID_PRED}\\b`, 'i');
const INERT_EPISTEMIC_RE = new RegExp(`\\b(?:${EPISTEMIC_VERB}|contradict\\w*|conflict\\w*)\\b`, 'i');

const isCountTok = (t) => NUMWORD_TOK.has(t) || /^\d+$/.test(t);
const isPastTok = (t) => IRREG_PAST_TOK.has(t) || (t.length > 3 && t.endsWith('ed'));
const isPartTok = (t) => IRREG_PART_TOK.has(t) || (t.length > 3 && t.endsWith('ed'));
const isPres3Tok = (t) => t.length > 3 && t.endsWith('s') && !t.endsWith('ss');

// NP at toks[i]: optional determiner/count, then 1..4 nounish tokens.
// Returns every viable end index (longest first).
function npEnds(toks, i, { bareOk }) {
  const ends = [];
  let j = i;
  if (OBJ_DET_TOK.has(toks[j]) || isCountTok(toks[j])) j++;
  else if (!bareOk) return ends;
  let n = 0;
  while (j < toks.length && n < 4) {
    const t = toks[j];
    if (PREP_TOK.has(t) || AUX_STOP_TOK.has(t) || PRONOUN_TOK.has(t)) break;
    if (t.length > 3 && t.endsWith('ly')) break; // adverbs are not NP material
    if (!/^[a-z][a-z-]*$|^\d+$/.test(t)) break;
    j++;
    n++;
    ends.push(j);
  }
  return ends.reverse();
}

// Tail from toks[i]: PPs, one-word adverbs and a trailing "last year"-type
// adjunct, partitioning exactly to the end of the sentence.
function tailOk(toks, i) {
  if (i === toks.length) return true;
  if (i > toks.length) return false;
  const t = toks[i];
  if (t === 'twice' || t === 'once' || TAIL_ADVERB_TOK.has(t)) {
    if (tailOk(toks, i + 1)) return true;
  }
  if ((t === 'last' || t === 'next' || t === 'this') &&
      ['year', 'month', 'week'].includes(toks[i + 1]) && i + 2 === toks.length) {
    return true;
  }
  if (PREP_TOK.has(t)) {
    for (const e of npEnds(toks, i + 1, { bareOk: true })) {
      if (tailOk(toks, e)) return true;
    }
  }
  return false;
}

// Verb-phrase + tail from toks[v] for one subject split.
function predicateOk(toks, v) {
  const verb = toks[v];
  if (verb === undefined) return false;
  if (COPULA_TOK.has(verb)) {
    const n1 = toks[v + 1];
    if (n1 === undefined) return false;
    if (n1 === 'made' && toks[v + 2] === 'of') {
      return npEnds(toks, v + 3, { bareOk: true }).some((e) => tailOk(toks, e));
    }
    if (isPartTok(n1)) {
      // passive mundane handling, optional resultative adjective
      if (tailOk(toks, v + 2)) return true;
      const adj = toks[v + 2];
      if (adj && !PREP_TOK.has(adj) && tailOk(toks, v + 3)) return true;
      return false;
    }
    if (PREP_TOK.has(n1)) return tailOk(toks, v + 1);
    // static attribute: 1..3 plain tokens, then PPs
    for (let e = v + 2; e <= Math.min(v + 4, toks.length); e++) {
      if (PREP_TOK.has(toks[e - 1]) || AUX_STOP_TOK.has(toks[e - 1])) break;
      if (tailOk(toks, e)) return true;
    }
    return false;
  }
  if (verb === 'has' || verb === 'had' || verb === 'have') {
    return npEnds(toks, v + 1, { bareOk: true }).some((e) => tailOk(toks, e));
  }
  if (PLACE_VERB_TOK.has(verb)) {
    return PREP_TOK.has(toks[v + 1]) && tailOk(toks, v + 1);
  }
  if (isPastTok(verb) || isPres3Tok(verb)) {
    // mundane organisation / coverage: object NP and/or PP tail
    for (const e of npEnds(toks, v + 1, { bareOk: true })) {
      if (tailOk(toks, e)) return true;
    }
    return toks[v + 1] !== undefined && tailOk(toks, v + 1);
  }
  return false;
}

// Independence asserted over explicitly UNDECLARED physical quantities
// ("Under each panel type, room temperature and humidity are
// independent.") is positive scene evidence: both relata are named
// concrete quantities (name-free by the caller's hits check), so the
// claim cannot contradict a license over the declared events. An
// anaphoric subject ("the two outputs ...") never qualifies — behaviour
// nouns and bare pair references stay material.
function undeclaredIndependenceOk(body) {
  const m = body.match(
    /^(?:(?:under|within|for|on|in)\s+(?:each|every)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*)?,\s*)?(.+)$/i
  );
  if (!m) return false;
  const toks = m[1].toLowerCase().split(' ');
  if (toks.length < 4 || toks.length > 9) return false;
  if (toks[toks.length - 1] !== 'independent') return false;
  if (!COPULA_TOK.has(toks[toks.length - 2])) return false;
  const subj = toks.slice(0, -2);
  if (!subj.includes('and')) return false;
  return subj.every(
    (t) => t === 'and' || (/^[a-z][a-z-]*$/.test(t) && !PRONOUN_TOK.has(t) &&
      !INERT_BEHAVIOR_TOK.has(t) && t !== 'two')
  );
}

export function isInertSceneSentence(raw) {
  const body = raw.trim().replace(/\s+/g, ' ').replace(/[.!?]+$/, '').trim();
  if (!body) return false;
  if (INERT_INVALID_RE.test(body) || INERT_EPISTEMIC_RE.test(body)) return false;
  // the undeclared-quantities independence form is the one inert frame
  // that may carry a leading scope comma; every other frame is a single
  // comma-free clause
  const indepForm = undeclaredIndependenceOk(body);
  if (/[,;:"“”‘’()—–]/.test(body) && !indepForm) return false;
  if (/[,;:"“”‘’()—–]/.test(body.replace(/^[^,]+,\s*/, ''))) return false;
  const toks = body.replace(/,/g, '').toLowerCase().split(' ');
  if (toks.length < 3 || toks.length > 16) return false;
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i];
    if (!/^[a-z][a-z-]*$|^\d+$/.test(t)) return false;
    if (INERT_FREQ_TOK.has(t) || INERT_EDITORIAL_TOK.has(t) ||
        INERT_BEHAVIOR_TOK.has(t) || INERT_MODAL_NEG_TOK.has(t) ||
        INERT_MUTATION_TOK.has(t) || INERT_BEHAVIOR_VERB_TOK.has(t) ||
        INERT_COUPLING_TOK.has(t)) return false;
    if (t.startsWith('simultan') || t.startsWith('activat') ||
        t.startsWith('trigger') || t.startsWith('restrict')) return false;
    if (t === "n't" || t.endsWith("n't")) return false;
    // a weekday is inert only as a one-off "on Friday" adjunct — never as
    // a recurrence ("every Tuesday" is already voided by 'every')
    if (WEEKDAY_TOK.has(t) && toks[i - 1] !== 'on') return false;
    // anaphoric pronouns void the evidence — except a determiner 'this'/
    // 'that' opening the concrete subject noun phrase
    if (PRONOUN_TOK.has(t) && i > 0) return false;
  }
  if (indepForm) return true;
  if (toks[0] === 'there') {
    if (!COPULA_TOK.has(toks[1])) return false;
    return npEnds(toks, 2, { bareOk: false }).some(
      (e) => toks.slice(2, e).some(isCountTok) && tailOk(toks, e)
    );
  }
  // concrete subject: [det] 1..4 nounish tokens, or COUNT 1..3 tokens.
  // Try every subject/verb split; one full parse is positive evidence.
  const tryFrom = (start, minNP, maxNP) => {
    // an anaphoric 'this/that <discourse-noun>' subject points back at a
    // prior claim, never at furniture — void it; a definite document
    // subject under mundane handling ("The report was formatted ...")
    // stays a concrete scene statement
    const anaphoric = toks[0] === 'this' || toks[0] === 'that';
    for (let e = start + minNP; e <= Math.min(start + maxNP, toks.length - 1); e++) {
      const np = toks.slice(start, e);
      if (np.some((t) => PREP_TOK.has(t) || AUX_STOP_TOK.has(t))) continue;
      if (anaphoric && HEAD_BAN_TOK.has(np[np.length - 1])) continue;
      // a mechanism subject under an active verb asserts what the
      // mechanism DOES ("A single switch feeds the whole rack.") — that
      // is structure, not scenery; copula/placement/passive stay inert
      if (MECH_HEAD_TOK.has(np[np.length - 1]) &&
          (isPastTok(toks[e]) || isPres3Tok(toks[e])) &&
          !PLACE_VERB_TOK.has(toks[e]) && !COPULA_TOK.has(toks[e])) continue;
      if (predicateOk(toks, e)) return true;
    }
    return false;
  };
  if (['the', 'a', 'an', 'each', 'this', 'that'].includes(toks[0])) {
    return tryFrom(1, 1, 4);
  }
  if (isCountTok(toks[0])) return tryFrom(1, 1, 3);
  return false;
}

export function findPremiseDoubt(X, segs = sentences(X)) {
  for (const seg of segs) {
    if (seg.text.includes('?')) continue;
    const marker = premiseDoubt(seg.text);
    if (marker) return { marker, span: { s: seg.s, e: seg.e } };
  }
  return null;
}

// --- query-path-scoped premise doubt ---------------------------------------
// A sentence that only reports an uninspected physical attribute ("The
// secondary dial hasn't been checked.")
// carries the surface shape of premise doubt but qualifies no stated
// figure, claim or queried event. It poisons nothing. The classification
// is deliberately narrow: a negated check/test/inspect predicate over a
// concrete noun-phrase subject, with zero data/claim nouns (a tally
// is evidence, not a body part), zero retraction vocabulary, and no bare
// anaphoric subject ("That hasn't been checked." points at the preceding
// claim and stays poisoning).
const UNCHECKED_PRED =
  /\b(?:hasn['’]t|haven['’]t|wasn['’]t|weren['’]t|has\s+not|have\s+not|was\s+not|were\s+not|is\s+not|are\s+not|not|never)\s+(?:(?:yet|ever)\s+)?(?:been\s+)?(?:checked|tested|inspected|examined|measured)\b|\b(?:unchecked|untested|uninspected|unexamined)\b|\bhas\s+yet\s+to\s+be\s+(?:checked|tested|inspected|examined|measured)\b/i;
const DOUBT_DATA_NOUN =
  /\b(?:tally|tallies|rates?|figures?|counts?|records?|readings?|results?|claims?|findings?|reports?|shares?|proportions?|percentages?|data|evidence|premises?|statements?|assertions?|conclusions?|notebooks?|logs?|ledgers?|manuals?|memos?|rules?|guidance|information|relays?|connections?|wiring|wires?|switch(?:es)?|circuits?|links?|linkage|couplings?|mechanisms?|setups?|arrangements?)\b/i;
const DOUBT_RETRACT_WORD =
  /\b(?:retract\w*|withdraw\w*|wrong|false|lost|missing|invalid|errors?|mistakes?|superseded|obsolete|rescinded|voided|disavowed|stricken|corrected|revised)\b/i;
const BARE_ANAPHOR_SUBJ =
  /^\s*["'“”‘’]?\s*(?:that|this|it|they)\s+(?:has|have|was|were|is|are|hasn['’]t|haven['’]t|wasn['’]t|weren['’]t)\b/i;

export function uncheckedOnlyMarker(text) {
  const m = UNCHECKED_PRED.exec(text);
  if (!m) return null;
  if (DOUBT_DATA_NOUN.test(text) || DOUBT_RETRACT_WORD.test(text)) return null;
  if (BARE_ANAPHOR_SUBJ.test(text)) return null;
  return m[0];
}

// Is `text` an unchecked-attribute remark that touches none of the
// query-relevant names and carries no probability literal?
export function isIrrelevantUnchecked(text, relevantNames = []) {
  if (!uncheckedOnlyMarker(text)) return false;
  if (probLikeLiteral(text)) return false;
  return !relevantNames.some((n) => n && inflectedNameRe(n).test(text));
}

const LOAD_BEARING_OPS = new Set([
  'PRIOR', 'LIK', 'POSTERIOR_STATED', 'JOINT_LIK', 'OBS',
  'COND_INDEP', 'DET_LINK', 'PART_EXHAUSTIVE',
]);

// Event names on the query's witness path: conditioned-on events, an event
// target, observed events, and events related by a declared license/joint.
function relevantEventNames(X, prog) {
  if (!prog) return [];
  const evNameById = eventNameMap(X, prog);
  const ids = new Set();
  for (const g of prog.query?.given ?? []) ids.add(g.ev);
  if (prog.query?.targetKind === 'ev' || evNameById.has(prog.query?.target)) {
    ids.add(prog.query.target);
  }
  for (const d of prog.decls) {
    if (d.op === 'OBS') ids.add(d.ev);
    if (d.op === 'COND_INDEP') for (const e of d.evs ?? []) ids.add(e);
    if (d.op === 'DET_LINK') { ids.add(d.a); ids.add(d.b); }
    if (d.op === 'JOINT_LIK') for (const s2 of d.spec ?? []) ids.add(s2.ev);
  }
  return [...ids].map((id) => evNameById.get(id)).filter(Boolean);
}

// An unchecked-attribute sentence is irrelevant to THIS program when it
// mentions no witness-path event and no load-bearing decl span sits in it.
export function irrelevantUncheckedSeg(X, prog, seg) {
  if (!isIrrelevantUnchecked(seg.text, relevantEventNames(X, prog))) return false;
  return !prog.decls.some(
    (d) => LOAD_BEARING_OPS.has(d.op) && d.span && d.span.s < seg.e && seg.s < d.span.e
  );
}

// Unrecorded-overall-tally remark in a program that declares ZERO
// priors ("The clerk kept no overall tally of how many items each
// workshop handled."): with no PRIOR premise it contradicts nothing —
// a direct-posterior program still solves, and a LIK-only program
// still refuses MISSING_BASE_RATE in the credal kernel untouched.
// With ANY PRIOR declared, the same sentence could dispute that prior
// and stays material. Single comma-free clause in a tight
// active/passive unrecorded-count frame only; zero literals, doubt,
// dependency/mechanism wording, declared names, or load-bearing spans.
const UNRECORDED_TALLY_ACTIVE =
  /^\s*(?:the|a|an|this|that)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2}\s+(?:kept|has|had|recorded|maintained|holds?)\s+no\s+(?:(?:overall|total|general|aggregate|separate|combined|master)\s+)?(?:tally|tallies|count|counts|record|records|log|logs|register|figures|totals?)\b[^.;,!?]*\.?\s*$/i;
const UNRECORDED_TALLY_PASSIVE =
  /^\s*no\s+(?:overall|total|general|aggregate|combined|master)\s+(?:tally|count|record|log|register|figures|totals?)\b[^.;,!?]*\b(?:was|were|is|are|has\s+been|have\s+been)\s+(?:kept|recorded|maintained|available|logged)\b[^.;,!?]*\.?\s*$/i;
export function irrelevantUnrecordedPriorSeg(X, prog, seg) {
  if (!prog || !seg) return false;
  if (prog.decls.some((d) => d.op === 'PRIOR')) return false;
  if (seg.text.includes('?') || QUESTION_VOCAB.test(seg.text)) return false;
  if (prog.decls.some(
    (d) => LOAD_BEARING_OPS.has(d.op) && d.span && d.span.s < seg.e && seg.s < d.span.e
  )) return false;
  if (probLikeLiteral(seg.text) || premiseDoubt(seg.text)) return false;
  if (DEPENDENCY_WORDS.test(seg.text) || MECHANISM_WORDS.test(seg.text)) return false;
  if (entityNames(X, prog).some((n) => n && inflectedNameRe(n).test(seg.text))) return false;
  if (/[,;]/.test(seg.text)) return false;
  return UNRECORDED_TALLY_ACTIVE.test(seg.text) ||
    UNRECORDED_TALLY_PASSIVE.test(seg.text);
}

// Qualitative sentence scoped exclusively to a DIFFERENT named
// stratum than the queried one ("On the lower deck every item, of
// either alloy, passes the load test." while the query keeps the
// upper deck): zero numeric literals, zero premise doubt, zero
// load-bearing spans, no two-event dependency claim, an affirmed
// stratum label whose canonical ctx differs from the query ctx, and
// NO mention — affirmed, negated, or as a hyphenated adjective — of
// the queried stratum itself. Querying the other stratum with only
// this stratum's premises still refuses in the kernel untouched.
export function irrelevantOtherStratumSeg(X, prog, seg) {
  const qCtx = prog?.query?.ctx;
  if (typeof qCtx !== 'string' || qCtx === 'ctx0' || qCtx === '*') return false;
  if (!prog || !seg) return false;
  if (seg.text.includes('?') || QUESTION_VOCAB.test(seg.text)) return false;
  if (probLikeLiteral(seg.text)) return false;
  if (premiseDoubt(seg.text)) return false;
  if (prog.decls.some(
    (d) => LOAD_BEARING_OPS.has(d.op) && d.span && d.span.s < seg.e && seg.s < d.span.e
  )) return false;
  if (DEPENDENCY_WORDS.test(seg.text)) {
    const evHits = [...eventNameMap(X, prog).values()]
      .filter((n) => n && inflectedNameRe(n).test(seg.text));
    if (evHits.length >= 2) return false;
  }
  const names = entityNames(X, prog);
  const excl = expandCtxExclude(names);
  const pol = contextLabelPolarity(seg.text, excl);
  if (pol.affirmed.size === 0) return false;
  const segCtx = canonicalCtxId(seg.text, excl);
  if (!segCtx || segCtx === qCtx) return false;
  // queried-stratum labels: the query ctx's own phrase plus every
  // label grounded on the query sentence
  const qLabels = new Set();
  const qName = qCtx.replace(/^ctx_/, '').replace(/_/g, ' ').trim();
  if (qName) qLabels.add(qName);
  for (const s of sentences(X)) {
    if (sentenceMateriality(s, names).kind !== 'query') continue;
    for (const l of ctxLabels(s.text, excl)) qLabels.add(l);
  }
  for (const l of [...pol.affirmed, ...pol.excluded]) {
    if (qLabels.has(l)) return false;
  }
  const escLbl = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  for (const ql of qLabels) {
    const re = new RegExp(`\\b${escLbl(ql).replace(/[\s-]+/g, '[\\s-]+')}\\b`, 'i');
    if (re.test(seg.text)) return false;
  }
  return true;
}

// --- conditional-query given grounding -------------------------------------
// A query sentence may itself condition on an event ("..., given that the
// notch is present?", "Given it carries a notch, ...?") without any separate
// declarative observation sentence. That explicit conditional clause is the
// given's textual witness; a given with neither an OBS sentence nor such a
// clause stays ungrounded. Returns the conditional clause's span, or null.
const QUERY_COND_MARKER =
  /\b(?:conditional\s+on|given\s+that|given\s+it|given|assuming\s+it|assuming\s+that|assuming)\b/gi;
const QUERY_GIVEN_NEG =
  /\b(?:not|no|never|without|n't|absent|missing|lacks?|lacking)\b/i;

export function queryConditionalGivenSpan(X, prog, evId, pol) {
  const evNameById = eventNameMap(X, prog);
  const evName = evNameById.get(evId);
  if (!evName) return null;
  const names = entityNames(X, prog);
  const evRe = inflectedNameRe(evName, 'gi');
  for (const seg of sentences(X)) {
    if (sentenceMateriality(seg, names).kind !== 'query') continue;
    QUERY_COND_MARKER.lastIndex = 0;
    let m;
    while ((m = QUERY_COND_MARKER.exec(seg.text)) !== null) {
      const from = m.index + m[0].length;
      const rest = seg.text.slice(from);
      const cut = rest.search(/[,;?]/);
      const clause = cut >= 0 ? rest.slice(0, cut) : rest;
      evRe.lastIndex = 0;
      if (!evRe.test(clause)) continue;
      const negated = QUERY_GIVEN_NEG.test(clause);
      if (pol === 'POS' && negated) continue;
      if (pol === 'NEG' && !negated) continue;
      return { s: seg.s + m.index, e: seg.s + from + clause.length };
    }
  }
  return null;
}

// findPremiseDoubt, scoped to the query path: irrelevant unchecked-attribute
// sentences are skipped; every other doubt (retraction, invalidation, or
// doubt touching a load-bearing premise / queried event) still poisons.
export function findPoisoningPremiseDoubt(X, prog = null, segs = sentences(X)) {
  for (const seg of segs) {
    if (seg.text.includes('?')) continue;
    const marker = premiseDoubt(seg.text);
    if (!marker) continue;
    if (prog && irrelevantUncheckedSeg(X, prog, seg)) continue;
    return { marker, span: { s: seg.s, e: seg.e } };
  }
  return null;
}

// Pure structural inventory-overview sentence: a single name-free
// clause that only introduces the passage's separate inventories or
// strata ("The depot keeps separate inventories of the valves on its
// upper and lower bays.", "The depot maintains separate records of
// the valves."). It carries zero probability literals, zero declared
// hypothesis or event names (conservative inflections included), and
// zero dependency/linkage/mechanism/retraction/doubt wording, so it
// asserts no premise the program must ledger. Any declared name,
// number, mechanism, or retraction in the same frame stays material.
const INVENTORY_OVERVIEW =
  /^\s*(?:(?:the|a|an|this|that)\s+)?[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,2}?\s+(?:keeps?|maintains?)\s+separate\s+(?:inventor(?:y|ies)|records?|logs?|tall(?:y|ies))\b/i;
const MECHANISM_WORDS =
  /\b(?:relays?|switch(?:es)?|wires?|wiring|circuits?|links?|linkage|couplings?|mechanisms?)\b/i;
export function isInventoryOverviewSentence(text, names) {
  if (/[,;:]/.test(text)) return false;
  if (!INVENTORY_OVERVIEW.test(text)) return false;
  if (probLikeLiteral(text) || premiseDoubt(text)) return false;
  if (DEPENDENCY_WORDS.test(text) || MECHANISM_WORDS.test(text)) return false;
  return !names.some((n) => n && inflectedNameRe(n).test(text));
}

// Pure observational/inspection-method background remark: a single
// comma-free clause in the narrow "<subject> can tell/see/check ...
// which ... by <gerund> ..." frame ("You can tell which coating they
// applied by inspecting the seam."), with zero probability literals,
// zero declared names, zero dependency/mechanism/doubt wording, and
// zero negation/editorial/behaviour/coupling/mutation/frequency
// tokens (the frame's own "can" is the one modal allowed). It only
// says an undeclared attribute is inspectable; any declared name,
// figure, or retraction in the frame stays material.
const INSPECTION_METHOD =
  /^\s*(?:you|one|(?:the|a|an)\s+[a-z][a-z-]*|[a-z]+s)\s+can\s+(?:tell|see|check|identify|determine|spot|distinguish)\s+(?:which|what|whether)\s+[a-z][a-z\s'-]{1,60}\bby\s+[a-z]{3,}ing\b[a-z\s'-]{0,50}\.?\s*$/i;
export function isInspectionMethodSentence(text, names = []) {
  if (/[,;:?!]/.test(text)) return false;
  const toks = text.toLowerCase().match(/[a-z][a-z'-]*/g) ?? [];
  if (toks.length === 0 || toks.length > 22) return false;
  if (!INSPECTION_METHOD.test(text)) return false;
  if (probLikeLiteral(text) || premiseDoubt(text)) return false;
  if (DEPENDENCY_WORDS.test(text) || MECHANISM_WORDS.test(text)) return false;
  if ((names ?? []).some((n) => n && inflectedNameRe(n).test(text))) return false;
  let canSeen = false;
  for (const t of toks) {
    if (t.endsWith("n't")) return false;
    if (t === 'can' && !canSeen) { canSeen = true; continue; }
    if (INERT_MODAL_NEG_TOK.has(t) || INERT_FREQ_TOK.has(t) ||
        INERT_EDITORIAL_TOK.has(t) || INERT_BEHAVIOR_TOK.has(t) ||
        INERT_MUTATION_TOK.has(t) || INERT_COUPLING_TOK.has(t) ||
        INERT_BEHAVIOR_VERB_TOK.has(t)) return false;
  }
  return canSeen;
}

// Classify one sentence's materiality given the declared entity names.
//   'query'     — carries the ASK_PROB grounding sentence
//   'material'  — asserts content the program must ledger
//   'fluff'     — narrative filler; safe to leave ungrounded
export function sentenceMateriality(seg, names) {
  const t = seg.text;
  if (t.includes('?')) return { kind: 'query' };
  if (premiseDoubt(t)) return { kind: 'material', reason: 'premise doubt or retraction' };
  if (probLikeLiteral(t)) return { kind: 'material', reason: 'probability literal' };
  const hits = names.filter((n) => wordRe(n).test(t));
  if (hits.length >= 2) {
    return { kind: 'material', reason: `${hits.length} declared entities co-mentioned`, names: hits };
  }
  if (hits.length === 1 && DEPENDENCY_WORDS.test(t)) {
    return { kind: 'material', reason: 'entity under dependency/scope wording', names: hits };
  }
  // a proven inspection-method remark ("You can tell what dye was
  // used by checking the lining.") carries question vocabulary in its
  // embedded clause but asks nothing — its zero-name/literal/doubt
  // guards were all rechecked in isInspectionMethodSentence
  if (QUESTION_VOCAB.test(t) && !isInspectionMethodSentence(t, names)) {
    return { kind: 'query' };
  }
  // Positive-evidence rule (Bulma R3 F2a): ANY sentence mentioning a
  // declared entity is material — a lone counterexample over one event
  // ("The pump occasionally runs on its own.") can never fall through to
  // filler. Ungrounded, the ledger refuses it as
  // UNACCOUNTED_MATERIAL_SENTENCE.
  if (hits.length === 1) {
    return { kind: 'material', reason: `declared entity "${hits[0]}" mentioned`, names: hits };
  }
  if (ANAPHORA.test(t) && DEPENDENCY_WORDS.test(t)) {
    return {
      kind: 'material',
      reason: 'anaphoric entity co-reference under dependency wording',
    };
  }
  // Name-free sentences need POSITIVE inert-scene evidence (F2b); the
  // default is material, fail-closed. A pure inventory-overview frame
  // is positive evidence of the same kind (zero names, literals,
  // dependency/mechanism/doubt wording — every guard rechecked there).
  if (isInventoryOverviewSentence(t, names)) return { kind: 'fluff' };
  if (isInspectionMethodSentence(t, names)) return { kind: 'fluff' };
  if (isInertSceneSentence(t)) return { kind: 'fluff' };
  return { kind: 'material', reason: 'no positive inert-scene evidence (fail-closed)' };
}

// ---------------------------------------------------------------------------
// population ctx labels — lowercase "<modifier> <population-noun>" phrases and
// non-initial capitalized names label a sentence's population stratum. Moved
// here from invariance/ownership.js so the kernel itself can ground premise
// contexts without an invariance-layer dependency.
// ---------------------------------------------------------------------------
const POP_NOUN_LIST =
  'shift|cohort|batch|intake|group|crew|team|session|run|ward|clinic|campus|depot|plant|store|site|line|class|section|division|region|district|precinct|branch|office|unit|cell|block|wave|phase|arm';
const POP_NOUNS = `(?:${POP_NOUN_LIST})`;
const POP_STOP = new Set([
  'the', 'a', 'an', 'each', 'every', 'either', 'neither', 'both', 'this',
  'that', 'these', 'those', 'its',
  'their', 'his', 'her', 'our', 'your', 'one', 'per', 'of', 'in', 'on', 'at',
  'for', 'to', 'and', 'or', 'any', 'all', 'some', 'no', 'other', 'another',
  'same', 'such', 'which', 'what', 'whatever', 'whichever', 'random',
  'randomly', 'selected', 'given', 'certain', 'whole', 'entire', 'single',
]);

// Canonical CtxId for the population labels a sentence carries — the same
// naming the live extractor uses: `ctx_<sorted labels joined by '_'>`, or
// null when the sentence carries no population label.
export function canonicalCtxId(sentence, excludeLc) {
  const ls = [...ctxLabels(sentence, excludeLc)].sort();
  return ls.length ? `ctx_${ls.join('_').replace(/\s+/g, '_')}` : null;
}

// A population label inside a membership-denying clause — "is not a
// night-shift panel", "is outside the night shift", "a non-night-shift
// panel" — is negative evidence, never a positive context label. The
// negation window is the mention's own clause: text back to the previous
// clause boundary (a comma/semicolon/paren/dash resets it, so "... not
// damaged, is a night-shift panel" still counts positive).
const CTX_NEGATION =
  /\bnot\b|\b\w+n['’]t\b|\bnever\b|\bno\s+longer\b|\bnon-|\boutside\b|\bexcluding\b|\bexcludes?\b|\bexcept\b|\bwithout\b|\binstead\s+of\b|\brather\s+than\b|\bother\s+than\b|\bfar\s+from\b|\baway\s+from\b/i;

const CLAUSE_BOUNDARY = /[,;:()—–?!]/;

// English number/fraction words are never population labels: a
// capitalised "Half" / "Three" / "One" (sentence-initial or not) is a
// quantity, not a stratum.
const CTX_NUMBER_WORDS = new Set(
  ('zero one two three four five six seven eight nine ten eleven twelve ' +
    'thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty ' +
    'thirty forty fifty sixty seventy eighty ninety hundred thousand million ' +
    'half halves quarter quarters third thirds fourth fourths fifth fifths ' +
    'sixth sixths seventh sevenths eighth eighths ninth ninths tenth tenths ' +
    'eleventh elevenths twelfth twelfths twentieth twentieths').split(' ')
);

// Expand an exclusion set with every name's individual words: excluding
// a multi-word hypothesis name ("Alpha press") must also exclude its
// narrowed head name ("alpha") and its other content words, or a query
// naming the head alone ("came from Alpha?") manufactures a bogus
// population label. Genuine stratum labels are unaffected unless they
// literally reuse a declared entity's own word.
export function expandCtxExclude(names) {
  const out = new Set();
  for (const raw of names ?? []) {
    if (typeof raw !== 'string') continue;
    const n = raw.trim();
    if (!n) continue;
    out.add(n.toLowerCase());
    for (const tok of n.toLowerCase().split(/[\s-]+/)) {
      if (tok) out.add(tok);
    }
  }
  return out;
}

function expandedExcludeLc(excludeLc) {
  return expandCtxExclude(excludeLc ? [...excludeLc] : []);
}

// Genuine stratum modifiers discriminate populations: directional,
// temporal, ordinal and comparative/positional words. An introductory
// framing subject ("A shipping office has a box of cards ...", "A repair
// shop tossed its ...") is scene-setting, not a stratum — its modifier
// belongs to none of these classes and its place noun heads the opening
// subject, so it must not manufacture a CONTEXT_MISMATCH on a
// single-population passage.
const STRATUM_MODIFIERS = new Set([
  // directional
  'north', 'south', 'east', 'west', 'northern', 'southern', 'eastern',
  'western', 'central', 'northeast', 'northwest', 'southeast', 'southwest',
  // temporal
  'day', 'night', 'morning', 'evening', 'daily', 'weekly', 'monthly',
  'daytime', 'overnight', 'early', 'late',
  // ordinal
  'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
  'eighth', 'ninth', 'tenth', 'primary', 'secondary', 'initial', 'final',
  // comparative / positional
  'upper', 'lower', 'higher', 'inner', 'outer', 'top', 'bottom', 'front',
  'back', 'left', 'right', 'main',
]);
// Place nouns that are themselves strong stratum markers.
const STRATUM_NOUNS = new Set([
  'shift', 'cohort', 'stratum', 'ward', 'branch', 'region', 'district',
  'precinct', 'division', 'section', 'lane', 'wave', 'phase', 'intake',
]);
const FRAMING_VERB = new Set([
  'has', 'have', 'had', 'is', 'was', 'are', 'were', 'holds', 'held',
  'keeps', 'kept', 'stocks', 'stocked', 'stores', 'stored', 'tossed',
  'tosses', 'files', 'filed', 'received', 'receives', 'shipped', 'ships',
  'handles', 'handled', 'opened', 'opens', 'closed', 'closes', 'reports',
  'reported', 'noted', 'notes', 'stands', 'stood', 'sits', 'sat',
]);

// Is this lowercase "<mod> <place-noun>" mention the sentence's
// introductory framing subject? True only when: nothing but an optional
// article precedes it, a finite-style verb immediately follows it (the
// place noun IS the subject head — a stratum use instead pre-modifies a
// following item noun, as in "A night shift panel ..."), and neither the
// modifier nor the noun is a genuine stratum marker.
function isFramingSubject(sentence, m) {
  if (STRATUM_MODIFIERS.has(m[1].toLowerCase())) return false;
  if (STRATUM_NOUNS.has(m[2].toLowerCase())) return false;
  if (!/^\s*(?:a|an|the)?\s*$/i.test(sentence.slice(0, m.index))) return false;
  const next = /^[ \t]+([a-z][a-z'-]*)/i.exec(sentence.slice(m.index + m[0].length));
  if (!next) return false;
  const w = next[1].toLowerCase();
  return FRAMING_VERB.has(w) || /^[a-z]{3,}ed$/.test(w);
}

// Paired spatial/directional stratum phrases in a locative frame
// ("on the upper deck", "of the upper deck's", "from the lower deck",
// "at the north pier"): the POP_NOUNS list cannot enumerate every
// place noun, so a directional modifier + lowercase place noun under
// a locative/origin/possessive preposition also labels a stratum.
// Generic abstract/part/measure heads never do, and neither does a
// word belonging to a declared entity name.
const SPATIAL_STRATUM_MODS =
  'upper|lower|inner|outer|north|south|east|west|northern|southern|eastern|western|eastbound|westbound|northbound|southbound|uphill|downhill';
const SPATIAL_NOUN_STOP = new Set(
  ('time place case part side end corner edge cover face hand way order step test check point margin border surface half quarter third portion share rate count number total sum kind type sort form state stage item unit panel sample specimen reading').split(' ')
);
// Conservative singular for a spatial place noun: plural 's' strips,
// but '-ss'/'-us'/'-is' endings are not plurals ("class", "campus").
function spatialSingularOf(w) {
  const lc = w.toLowerCase();
  if (lc.length > 4 && lc.endsWith('ies')) return `${lc.slice(0, -3)}y`;
  if (/(?:ss|us|is)$/.test(lc)) return lc;
  if (/(?:x|ch|sh)es$/.test(lc)) return lc.slice(0, -2);
  if (lc.endsWith('s')) return lc.slice(0, -1);
  return lc;
}

function ctxLabelMentions(sentence, excludeLc) {
  const out = [];
  const excl = expandedExcludeLc(excludeLc);
  const firstWordEnd = sentence.search(/\S/);
  for (const m of sentence.matchAll(/\b[A-Z][a-zA-Z]{2,}\b/g)) {
    if (m.index <= firstWordEnd + 1) continue;
    const w = m[0];
    const wl = w.toLowerCase();
    if (!excl.has(wl) && !CTX_NUMBER_WORDS.has(wl)) {
      out.push({ label: wl, s: m.index, e: m.index + m[0].length });
    }
  }
  const popRe = new RegExp(`\\b([a-z][a-z']*)[\\s-](${POP_NOUNS})(?:s|'s|’s)?\\b`, 'g');
  // non-overlapping scan: "day-shift item" yields 'day shift' once — the tail
  // "shift item" is not itself a population label
  let lastEnd = -1;
  for (const m of sentence.matchAll(popRe)) {
    if (m.index < lastEnd) continue;
    const mod = m[1];
    if (POP_STOP.has(mod) || excl.has(mod)) continue;
    if (isFramingSubject(sentence, m)) continue;
    out.push({ label: `${mod} ${m[2]}`, s: m.index, e: m.index + m[0].length });
    lastEnd = m.index + m[0].length;
  }
  // paired spatial strata under a locative frame; the mention range
  // covers "<mod> <noun>" itself (the preposition is frame, not label)
  const spatialRe = new RegExp(
    `((?:on|in|at|from|of)\\s+(?:the|its|their|a|an)\\s+)(${SPATIAL_STRATUM_MODS})[\\s-]([a-z]{3,})(?:'s|’s)?\\b`,
    'gi'
  );
  for (const m of sentence.matchAll(spatialRe)) {
    // modifier and noun are lowercase as written (same discipline as
    // the POP_NOUNS scan; a capitalized name is the scan above's)
    if (m[2] !== m[2].toLowerCase() || m[3] !== m[3].toLowerCase()) continue;
    const mod = m[2].toLowerCase();
    const noun = spatialSingularOf(m[3]);
    if (POP_STOP.has(mod) || POP_STOP.has(noun)) continue;
    if (excl.has(mod) || excl.has(noun)) continue;
    if (SPATIAL_NOUN_STOP.has(noun)) continue;
    const s = m.index + m[1].length;
    const e = m.index + m[0].length;
    if (out.some((o) => s < o.e && o.s < e)) continue;
    out.push({ label: `${mod} ${noun}`, s, e });
  }
  return out;
}

function negatedMention(sentence, m) {
  const left = sentence.slice(0, m.s);
  let clauseStart = 0;
  for (let i = 0; i < left.length; i++) {
    if (CLAUSE_BOUNDARY.test(left[i])) clauseStart = i + 1;
  }
  return CTX_NEGATION.test(left.slice(clauseStart));
}

export function contextLabelPolarity(sentence, excludeLc) {
  const affirmed = new Set();
  const excluded = new Set();
  for (const m of ctxLabelMentions(sentence, excludeLc)) {
    (negatedMention(sentence, m) ? excluded : affirmed).add(m.label);
  }
  for (const label of excluded) affirmed.delete(label);
  return { affirmed, excluded };
}

export function ctxLabels(sentence, excludeLc) {
  return contextLabelPolarity(sentence, excludeLc).affirmed;
}

// population labels the sentence affirmatively DENIES membership in —
// "is not a night-shift panel" carries 'night shift' here, so a premise
// restricted to that population is a positive mismatch, not an unlabeled one.
export function negatedCtxLabels(sentence, excludeLc) {
  return contextLabelPolarity(sentence, excludeLc).excluded;
}

// Contrasting-adjective equal-rate arms: "with a <adj1> <noun> as for
// ... with a <adj2> <noun>" (a bright lamp
// vs a dim lamp). The contrast asserts independence only when the two
// adjectives differ on the same head noun; identical arms assert
// nothing, and a negated equality ("not the same", "differs") never
// licenses. Returns true when no with/with contrast is present.
const CONTRAST_STOP = new Set([
  'as', 'for', 'with', 'without', 'the', 'a', 'an', 'and', 'or',
  'is', 'are', 'was', 'were', 'same', 'identical', 'equal', 'no',
]);
function contrastingWithArms(text) {
  const m = /\bwith\s+(?:a|an|the)\s+([a-z][a-z-]*)\s+([a-z][a-z-]*)\b[^.;]*?\bas\s+for\b[^.;]*?\bwith\s+(?:a|an|the)\s+([a-z][a-z-]*)\s+([a-z][a-z-]*)\b/i.exec(text);
  if (!m) return null;
  const [adj1, head1, adj2, head2] = m.slice(1, 5).map((s) => s.toLowerCase());
  if ([adj1, head1, adj2, head2].some((t) => CONTRAST_STOP.has(t))) return null;
  return { adj1, head1, adj2, head2 };
}
function contrastingArmsOk(text) {
  const c = contrastingWithArms(text);
  if (!c) return true;
  if (/\bnot\s+(?:the\s+same|identical|equal)\b|\bdiffer\w*\b|\bdifferent\b/i.test(text)) return false;
  return c.adj1 !== c.adj2 && singularOf(c.head1) === singularOf(c.head2);
}

// ---------------------------------------------------------------------------
// affirmative-independence wording — a COND_INDEP license is only valid over
// text that ASSERTS independence. Negated, hedged, open-question or
// dependence-asserting clauses all fail. Moved here from invariance/gate.js
// so the kernel and gate share one implementation.
// ---------------------------------------------------------------------------
export function assertsIndependence(text) {
  const masked = text
    .replace(/\bnot\s+confounded\b/gi, ' INDEP_OK ')
    .replace(/\bfree\s+of\s+confound\w*\b/gi, ' INDEP_OK ')
    .replace(/\bwithout\s+confound\w*\b/gi, ' INDEP_OK ');
  // The mathematical definition of conditional independence also asserts
  // it: an explicit equal-conditional-rate statement — "<B> rate is the
  // same for items with <A> as for items without <A>", "probability of
  // <B> is identical whether <A> occurs or not".
  const RATE_NOUN = '(?:rates?|probability|likelihood|chance|frequency|frequencies|proportion)';
  const EQUAL_RATE = new RegExp(
    `\\b${RATE_NOUN}\\b[^.;]{0,60}?\\b(?:is|remains?|stays?)\\s+(?:the\\s+same|identical|equal|unchanged)\\b[^.;]{0,100}?\\bwhether\\b[^.;]{0,80}?\\bor\\s+not\\b` +
    `|\\b${RATE_NOUN}\\b[^.;]{0,60}?\\b(?:is|remains?|stays?)\\s+(?:the\\s+same|identical|equal|unchanged)\\s+for\\b[^.;]{0,100}?\\bwith\\b[^.;]{0,80}?\\bas\\s+for\\b` +
    `|\\b(?:same|identical|equal)\\s+${RATE_NOUN}\\b[^.;]{0,100}?\\bwhether\\b[^.;]{0,80}?\\bor\\s+not\\b`,
    'i'
  );
  const POS =
    /\bdependen\w+\s+independen\w+|\bindependen\w+\b|\buncorrelated\b|\bseparately administered\b|\bINDEP_OK\b/i;
  const NEG =
    /\b(?:not|n't|never|no\s+longer|without|fails?|failing|lack\w*|denies?|denied|refus\w+)\b[^.;]{0,40}?\bindependen\w+\b|\bdependent\b|\bdepends\b|\bcorrelated\b|\bcorrelation\b|\bconfounded\b|\blinked\b|\bcoupled\b|\bentangled\b|\binteract\w+\b|\bwired\b|\bganged\b|\bslaved\b|\bin\s+lockstep\b|\bin\s+tandem\b|\bin\s+unison\b|\bsimultaneous\w*\b|\bco-?occur\w*\b|\bsame\s+(?:switch|trigger|source|driver|cause|mechanism|control|signal)\b|\balways\s+\w+\s+together\b|\bfire\s+together\b|\bactivat\w+\s+together\b/i;
  const EQUAL_NEG = new RegExp(
    `\\bnot\\s+(?:the\\s+same|identical|equal)\\b|\\b${RATE_NOUN}\\b[^.;]{0,40}?\\b(?:differ\\w*|different|var(?:y|ies|ied))\\b|\\bnever\\s+matched\\b`,
    'i'
  );
  const HEDG =
    /\bopen\s+question\b|\bremains?\s+(?:to\s+be|an?\s+open|unresolve\w*|undetermin\w*|unverif\w*|unestablish\w*|untest\w*|unconfirm\w*|unknown|unclear)\b|\bunverif\w+\b|\buntest\w+\b|\bunconfirm\w+\b|\bunestablish\w+\b|\bunproven\b|\bunknown\b|\bundetermin\w+\b|\bunclear\b|\buncertain\b|\bspeculat\w+\b|\bconjectur\w+\b|\btentativ\w+\b|\bdebated?\b|\bin\s+dispute\b|\bquestionable\b|\bassum\w+\b|\bhypothesi\w+\b|\bnot\s+(?:yet\s+)?(?:established|verified|confirmed|known|determined|settled)\b/i;
  // Conditional independence is asserted or it is not: hedged degrees
  // ("partially independent", "roughly independent") and population-
  // restricted independence ("independent on some panels") license nothing.
  const PART =
    /\b(?:partially|roughly|approximately|mostly|somewhat|nearly|largely|fairly|essentially|basically|practically|virtually|almost)\s+(?:conditionally\s+)?independen\w+|\bindependen\w*\b[^.;]{0,30}?\bon\s+some\b|\bon\s+some\b[^.;]{0,30}?\bindependen\w*/i;
  return (POS.test(masked) || EQUAL_RATE.test(masked)) && !NEG.test(masked) && !EQUAL_NEG.test(masked) && !HEDG.test(masked) &&
    !PART.test(masked) && !premiseDoubt(masked) && contrastingArmsOk(masked);
}

// ---------------------------------------------------------------------------
// declared-event reference resolution — a license span must reference the
// events it claims to relate, either by name inside its sentence or
// anaphorically ("the two outputs share a relay" resolves to the two most
// recently named declared events).
// ---------------------------------------------------------------------------
const escRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// EVENT decl spans -> {id -> name} map.
export function eventNameMap(X, prog) {
  const out = new Map();
  for (const d of prog.decls) {
    if (d.op === 'EVENT' && d.span) {
      const t = X.slice(d.span.s, d.span.e).trim();
      if (t.length >= 2) out.set(d.id, t);
    }
  }
  return out;
}

// Latest mention position of each declared event name before `pos`.
function lastMentions(X, pos, evNameById) {
  const out = [];
  const head = X.slice(0, pos);
  for (const [id, name] of evNameById) {
    if (!name || name.length < 2) continue;
    const re = inflectedNameRe(name, 'gi');
    let m;
    let last = -1;
    while ((m = re.exec(head)) !== null) last = m.index;
    if (last >= 0) out.push({ id, pos: last });
  }
  return out.sort((a, b) => b.pos - a.pos);
}

// The resolver only certifies local antecedents. A separate intervening
// sentence must itself name a candidate event, and a competing pair or
// plural reference within an event sentence makes the resolution ambiguous.
function unambiguousAntecedents(X, seg, anaphor, last, evNameById) {
  if (last[0].pos >= seg.s || last.some(({ id, pos }, i) =>
    i > 0 && pos + evNameById.get(id).length > last[i - 1].pos
  )) return false;
  const names = last.map(({ id }) => evNameById.get(id));
  const events = new RegExp(`\\b(?:${names.map(escRe).join('|')})\\b`, 'gi');
  const recent = sentences(X.slice(0, seg.s)).filter(({ e }) => e > last.at(-1).pos);
  const plural = /\b(?:the|these|those|some|several|both|many|multiple|various|two|three|four|beside|near|with|alongside|contains?|includes?|houses?|holds?)\s+(?:[a-z-]+\s+){0,2}(?:[a-z-]+s|people|children|equipment)\b|\b(?:[a-z-]+s|people|children)\s+(?:are|were|have|had)\b|\b(?:pair|duo|couple|group|set)\s+of\b/i;
  const nounList = /\b(?:a|an|the)\s+[a-z-]+\s*,\s*(?:a|an|the)\s+[a-z-]+\b/i;
  // Denominator populations in rate expressions belong to the named
  // event's evidence. They do not introduce additional events.
  const withoutRates = (t) => t
    .replace(/\b\d+(?:\.\d+)?%\s+of\s+(?:[a-z-]+\s+)?[a-z-]+(?=\s*(?:and\s+\d|[.!?]|$))/gi, '#')
    .replace(/#\s+and\s+#/gi, '#')
    .replace(/\bat a rate expressed as (?:\d+\/\d+|\d+ out of \d+|\d+(?:\.\d+)?%|odds of \d+ to \d+ in favor|a \d+:\d+ ratio of successes to failures)(?: among [a-z-]+ cases)?(?=[.!?]|$)/gi, '#');
  const interferes = (t) => {
    const masked = withoutRates(t)
      .replace(events, '@')
      .replace(/@\s+(?:and|or)\s+(?:the\s+)?@/gi, '@');
    return /\b(?:and|or|plus|as well as|along with)\b/i.test(masked) ||
      plural.test(masked) || nounList.test(masked);
  };
  for (const sentence of recent) {
    // A preceding, canonical anaphoric relation can continue the same
    // reference chain ("The two outputs share a relay. These outputs ...").
    if (assertsUniversalRelation('EQUIV', sentence.text)) continue;
    if (!names.some((name) => wordRe(name).test(sentence.text)) || interferes(sentence.text)) {
      return false;
    }
    for (const clause of withoutRates(sentence.text).split(/[;:—–]/).slice(1)) {
      if (clause.trim() && !names.some((name) => wordRe(name).test(clause))) return false;
    }
  }
  return !interferes(seg.text.slice(0, anaphor.index));
}

/**
 * referencesEvents(X, seg, evIds, evNameById) -> {ok, via}
 * Does the sentence `seg` ({s,e,text} in X-coordinates) reference every event
 * in evIds — by name in the sentence, or by a plural anaphor resolving to
 * exactly the |evIds| most recently named declared events?
 */
export function referencesEvents(X, seg, evIds, evNameById) {
  const mentioned = new Set();
  for (const [id, name] of evNameById) {
    if (name && name.length >= 2 && inflectedNameRe(name).test(seg.text)) mentioned.add(id);
  }
  if (evIds.every((e) => mentioned.has(e))) return { ok: true, via: 'names' };
  const anaphor = ANAPHORA.exec(seg.text);
  if (anaphor) {
    const phrase = seg.text.slice(anaphor.index);
    const pair = /^(?:both|the\s+(?:two|pair|duo)|(?:these|those)\s+two)\b/i.test(phrase);
    const last = lastMentions(X, seg.s + anaphor.index, evNameById).slice(0, evIds.length);
    const resolved = new Set(last.map((x) => x.id));
    if ((!pair || evIds.length === 2) &&
      resolved.size === evIds.length && evIds.every((e) => resolved.has(e)) &&
      unambiguousAntecedents(X, seg, anaphor, last, evNameById)) {
      return { ok: true, via: 'anaphora' };
    }
  }
  return { ok: false };
}

// Relation wording a DET_LINK span must actually express.
export const REL_WORDS = {
  EQUIV:
    /\bwired\b|\bhard-?wired\b|\bsame\s+(?:switch|trigger|source|driver|cause|mechanism|control|signal|circuit|relay)\b|\brelays?\b|\bcircuits?\b|\blockstep\b|\bin\s+tandem\b|\bin\s+unison\b|\btogether\b|\bsimultaneous\w*\b|\bco-?occur\w*\b|\bequivalent\b|\bidentical\b|\bif and only if\b|\biff\b|\bganged\b|\bslaved\b|\byok\w+\b|\bcoupled\b|\bactivat\w*\s+together\b|\bfire\w*\s+together\b|\bdaisy-?chain\w*\b|\bshar\w+\s+a\b/i,
  IMPLIES:
    /\bwhenever\b|\bimplies\b|\bimplication\b|\bonly\s+(?:when|if)\b|\bif\b[^.;]{0,60}\bthen\b|\btriggers?\b|\bforces?\b|\bguarantee\w*\b|\bso\s+long\s+as\b|\brequires?\s+(?:the\s+\w+\s+(?:to\s+)?)?(?:fire|sound|activate|light|trigger|also)\b|\bmust\s+(?:also\s+)?(?:fire|sound|activate|light|trigger|be)\b/i,
  MUTEX:
    /\bmutually exclusive\b|\bexclusiv\w+\b|\bmutex\b|\bnever\s+both\b|\bnot\s+both\b|\bcannot\s+both\b|\bcan'?t\s+both\b|\bat\s+most\s+one\b|\bincompatible\b|\bdisjoint\b|\bpreclude\w*\b|\brules?\s+out\b|\bimpossible\b[^.;]{0,40}\bboth\b|\bboth\b[^.;]{0,40}\bimpossible\b|\beither\b[^.;]{0,30}\bbut\s+never\s+both\b|\bnever\b[^.;]{0,40}?\btogether\b|\b(?:do|does|did)\s+not\b[^.;]{0,40}?\btogether\b|\bnever\b[^.;]{0,40}?\bsimultaneous\w*\b|\bnever\b[^.;]{0,40}?\bco-?occur\w*\b|\bonly\s+one\b[^.;]{0,30}\bat\s+a\s+time\b/i,
};

// Wording that contradicts an affirmative co-activation claim: negation and
// exclusion vocabulary. An EQUIV or IMPLIES span carrying any of these
// asserts the OPPOSITE polarity of the declared relation — "the lamp and
// buzzer never activate together" cannot license EQUIV (it is a MUTEX fact)
// and "a fires only when b does not" cannot license IMPLIES.
const NEG_EXCL_WORDS =
  /\bnever\b|\bnot\b|\bn't\b|\bcannot\b|\bcan'?t\b|\bcan\s+not\b|\bneither\b|\bnor\b|\bno\s+longer\b|\bfails?\s+to\b|\bfailed\s+to\b|\bunable\s+to\b|\bmutually\s+exclusive\b|\bexclusiv\w+\b|\bmutex\b|\bdisjoint\b|\bincompatible\b|\bat\s+most\s+one\b|\bpreclude\w*\b|\brules?\s+out\b|\bbut\s+not\s+both\b|\bonly\s+one\b|\bno\s+more\s+than\s+one\b|\bone\s+or\s+the\s+other\b/i;

// Hedged or partial co-occurrence: the sentence admits BOTH joint and
// separate outcomes ("sometimes activate together and sometimes activate
// separately") — it asserts no deterministic relation, so it licenses no
// DET_LINK direction at all.
const PARTIAL_WORDS =
  /\bsometimes\b|\boccasionally\b|\bat\s+times\b|\bintermittent\w*\b|\bsporadic\w*\b|\bseparat\w*\b|\bindependent\w*\b|\bpart\s+of\s+the\s+time\b|\bon\s+and\s+off\b|\bfrom\s+time\s+to\s+time\b/i;

const POSITIVE_LINK_DENY = new RegExp(
  `(?:${NEG_EXCL_WORDS.source})|(?:${PARTIAL_WORDS.source})`,
  'i'
);

// ---------------------------------------------------------------------------
// Universal Entailment Principle — R5 structural invariants (fail-closed).
// A DET_LINK asserts a deterministic relation over the WHOLE queried
// population, not a merely observed co-occurrence:
//   EQUIV    forall x: e0(x) <-> e1(x)
//   IMPLIES  forall x: e0(x)  -> e1(x)
//   MUTEX    forall x: not (e0(x) and e1(x))
// Substring cue + deny-list matching is fundamentally defeatable: an outer
// modal, subset, or hypothesis clause can always wrap the cue ("On a subset
// of panels, ... always ...", "It is possible that ...", "... on Ash
// panels"). Licensing therefore does NOT scan for cues — the ENTIRE sentence
// containing the license span must match an anchored ^...$ canonical
// universal form: an optional declared-population scope phrase, the
// universal core, and zero other clauses or modifiers. Anything the grammar
// does not enumerate fails closed.
// ---------------------------------------------------------------------------
const VERB = `[a-z][a-z-]*`;
// Finite verbs licensing an ongoing/general premise exclude historical
// inflections. Participles below present auxiliaries and verbs below
// negative/modal auxiliaries occupy different slots.
const PAST_VERBS =
  'was|were|had|did|arose|ate|awoke|became|began|bent|bit|bled|blew|bore|bought|broke|brought|built|burnt|came|caught|chose|clung|crept|dealt|drank|drew|drove|dug|fed|fell|felt|fled|flew|flung|forbade|forgot|fought|froze|gave|got|grew|heard|held|hung|kept|knew|laid|lay|leapt|led|left|lent|lit|lost|made|meant|met|paid|rang|ran|rode|rose|said|sang|sank|sat|saw|sent|shone|shook|shot|shrank|slept|slid|sold|sought|spat|sped|spoke|spun|sprang|stood|stole|struck|swam|swept|swore|swung|taught|thought|threw|told|took|tore|understood|went|wept|woke|won|wore|wove|wrote|wrung';
const PRESENT_VERB =
  `(?:(?:bleed|breed|feed|heed|need|seed|shed|speed|weed)|(?!(?:${PAST_VERBS}|[a-z-]+ed)\\b)${VERB})`;
// discourse adverbs permitted between subject and the universal quantifier
// ("These outputs therefore always activate together.")
const LINK_ADVERB = `(?:therefore|thus|hence|consequently|accordingly)`;
// The anaphoric subject can end at the pronoun or at an explicit event
// classifier. An open-ended noun phrase can consume an embedded qualifier
// before the anchored relation ever sees it.
const EVENT_CLASSIFIER =
  `(?:events|outputs|signals|tests|readings|indicators|alarms|results|cues|observations|findings|checks)`;
const ANA_SUBJ =
  `(?:they|the\\s+(?:pair|duo)|(?:the\\s+two|both)\\s+of\\s+them|` +
  `(?:the\\s+two|these|those|both)(?:\\s+${EVENT_CLASSIFIER})?|` +
  `(?:these|those)\\s+two(?:\\s+${EVENT_CLASSIFIER})?)`;
export const HEDGED_NEVER = /\b(?:may|might|could|would|should|shall)\s+never\b/i;

// the shared-device noun in "wired/connected to the same <dev>" and
// "share a <dev>" is exactly ONE hyphenated/alphanumeric word — zero
// multi-word wildcards. A trailing exception or time clause ("same switch
// except during testing", "same switch on weekdays") is a domain
// restriction the anchored template must fail, never part of the device
// name: no function/scope word can be smuggled past a single-word capture.
const NP = `[a-z0-9-]+`;

// partial-quantity vocabulary inside a scope phrase — a scope carrying any
// of these is a restriction (some/most/a fraction of the population), never
// a population partition
const PARTIAL_SCOPE_WORDS =
  /\b(?:some|most|many|few|several|nearly|almost|half|quarter|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|subset|portion|majority|minority|fraction|percentage|certain|various|select|part|number|couple|handful|sliver|share)\b/i;

const SCOPE_FUNC_WORDS = new Set(
  'the a an of and or in on at for to by with within across during under over per'
    .split(' ')
);

// Stratum classifiers — a quantified scope may range over a partition
// dimension ("within each cohort", "for each panel type") instead of the
// population noun itself. The claim is still universal; what fails is a
// RESTRICTED domain ("every intact panel", "every panel except ...").
const STRATUM_WORDS = new Set([
  ...POP_NOUN_LIST.split('|'),
  'type', 'types', 'kind', 'kinds', 'stratum', 'strata',
  'category', 'categories', 'variety', 'varieties', 'lane', 'lanes',
]);

// Partition-dimension vocabulary ("each panel type", "on both strata",
// "for each cohort", "within either cohort") quantifies over the declared
// hypothesis partition itself — universal by construction, no passage
// grounding needed. 'cohort' is the corpus's canonical partition word
// (every shipped independence license phrases it so). The other
// POP_NOUN_LIST members are CONCRETE population nouns instead: a scope
// head built from one ("every clinic", "for all crews") is a foreign
// domain unless the passage's own non-license sentences establish it —
// a noun the text never mentions cannot silently partition the queried
// population.
const GENERIC_STRATUM = new Set([
  'type', 'types', 'kind', 'kinds', 'stratum', 'strata',
  'category', 'categories', 'variety', 'varieties', 'lane', 'lanes',
  'cohort',
]);

// Regular-plural singularization shared by extraction and scope matching.
// Ambiguous -es forms prefer the simple -s strip ("cases" -> "case"), so a
// missed form fails closed rather than grounding loosely.
const singularOf = (w) => {
  if (w.length > 3 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 3 && /(?:x|ch|sh|ss)es$|[^aeiou]zes$/.test(w)) return w.slice(0, -2);
  if (w.length > 1 && w.endsWith('s') && !w.endsWith('ss')) return w.slice(0, -1);
  return w;
};

const isPopNounTok = (tok, popSing) =>
  !!popSing && (tok === popSing || singularOf(tok) === popSing);

// ---------------------------------------------------------------------------
// exact population-noun grounding (R6). A quantified scope noun phrase
// "<Q> <N>" — "on every <N>", "for all <N>", "every <N>'s" — is a population
// universal ONLY when <N> is, with zero wildcards:
//   (a) the bare queried-population noun itself        (panel, panels)
//   (b) <grounded ctx label> <popnoun>                 (night-shift panels)
//   (c) <every declared hyp name> <popnoun>            (Ash and Fir panels)
//   (d) [<popnoun>] <stratum classifier>               (each cohort, every
//       panel type — universal over a partition dimension)
// Anything else — adjectives ("intact"), prepositional restrictions ("with
// a common relay"), exceptions ("except damaged"), or a foreign head noun
// ("prototype") — is a restricted domain and fails closed.
// ---------------------------------------------------------------------------
const npToks = (np) =>
  np
    .toLowerCase()
    .split(/[\s-]+/)
    .map((t) => t.replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, '').replace(/'s$|'$/, ''))
    .filter(Boolean);

const ctxModOk = (mods, info) =>
  info.ctxToks.some((c) => c.length === mods.length && c.every((t, i) => t === mods[i]));

const hypListOk = (mods, info) => {
  if (info.hypLc.size === 0) return false;
  const toks = mods[0] === 'both' || mods[0] === 'all' ? mods.slice(1) : mods;
  const seen = new Set();
  for (const t of toks) {
    if (t === 'and' || t === 'or') continue;
    if (!info.hypLc.has(t)) return false;
    seen.add(t);
  }
  return seen.size === info.hypLc.size;
};

function popNPOk(np, info, possessive = false) {
  const toks = npToks(np);
  if (toks.length === 0) return false;
  const last = toks[toks.length - 1];
  const mods = toks.slice(0, -1);
  const headSing = singularOf(last);
  // The population head matches the consensus noun, or — when the
  // passage left the consensus split across heads (popSing null) — one
  // of the candidate heads its own PRIOR/LIK/OBS/query sentences
  // established ("cards"/"card", "fare cards" -> "card").
  const popTokOk = (tok) =>
    isPopNounTok(tok, info.popSing) ||
    (info.popSing == null && (info.popCands?.has(singularOf(tok)) ?? false));
  if (popTokOk(last)) {
    if (mods.length === 0) return true; // (a) bare population noun
    if (ctxModOk(mods, info)) return true; // (b) ctx-grounded modifier
    if (hypListOk(mods, info)) return true; // (c) every exhaustive hypothesis named
    // (c2) the full compound population phrase itself, as established
    // by a population-defining sentence ("fare card" from "Every fare
    // card is either ..."): its modifier is part of the population
    // noun, never a restriction. A hypothesis or event token can
    // never be a compound modifier here — "Every Alpha card ..." is a
    // hypothesis-scoped head and stays refused.
    const phraseKey = [...mods, headSing].join(' ');
    return (
      (info.popPhrases?.has(phraseKey) ?? false) &&
      !mods.some((t) => (info.hypLc?.has(t) ?? false) || (info.evLc?.has(t) ?? false))
    );
  }
  // Bare-noun grounding with no consensus head: a single unmodified
  // item noun established by the passage OUTSIDE the license sentence
  // (nounGround: material/query sentences, never FLUFF, never the
  // license itself) still grounds the quantified NP. A foreign noun
  // the passage never establishes, a hypothesis/event token, or any
  // modified NP fails closed; so does every noun once a different
  // consensus head (popSing) exists.
  if (
    mods.length === 0 &&
    info.popSing == null &&
    ((info.nounGround?.has(last) ?? false) || (info.nounGround?.has(headSing) ?? false)) &&
    !(info.hypLc?.has(last) ?? false) && !(info.hypLc?.has(headSing) ?? false) &&
    !(info.evLc?.has(last) ?? false) && !(info.evLc?.has(headSing) ?? false)
  ) {
    return true;
  }
  // (d) stratum classifiers — scope position ONLY ("for each cohort",
  // "within either lane", "each panel type"). A possessive subject is a
  // concrete owner of the linked events, never a partition dimension:
  // "every <N>'s lamp and buzzer" quantifies over <N>-owned units, so <N>
  // must be the queried population noun itself (fail closed otherwise).
  if (!possessive && STRATUM_WORDS.has(last)) {
    // generic partition vocabulary is always a partition-dimension
    // universal; a concrete population noun must appear somewhere in the
    // passage OUTSIDE the license sentence ("on every clinic" over a text
    // that never mentions clinics is a foreign domain — fail closed)
    const stratumOk =
      GENERIC_STRATUM.has(last) ||
      info.nounGround.has(last) ||
      info.nounGround.has(singularOf(last));
    if (!stratumOk) return false;
    if (mods.length === 0) return true; // "each cohort", "within either lane"
    return mods.length === 1 && popTokOk(mods[0]); // "each panel type"
  }
  return false;
}

// Is `phrase` a declared-population scope — "the night-shift", "South",
// "every panel", "both Ash and Fir panels" — rather than a restriction?
// Quantified phrases delegate to popNPOk; a phrase naming every declared
// hypothesis must still end in the population noun; non-quantified phrases
// must consist ONLY of ctxLabels population-label tokens plus function
// words (and, when grounded, the population noun itself).
function licenseScopeOk(phrase, info) {
  // "On each panel separately, ..." — 'separately' reinforces the
  // per-stratum universal; it is not part of the population noun phrase.
  const hadSeparately = /\bseparately\s*$/i.test(phrase.trim());
  const p = phrase.trim().replace(/\s+separately$/i, '').replace(/[.!?,;:]+$/, '').trim();
  if (!p || /\d/.test(p) || PARTIAL_SCOPE_WORDS.test(p)) return false;
  const bare = p.replace(/^the\s+/i, '');
  const q = bare.match(/^(?:each|every|either|all|both)\s+(.+)$/i);
  if (q) {
    if (popNPOk(q[1], info)) return true;
    // Per-item 'separately' scope over ONE bare noun ("On each card
    // separately, ...", "Within each press separately, ..."): the exact
    // population-noun ground can miss when the passage's own grounding
    // sentences name the population with a synonym or leave it split
    // across heads (popSing null/conflicting). The scope still grounds
    // when that bare noun is established by the passage OUTSIDE the
    // license sentence (nounGround) or is a generic partition word —
    // never for a modified/restricted NP ("each Alpha panel separately"
    // is a subset, not a universal) and never without 'separately'.
    if (!hadSeparately) return false;
    const toks = npToks(q[1]);
    if (toks.length !== 1) return false;
    const tok = toks[0];
    if (info.hypLc?.has(tok) || info.evLc?.has(tok)) return false;
    return GENERIC_STRATUM.has(tok) ||
      (info.popCands?.has(singularOf(tok)) ?? false) ||
      (info.nounGround?.has(tok) ?? false) ||
      (info.nounGround?.has(singularOf(tok)) ?? false);
  }
  if (info.hypLc.size > 0 && info.hypNames.every((n) => wordRe(n).test(bare))) {
    return popNPOk(bare, info);
  }
  // 'x ' prefix so a leading capitalized name is non-initial for ctxLabels
  const labels = ctxLabels(`x ${bare}`, info.exclLc);
  if (labels.size === 0) return false;
  const labelTok = new Set();
  for (const l of labels) {
    for (const t of l.split(/\s+/)) {
      labelTok.add(t);
      if (t.endsWith('s')) labelTok.add(t.slice(0, -1));
    }
  }
  return bare
    .toLowerCase()
    .split(/[\s-]+/)
    .filter(Boolean)
    .every(
      (t) =>
        SCOPE_FUNC_WORDS.has(t) ||
        labelTok.has(t) ||
        (t.endsWith('s') && labelTok.has(t.slice(0, -1))) ||
        isPopNounTok(t, info.popSing)
    );
}

// optional scope phrases around the universal core — captured, then
// validated by licenseScopeOk. Leading requires a comma ("On the
// night-shift, ..."); trailing must close the sentence.
const LEAD_SCOPE = `(?:(?:on|in|for|during|within|across|under)\\s+([^;]+?),\\s*)?`;
const TAIL_SCOPE = `(?:\\s+(?:on|in|for|during|within|across|under)\\s+([^;.]+?))?[.!?]?`;

const nameSlot = (n) => `(?:the\\s+)?${escRe(n).replace(/[\s-]+/g, '[\\s-]+')}`;

// Anchored canonical cores for a DET_LINK relation. <A>/<B> are the declared
// event names (either order for symmetric forms); the subject also admits a
// plural anaphor resolved by referencesEvents.
function linkCoreSources(rel, nameA, nameB) {
  const A = nameA ? nameSlot(nameA) : null;
  const B = nameB ? nameSlot(nameB) : null;
  const both = (f) => (A && B ? [f(A, B), f(B, A)] : []);
  const namedPair = both((x, y) => `(?:both\\s+)?${x}\\s+and\\s+${y}`);
  const SUBJ = `(?:${[...namedPair, ANA_SUBJ].join('|')})`;
  // present-tense copula ONLY — a past-tense clause ("the lamp and buzzer
  // were wired to the same switch") asserts historical wiring and leaves the
  // present observation unspecified (Bulma R7 blocker).
  const cop = `(?:are|is|remains?|stays?)`;
  switch (rel) {
    case 'EQUIV':
      return [
        `${SUBJ}\\s+(?:${LINK_ADVERB}\\s+)?always\\s+${PRESENT_VERB}\\s+together`,
        `${SUBJ}\\s+(?:${LINK_ADVERB}\\s+)?always\\s+${PRESENT_VERB}\\s+in\\s+(?:tandem|unison|lockstep)`,
        `${SUBJ}\\s+${cop}\\s+(?:hard-?)?wired\\s+to\\s+the\\s+same\\s+${NP}`,
        `${SUBJ}\\s+${cop}\\s+(?:hard-?)?wired\\s+together`,
        `${SUBJ}\\s+${cop}\\s+connected\\s+to\\s+the\\s+same\\s+${NP}`,
        `${SUBJ}\\s+shares?\\s+(?:a|an|one|the\\s+same|a\\s+single)\\s+${NP}`,
        `${SUBJ}\\s+${cop}\\s+(?:(?:mechanically|electrically|physically|tightly)\\s+)?` +
          `(?:coupled|ganged|slaved|yoked|linked|chained|daisy-?chained|synchroni[sz]ed|synced)` +
          `(?:\\s+(?:together|to\\s+each\\s+other|to\\s+one\\s+another|in\\s+lockstep|in\\s+tandem|in\\s+unison))?`,
        `${SUBJ}\\s+(?:are|is)\\s+locked\\s+(?:together|in\\s+lockstep|in\\s+tandem|in\\s+unison)`,
        `${SUBJ}\\s+${PRESENT_VERB}\\s+in\\s+(?:lockstep|tandem|unison)`,
        `${SUBJ}\\s+(?:are|is)\\s+(?:always\\s+)?(?:identical|equivalent|the\\s+same)`,
        `${SUBJ}\\s+${PRESENT_VERB}\\s+if\\s+and\\s+only\\s+if\\s+the\\s+other\\s+${PRESENT_VERB}`,
        ...both((x, y) => `${x}\\s+${PRESENT_VERB}\\s+if\\s+and\\s+only\\s+if\\s+${y}\\s+${PRESENT_VERB}`),
        // possessive quantified subject ("every panel's lamp and buzzer are
        // wired to the same switch") — the noun phrase is CAPTURED and must
        // ground to the queried population exactly (popNPOk); an arbitrary
        // noun ("every prototype's") fails closed downstream.
        ...both(
          (x, y) =>
            `(?:every|each)\\s+([a-zA-Z][\\w'-]*(?:[\\s-]+[a-zA-Z][\\w'-]*)*?)(?:'s|')\\s+` +
            `${x}\\s+and\\s+${y}\\s+(?:are|is)\\s+` +
            `(?:hard-?)?wired\\s+to\\s+the\\s+same\\s+${NP}`
        ),
      ];
    case 'IMPLIES':
      return [
        ...both((x, y) => `whenever\\s+${x}\\s+${PRESENT_VERB}\\s*,?\\s*${y}\\s+${PRESENT_VERB}`),
        ...both((x, y) => `${x}\\s+${PRESENT_VERB}\\s+whenever\\s+${y}\\s+${PRESENT_VERB}`),
        ...both(
          (x, y) => `if\\s+${x}\\s+${PRESENT_VERB}\\s*,?\\s*(?:then\\s+)?${y}\\s+${PRESENT_VERB}`
        ),
        ...(A && B
          ? [
              `${A}\\s+(?:always\\s+)?` +
                `(?:triggers?|forces?|causes?|guarantees?|implies|activates|drives|sparks|induces|compels|ignites)` +
                `\\s+${B}(?:\\s+to\\s+${VERB})?`,
            ]
          : []),
        // Universal relative-clause implication, DIRECTIONAL (declared a is
        // the relative-clause antecedent, b the forced consequent):
        // "Every card in this box that bears a star has a notch as well.",
        // "Every unit that has a star also has a notch.", "All cards that
        // bear a star have a notch." Either verb slot is any present-tense
        // verb (has/have, bears/bear, carries, shows, displays, exhibits,
        // contains, ...). The quantified item NP is CAPTURED and must
        // ground to the queried population exactly (popNPOk in
        // canonicalLicenseMatch), so "Every Alpha card that ..." and
        // other restricted heads fail closed.
        ...(nameA && nameB
          ? [
              `(?:every|each|any)\\s+([a-zA-Z][\\w'-]*(?:[\\s-]+[a-zA-Z][\\w'-]*)*?)` +
                `(?:\\s+(?:in|on|from|of)\\s+(?:this|the)\\s+[a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?)?` +
                `\\s+(?:that|which)\\s+${PRESENT_VERB}\\s+${inflSlot(nameA)}` +
                `\\s+(?:also\\s+)?${PRESENT_VERB}\\s+${inflSlot(nameB)}` +
                `(?:\\s+(?:as\\s+well|too))?`,
              `(?:all)\\s+(?:the\\s+)?([a-zA-Z][\\w'-]*(?:[\\s-]+[a-zA-Z][\\w'-]*)*?)` +
                `(?:\\s+(?:in|on|from|of)\\s+(?:this|the)\\s+[a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?)?` +
                `\\s+(?:that|which)\\s+${PRESENT_VERB}\\s+${inflSlot(nameA)}` +
                `\\s+(?:also\\s+)?${PRESENT_VERB}\\s+${inflSlot(nameB)}` +
                `(?:\\s+(?:as\\s+well|too))?`,
            ]
          : []),
      ];
    case 'MUTEX':
      return [
        `${SUBJ}\\s+never\\s+(?:both\\s+)?${PRESENT_VERB}(?:\\s+together)?`,
        `${SUBJ}\\s+(?:can|will|must)\\s+never\\s+(?:both\\s+)?${VERB}(?:\\s+together)?`,
        `${SUBJ}\\s+(?:cannot|can\\s+not|can['’]t|will\\s+not|won['’]t|` +
          `must\\s+not|mustn['’]t|do\\s+not|don['’]t|does\\s+not|doesn['’]t)` +
          `\\s+(?:both\\s+)?${VERB}(?:\\s+together)?`,
        `${SUBJ}\\s+${cop}\\s+(?:mutually\\s+)?(?:exclusive|incompatible|disjoint)`,
        `${SUBJ}\\s+(?:are|is)\\s+mutex`,
        `${SUBJ}\\s+(?:precludes?|rules?\\s+out|excludes?)\\s+(?:each\\s+other|one\\s+another)`,
        ...both((x, y) => `at\\s+most\\s+one\\s+of\\s+${x}\\s+and\\s+${y}\\s+${PRESENT_VERB}`),
        ...both(
          (x, y) => `only\\s+one\\s+of\\s+${x}\\s+and\\s+${y}\\s+(?:${PRESENT_VERB}\\s+)?at\\s+a\\s+time`
        ),
        ...both(
          (x, y) =>
            `either\\s+${x}\\s+or\\s+${y}\\s+${PRESENT_VERB}\\s*,?\\s*` +
            `(?:but\\s+never\\s+both|but\\s+not\\s+both|not\\s+both)`
        ),
      ];
    default:
      return [];
  }
}

// Inflected name slot for independence cores: an optional article plus any
// conservative singular/plural inflection of the declared event name.
const inflSlot = (n) =>
  `(?:(?:the|a|an)\\s+)?(?:${inflectedNameVariants(n)
    .slice()
    .sort((a, b) => b.length - a.length)
    .map((v) => escRe(v).replace(/[\s-]+/g, '[\\s-]+'))
    .join('|')})`;

// Anchored canonical cores for COND_INDEP — the declared events as an
// "A and B (and C...)" list subject or a resolved plural anaphor.
function indepCoreSources(evNames) {
  const items = evNames.map(nameSlot);
  const ITEM = `(?:${items.join('|')})`;
  const LIST = items.length
    ? `(?:${ITEM}(?:(?:\\s*,\\s*|\\s+and\\s+|\\s*,\\s*and\\s+)${ITEM})+|${ANA_SUBJ})`
    : ANA_SUBJ;
  const cop = `(?:are|is|remains?|stays?)`;
  const cores = [
    `${LIST}\\s+${cop}\\s+(?:(?:conditionally|statistically|mutually|fully|truly|completely|` +
      `entirely|wholly|genuinely|perfectly|functionally)\\s+)?independent` +
      `(?:\\s+of\\s+(?:each\\s+other|one\\s+another))?`,
    `${LIST}\\s+(?:are|is|have\\s+been|has\\s+been|get)\\s+${VERB}\\s+independently`,
    `${LIST}\\s+${PRESENT_VERB}\\s+independently(?:\\s+of\\s+(?:each\\s+other|one\\s+another))?`,
    `${LIST}\\s+(?:are|is)\\s+separately\\s+administered`,
    `${LIST}\\s+(?:are|is)\\s+uncorrelated`,
  ];
  // Equal-conditional-rate definition cores, per ordered pair (A,B): B is
  // the event whose rate/probability is predicated, A the conditioner.
  // The contrasting arm must negate A ("without a <A>", "with no <A>",
  // "lacking <A>", "whether <A> occurs or not") or state an explicit
  // contrasting adjective on A's head noun ("with a bright lamp" vs
  // "with a dim lamp" — contrastingArmsOk in canonicalLicenseMatch
  // strictly requires different adjectives on the same head) — a plain
  // restatement of the same arm asserts nothing. All groups are
  // non-capturing: middle capture groups in canonicalLicenseMatch are
  // popNP-validated.
  const ITEMNP = `[a-z][a-z-]*(?:\\s+[a-z][a-z-]*){0,2}`;
  const RATE = `(?:rate|probability|likelihood|chance|frequency|proportion)`;
  const SAME = `(?:is|remains|stays)\\s+(?:the\\s+same|identical|equal|unchanged)`;
  for (const aName of evNames) {
    for (const bName of evNames) {
      if (aName === bName) continue;
      const A = inflSlot(aName);
      const B = inflSlot(bName);
      const subjB =
        `(?:(?:the\\s+)?${B}\\s+${RATE}|(?:the\\s+)?${RATE}\\s+of\\s+${B})`;
      const contra =
        `(?:without\\s+${A}|with\\s+no\\s+${A}|lacking\\s+${A}|with\\s+${A}\\s+(?:absent|missing))`;
      cores.push(
        `${subjB}\\s+${SAME}\\s+whether\\s+${A}\\s+${PRESENT_VERB}\\s+or\\s+not`,
        `${subjB}\\s+${SAME}\\s+whether\\s+or\\s+not\\s+${A}\\s+${PRESENT_VERB}`,
        `${subjB}\\s+${SAME}\\s+for\\s+${ITEMNP}\\s+with\\s+${A}\\s+as\\s+for\\s+${ITEMNP}\\s+${contra}`
      );
      // Contrasting-adjective arm: both arms are "with a <adj> <head>"
      // where <head> is A's head noun (its final token); adjective
      // difference and head identity are verified by contrastingArmsOk.
      const aToks = String(aName).trim().split(/[\s-]+/).filter(Boolean);
      const aHead = aToks[aToks.length - 1];
      if (aHead && /^[A-Za-z]+$/.test(aHead)) {
        const HEAD = `(?:${inflectedNameVariants(aHead)
          .slice()
          .sort((x, y) => y.length - x.length)
          .map((v) => escRe(v))
          .join('|')})`;
        cores.push(
          `${subjB}\\s+${SAME}\\s+for\\s+${ITEMNP}\\s+with\\s+(?:a|an|the)\\s+[a-z][a-z-]*\\s+${HEAD}\\s+as\\s+for\\s+${ITEMNP}\\s+with\\s+(?:a|an|the)\\s+[a-z][a-z-]*\\s+${HEAD}`
        );
      }
    }
  }
  return cores;
}

// Full-sentence anchored match: ^ [scope,]? core [scope]? [.]? $ — every
// clause of the containing sentence is accounted for or the match fails.
// Scope phrases are validated by licenseScopeOk; captured possessive
// subjects are validated by popNPOk — all must ground to the queried
// population (fail-closed).
function canonicalLicenseMatch(coreSources, text, info) {
  const t = text.trim().replace(/\s+/g, ' ');
  const re = new RegExp(`^${LEAD_SCOPE}(?:${coreSources.join('|')})${TAIL_SCOPE}$`, 'i');
  const m = t.match(re);
  if (!m) return false;
  if (!contrastingArmsOk(t)) return false;
  // Group layout: m[1] = leading scope; middle groups = captured possessive
  // subject NPs inside the core alternation; the LAST group = trailing scope
  // (it opens after the alternation). Scopes go through licenseScopeOk;
  // possessive subjects through popNPOk — both fail closed.
  const tail = m.length - 1;
  if (m[1] !== undefined && !licenseScopeOk(m[1], info)) return false;
  if (m[tail] !== undefined && !licenseScopeOk(m[tail], info)) return false;
  for (let i = 2; i < tail; i++) {
    if (m[i] !== undefined && !popNPOk(m[i], info, true)) return false;
  }
  return true;
}

// Does the sentence affirmatively entail the universal relation `rel` over
// the whole queried population? nameA/nameB are the declared event names.
// `scope` carries {hypNames, hypLc, exclLc, popSing, ctxToks} — the exact
// population grounding R6 requires; missing pieces normalize fail-closed.
export function assertsUniversalRelation(rel, text, nameA = null, nameB = null, scope = {}) {
  const cores = linkCoreSources(rel, nameA, nameB);
  if (cores.length === 0) return false;
  const hypNames = scope.hypNames ?? [];
  const info = {
    hypNames,
    hypLc: scope.hypLc ?? new Set(hypNames.map((n) => n.toLowerCase())),
    exclLc: scope.exclLc ?? new Set(),
    popSing: scope.popSing ?? null,
    popCands: scope.popCands ?? new Set(),
    popPhrases: scope.popPhrases ?? new Set(),
    evLc: scope.evLc ?? new Set(),
    ctxToks: scope.ctxToks ?? [],
    nounGround: scope.nounGround ?? new Set(),
  };
  return canonicalLicenseMatch(cores, text, info);
}

// Declared hypothesis names: Map<hypId, name> from HYP span text.
function declaredHypNames(X, prog) {
  const names = new Map();
  for (const d of prog.decls) {
    if (d.op === 'HYP' && d.span) {
      const n = nameFromSpan(X, d.span);
      if (n) names.set(d.id, n);
    }
  }
  return names;
}

// Hypothesis-Restriction Guard: a population-wide DET_LINK (or a
// GIVEN-* COND_INDEP) applies across EVERY hypothesis in its ctx. A license
// sentence naming a strict subset of the declared hypotheses ("always
// activate together on Ash panels") scopes the relation to that stratum —
// it cannot license the queried population. A COND_INDEP explicitly GIVEN
// one hypothesis legitimately names exactly its own stratum.
function hypothesisScopedLicense(decl, hypMap, text) {
  if (hypMap.size === 0) return null;
  const mentioned = [...hypMap.values()].filter((n) => wordRe(n).test(text));
  if (mentioned.length === 0 || mentioned.length === hypMap.size) return null;
  if (decl.op === 'COND_INDEP' && decl.hyp !== '*') {
    const own = hypMap.get(decl.hyp);
    if (own && mentioned.every((m) => m === own)) return null;
  }
  return mentioned;
}

// A DET_LINK a-IMPLIES-b span whose when/if antecedent names b asserts the
// reverse direction: "whenever B fires, A fires" licenses B→A, not A→B.
function impliesReversed(seg, d, evNameById) {
  const aNm = evNameById.get(d.a);
  const bNm = evNameById.get(d.b);
  if (!aNm || !bNm) return false;
  const m = seg.text.match(/\b(?:whenever|when|if|once|after)\b([^.;]{0,80})/i);
  if (!m) return false;
  const aIn = wordRe(aNm).test(m[1]);
  const bIn = wordRe(bNm).test(m[1]);
  return bIn && !aIn;
}

// Sentence a decl span is grounded in (the span's own sentence, or the
// sentence it overlaps most when it straddles a boundary).
function declSentence(segs, span) {
  if (!span) return null;
  const hit = segs.find((s) => span.s >= s.s && span.s < s.e);
  if (hit) return hit;
  let best = null;
  let bestOv = 0;
  for (const s of segs) {
    const ov = Math.min(span.e, s.e) - Math.max(span.s, s.s);
    if (ov > bestOv) {
      bestOv = ov;
      best = s;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Population-noun grounding (R6): the queried unit head noun, extracted from
// the sentences that DEFINE the queried population — PART_EXHAUSTIVE
// ("Every panel is either Ash or Fir", "…of cases"), PRIOR/LIK proportions
// ("40% of Ash panels"), OBS subjects ("This panel", "A night-shift panel"),
// and the QUERY subject ("this item", "the case is"). A sentence mentioning
// a declared event name in the head slot is rejected as the population head
// ("the lamp lights", "the flag test is positive" are events, not units).
// Candidates are consensus-checked: zero or conflicting heads leave the
// population ungrounded (popSing null) — scopes requiring grounding then
// fail closed.
// ---------------------------------------------------------------------------
function populationInfo(X, prog, segs, evNameById, exclLc) {
  const evNames = [...evNameById.values()].map((n) => n.toLowerCase());
  const hasEvName = (t) => evNames.some((n) => n && wordRe(n).test(t));
  const cands = new Set();
  // Full compound population phrases ("fare card") established by a
  // population-defining subject (PART_EXHAUSTIVE / OBS / query via
  // subjScan): the head alone grounds a bare quantified NP, the full
  // phrase grounds the same compound NP in a license. Rate-sentence
  // "of <mods> <plural>" phrases are NOT recorded — their modifiers
  // are hypothesis/ctx strata, not part of the population noun.
  const popPhrases = new Set();
  const pushHead = (np) => {
    if (!np) return;
    const raw = np.trim().toLowerCase();
    const toks = npToks(raw);
    if (toks.length === 0 || hasEvName(raw)) return;
    const head = singularOf(toks[toks.length - 1]);
    if (head.length < 2 || POP_STOP.has(head)) return;
    cands.add(head);
    popPhrases.add([...toks.slice(0, -1), head].join(' '));
  };
  // "<Q> <mods> <head> is either …" — head = the noun right before the copula
  const SUBJ_VERB = 'is|was|were|are|has|had|been|belongs?|falls?|comes';
  const subjScan = (text) => {
    // Random-sampling subject ("A unit drawn at random from the bin
    // has ...", "A random card from the box has ..."): the sampled
    // item noun is the population head. The generic scans below would
    // instead take the container noun before the main verb ("bin")
    // and split the consensus, so this subject grounds first and alone.
    const samp =
      text.match(
        /^\s*(?:a|an|one|the)\s+(?:randomly\s+)?([a-zA-Z][\w'-]*(?:[\s-]+[a-zA-Z][\w'-]*)*?)\s+(?:drawn|picked|chosen|selected|pulled|taken|sampled)\s+at\s+random\b/i
      ) ||
      text.match(
        /^\s*(?:a|an|one|the)\s+random\s+([a-zA-Z][\w'-]*(?:[\s-]+[a-zA-Z][\w'-]*){0,1}?)(?=\s+(?:from|has|have|is|are|was|were)\b)/i
      );
    if (samp) {
      pushHead(samp[1]);
      return;
    }
    let m = text.match(
      new RegExp(
        `\\b(?:every|each|all(?:\\s+the)?)\\s+((?:[a-zA-Z][\\w'-]*[\\s-]+)*?)([a-zA-Z][\\w'-]*)\\s+(?:${SUBJ_VERB})\\b`,
        'i'
      )
    );
    if (m) pushHead(`${m[1]}${m[2]}`);
    m = text.match(
      new RegExp(`\\bthis\\s+((?:[a-zA-Z][\\w'-]*[\\s-]+)*?[a-zA-Z][\\w'-]*?)\\s+(?:${SUBJ_VERB})\\b`, 'i')
    );
    if (m) pushHead(m[1]);
    else {
      const t = text.match(/\bthis\s+([a-zA-Z][\w'-]*)/i);
      if (t) pushHead(t[1]);
    }
    m = text.match(
      new RegExp(
        `\\b(?:a|an|the)\\s+((?:[a-zA-Z][\\w'-]*[\\s-]+)*?[a-zA-Z][\\w'-]*?)\\s+(?:${SUBJ_VERB})\\b`,
        'i'
      )
    );
    if (m) pushHead(m[1]);
  };
  // "40% of <mods> <plural>" — plural head after 'of'
  const ofScan = (text) => {
    for (const m of text.matchAll(/\bof\s+(?:[a-zA-Z][\w'-]*[\s-]+)*?([a-z][\w'-]*s)\b/gi)) {
      const head = m[1].toLowerCase();
      if (!hasEvName(head) && !POP_STOP.has(head)) cands.add(singularOf(head));
    }
  };
  for (const d of prog.decls) {
    if (!d.span) continue;
    const seg = declSentence(segs, d.span);
    if (!seg) continue;
    if (d.op === 'PART_EXHAUSTIVE') {
      subjScan(seg.text);
      ofScan(seg.text);
    } else if (d.op === 'OBS') {
      subjScan(seg.text);
    } else if (d.op === 'PRIOR' || d.op === 'LIK') {
      ofScan(seg.text);
    }
  }
  const names = entityNames(X, prog);
  for (const seg of segs) {
    if (sentenceMateriality(seg, names).kind === 'query') subjScan(seg.text);
  }
  const popSing = cands.size === 1 ? [...cands][0] : null;
  // ctx modifiers + scope-noun grounding (R7). A DET_LINK / COND_INDEP
  // sentence may never introduce the context token or partition noun it is
  // itself scoped to — "On every Refurbished panel, ..." cannot
  // self-ground 'refurbished'. ctx labels ground ONLY from the queried
  // population's own sentences (OBS witnesses and the QUERY) plus the ctx
  // ids those decls actually declare. Concrete stratum nouns ground from
  // material or query sentences outside the license and FLUFF ledger.
  const LICENSE_OPS = new Set(['DET_LINK', 'COND_INDEP']);
  const licenseSegIdx = new Set();
  for (const d of prog.decls) {
    if (LICENSE_OPS.has(d.op) && d.span) {
      const seg = declSentence(segs, d.span);
      if (seg) licenseSegIdx.add(seg.idx);
    }
  }
  const ctxToks = [];
  const addCtxLabels = (text) => {
    for (const l of ctxLabels(text, exclLc)) ctxToks.push(l.split(/\s+/));
  };
  for (const d of prog.decls) {
    if (d.op === 'OBS' && d.span) {
      const seg = declSentence(segs, d.span);
      if (seg) addCtxLabels(seg.text);
    }
  }
  for (const seg of segs) {
    if (sentenceMateriality(seg, names).kind === 'query') addCtxLabels(seg.text);
  }
  const usedCtx = new Set(
    prog.decls.filter((d) => d.op === 'OBS').map((d) => d.ctx).filter(Boolean)
  );
  if (prog.query?.ctx) usedCtx.add(prog.query.ctx);
  for (const cid of usedCtx) {
    if (cid === 'ctx0' || cid === '*') continue;
    ctxToks.push(cid.replace(/^ctx_/, '').split(/[_\s-]+/).filter(Boolean));
  }
  const nounGround = new Set();
  const fluffIdx = new Set(
    prog.decls.filter((d) => d.op === 'SENT_TAG' && d.tag === 'FLUFF').map((d) => d.idx)
  );
  for (const seg of segs) {
    if (licenseSegIdx.has(seg.idx) || fluffIdx.has(seg.idx)) continue;
    // inert filler ("The clinic is open.") grounds NOTHING — a bare noun
    // mention supplies no mapping from the queried population to foreign
    // equipment (Bulma R7: FLUFF-supplied 'clinic' self-grounded "every
    // clinic" as a partition dimension).
    if (sentenceMateriality(seg, names).kind === 'fluff') continue;
    for (const m of seg.text.toLowerCase().matchAll(/[a-z][a-z'-]*/g)) {
      const t = m[0].replace(/'s$|'$/, '');
      nounGround.add(t);
      nounGround.add(singularOf(t));
    }
  }
  return { popSing, popCands: cands, popPhrases, ctxToks, nounGround };
}

/**
 * validateLicenseDecls(X, prog) -> [{decl, reason}]
 * DET_LINK / COND_INDEP are licenses, not facts: the bound sentence must
 * (a) reference every declared event the license claims to relate — by name
 * or by a resolvable plural anaphor — and (b) express the declared relation
 * (EQUIV/IMPLIES/MUTEX wording, or affirmative independence). An unsupported
 * license ("The room was quiet", a temperature/humidity clause over lamp/
 * buzzer events) is reported for the caller to drop from the effective
 * program — the pair then stands unlicensed rather than silently licensed.
 */
export function validateLicenseDecls(X, prog) {
  const evNameById = eventNameMap(X, prog);
  const hypMap = declaredHypNames(X, prog);
  const hypNames = [...hypMap.values()];
  // ctxLabels exclusion: declared entity names are not population labels
  const exclLc = new Set(
    [...evNameById.values(), ...hypNames].map((n) => n.toLowerCase())
  );
  const segs = sentences(X);
  const doubt = findPoisoningPremiseDoubt(X, prog, segs);
  if (doubt) {
    return prog.decls
      .filter((d) => (d.op === 'DET_LINK' || d.op === 'COND_INDEP') && d.span)
      .map((decl) => ({ decl, reason: `premise doubt or retraction: "${doubt.marker}"` }));
  }
  const pop = populationInfo(X, prog, segs, evNameById, exclLc);
  const scopeInfo = {
    hypNames,
    hypLc: new Set(hypNames.map((n) => n.toLowerCase())),
    evLc: new Set(
      [...evNameById.values()].flatMap((n) => n.toLowerCase().split(/[\s-]+/)).filter(Boolean)
    ),
    exclLc,
    popSing: pop.popSing,
    popCands: pop.popCands,
    popPhrases: pop.popPhrases,
    ctxToks: pop.ctxToks,
    nounGround: pop.nounGround,
  };
  const invalid = [];
  for (const d of prog.decls) {
    if (d.op === 'DET_LINK' && d.span) {
      const seg = declSentence(segs, d.span);
      const ref = seg ? referencesEvents(X, seg, [d.a, d.b], evNameById) : { ok: false };
      if (!ref.ok) {
        invalid.push({
          decl: d,
          reason: `DET_LINK ${d.a} ${d.rel} ${d.b} span [${d.span.s},${d.span.e}] does not reference the linked events`,
        });
        continue;
      }
      // Hypothesis-Restriction Guard over the WHOLE containing sentence: a
      // population-wide link cannot be scoped to a strict subset of the
      // exhaustive hypotheses.
      const scoped = hypothesisScopedLicense(d, hypMap, seg.text);
      if (scoped) {
        invalid.push({
          decl: d,
          reason:
            `DET_LINK ${d.a} ${d.rel} ${d.b} span's sentence scopes the relation ` +
            `to hypothesis {${scoped.join(',')}} only — a hypothesis-restricted ` +
            `link is not population-wide (HYPOTHESIS_SCOPED_LICENSE)`,
        });
        continue;
      }
      // Canonical-template invariant: the FULL containing sentence must
      // match an anchored ^...$ universal form (optional declared scope
      // phrase, zero other clauses/modifiers) — an outer modal, subset or
      // extra clause fails structurally, never by deny-list.
      if (
        !assertsUniversalRelation(
          d.rel,
          seg.text,
          evNameById.get(d.a),
          evNameById.get(d.b),
          scopeInfo
        )
      ) {
        invalid.push({
          decl: d,
          reason:
            `DET_LINK ${d.a} ${d.rel} ${d.b} span's sentence is not a canonical ` +
            `population-wide ${d.rel} form (anchored full-sentence match) — ` +
            `cannot license ${d.rel}`,
        });
        continue;
      }
      // polarity enforcement: a positive relation (EQUIV co-activation, or
      // IMPLIES one-way forcing) may not be declared over text asserting
      // negation, exclusion or merely occasional co-occurrence.
      if ((d.rel === 'EQUIV' || d.rel === 'IMPLIES') && POSITIVE_LINK_DENY.test(seg.text)) {
        invalid.push({
          decl: d,
          reason:
            `DET_LINK ${d.a} ${d.rel} ${d.b} span's sentence asserts exclusion, ` +
            `negation or non-deterministic co-occurrence — cannot license ${d.rel}`,
        });
        continue;
      }
      if (d.rel === 'IMPLIES' && impliesReversed(seg, d, evNameById)) {
        invalid.push({
          decl: d,
          reason: `DET_LINK ${d.a} IMPLIES ${d.b} direction contradicts the clause`,
        });
      }
    }
    if (d.op === 'COND_INDEP' && d.span) {
      const seg = declSentence(segs, d.span);
      const ref = seg ? referencesEvents(X, seg, d.evs, evNameById) : { ok: false };
      if (!ref.ok) {
        invalid.push({
          decl: d,
          reason: `COND_INDEP (${d.evs.join(' ')}) span [${d.span.s},${d.span.e}] does not reference the declared events`,
        });
        continue;
      }
      if (!assertsIndependence(seg.text)) {
        invalid.push({
          decl: d,
          reason: `COND_INDEP (${d.evs.join(' ')}) span's sentence does not affirmatively assert independence`,
        });
        continue;
      }
      const scoped = hypothesisScopedLicense(d, hypMap, seg.text);
      if (scoped) {
        invalid.push({
          decl: d,
          reason:
            `COND_INDEP (${d.evs.join(' ')}) span's sentence scopes independence ` +
            `to hypothesis {${scoped.join(',')}} only — a hypothesis-restricted ` +
            `license is not population-wide (HYPOTHESIS_SCOPED_LICENSE)`,
        });
        continue;
      }
      if (
        !canonicalLicenseMatch(
          indepCoreSources(
            d.evs.map((e) => evNameById.get(e)).filter(Boolean)
          ),
          seg.text,
          scopeInfo
        )
      ) {
        invalid.push({
          decl: d,
          reason:
            `COND_INDEP (${d.evs.join(' ')}) span's sentence is not a canonical ` +
            `population-wide independence form (anchored full-sentence match)`,
        });
      }
    }
  }
  return invalid;
}

/**
 * claimsRelation(X, prog, evPair) -> seg | null
 * The first sentence of X that (a) references both events of `evPair` — by
 * name or anaphora — and (b) carries relation/dependency wording. Presence
 * of such a sentence means the source raises the license question the
 * program must answer; silent text falls through to the credal interval.
 */
export function claimsRelation(X, prog, evPair) {
  const evNameById = eventNameMap(X, prog);
  for (const seg of sentences(X)) {
    if (!DEPENDENCY_WORDS.test(seg.text)) continue;
    if (referencesEvents(X, seg, evPair, evNameById).ok) return seg;
  }
  return null;
}

// Ops that ground a sentence as accounted. SENT_TAG itself does not count —
// it is the tag, not the grounding. NON_ASSERTED does count: it is an
// explicit ledger action over the span.
const ACCOUNTING_OPS = new Set([
  'HYP', 'PART_EXHAUSTIVE', 'EVENT', 'PRIOR', 'LIK', 'POSTERIOR_STATED',
  'JOINT_LIK', 'OBS', 'COND_INDEP', 'DET_LINK', 'CONFLICT', 'NON_ASSERTED',
]);

// --- residual sub-clause accounting (Bulma R4 decisive replay) -----------
// Coverage is per CLAUSE, not per sentence container: a grounded decl
// whose span sits in one clause of a compound sentence does not account
// for a sister clause sharing that sentence. Joining an ungrounded
// counterexample to a grounded observation with ';' instead of '.'
// changed REFUSED into 1/1 — the observation span covered only its own
// clause, yet sentence-level overlap ledgered the whole segment. Every
// accounted sentence is therefore split on clause breaks and any
// residual sub-clause no accounting span touches must carry its own
// positive inert evidence (static scene clause, or a verbless
// observation-side scene/time/location adjunct) or the sentence refuses.
// `sentences()` itself is unchanged, so every SENT_TAG span and index
// in existing programs keeps its meaning.
const hasStrictFiniteVerb = (text) =>
  FINITE_VERB_TOK.test(text) || /\b[a-z]{4,}ed\b/.test(text);

const GENERIC_VERB_STOP = new Set(
  ('this its his hers ours theirs yours across sometimes always towards besides thus yes plus versus').split(' ')
);
const GENERIC_PREP_TOK = new Set(
  ('among of in on at by for with from to under within across during between against per via over after before into onto upon about').split(' ')
);
// A coordinated fragment counts as an independent clause only when it
// carries a finite predicate. In NON-final position the closed finite
// list plus generic lower-case '-ed'/'-s' shapes prove one. In FINAL
// position only unambiguous finite forms count (Bulma R5 F04): a clause
// can END on its verb ("Yesterday only the lamp lit" — the earlier
// `i < toks.length - 1` scan skipped that final verb, fused the two
// clauses, and released the counterexample), but a fragment-final
// generic '-s' token is the noun the coordination attaches to, never a
// predicate: a plural ("... of Reef tanks"), a capitalized declared
// name ("... and Iris population" — generic '-s' verbs are therefore
// lowercase-only), or a verb-homograph event noun ending a shared
// subject ("both the pressure light and the flow siren read ...").
// Final-position proof is therefore restricted to auxiliaries/copula,
// irregular past forms and '-ed' past forms; homograph noun forms
// (light/lights, sound/sounds, fire/fires, runs, stands, ...) never
// count there.
const CLAUSE_FINAL_FINITE = new Set(
  ('is are was were has have had does do did can could will would shall should may might must ' +
   'ran stood sat hung lit fired activated occurred held applied ' +
   'rang went came fell kept blew shone flashed sounded').split(' ')
);
// Declared entity names masked behind a neutral placeholder, longest
// first, so a participial adjective or gerund INSIDE a declared event
// name ("cracked seal", "stamped lid", "flashing beacon") is never
// mistaken for a clause predicate — the same masking hasOwnPredicate
// already applies before its predicate scan.
function maskEntityNames(text, names = []) {
  let masked = text;
  const ordered = [...(names ?? [])]
    .filter((n) => typeof n === 'string' && n.trim())
    .sort((a, b) => b.length - a.length);
  for (const n of ordered) {
    masked = masked.replace(inflectedNameRe(n, 'gi'), ' @ ');
  }
  return masked;
}
// Articles/determiners: a generic '-ed' token immediately after one of
// these, before a following noun, is a prenominal participial modifier
// ("a cracked lid", "the sealed box"), never a finite verb — a finite
// verb follows its subject noun ("the seal cracked"), never "a"/"the".
const PARTICIPIAL_DET = new Set([
  'a', 'an', 'the', 'this', 'that', 'its', 'their', 'his', 'her',
  'both', 'each', 'every',
]);
function hasFiniteVerb(text, names = []) {
  const masked = names && names.length ? maskEntityNames(text, names) : text;
  const toks = masked.match(/@|[A-Za-z][a-z-]*/g) ?? [];
  for (let i = 0; i < toks.length; i++) {
    const raw = toks[i];
    const t = raw.toLowerCase();
    // a hyphenated token is a compound noun/adjective ("lamp-lit
    // capsules"), never the finite verb its tail resembles
    if (t.includes('-')) continue;
    if (i === toks.length - 1) {
      if (CLAUSE_FINAL_FINITE.has(t)) return true;
      if (raw === t && /^[a-z]{4,}ed$/.test(t)) return true;
      continue;
    }
    if (FINITE_VERB_TOK.test(t)) return true;
    if (raw !== t) continue;
    if (/^[a-z]{4,}ed$/.test(t)) {
      // prenominal participial modifier ("a cracked lid"): a non-final
      // '-ed' token immediately after an article/determiner modifies
      // the following noun; it is not the clause's finite verb
      const prevEd = i > 0 ? toks[i - 1].toLowerCase() : '';
      if (PARTICIPIAL_DET.has(prevEd)) continue;
      return true;
    }
    // a generic '-s' shape directly after a preposition is the plural
    // noun that preposition governs ("Among invoices with ..."), not
    // a third-person verb
    const prev = i > 0 ? toks[i - 1].toLowerCase() : '';
    if (GENERIC_PREP_TOK.has(prev)) continue;
    if (t.length >= 3 && t.endsWith('s') && !t.endsWith('ss') &&
        !t.endsWith('us') && !t.endsWith('ies') && !GENERIC_VERB_STOP.has(t)) {
      return true;
    }
  }
  return false;
}

// Nonfinite predicates assert an event state without a finite verb:
// gerund/participle clauses ("the buzzer being silent", "its pump
// remaining idle") are propositions, not noun phrases. Time/thing
// '-ing' nouns that frame a clause without asserting one are excluded.
const NONFINITE_STOP = new Set(
  // 'during' and 'according' (in 'according to') are grammaticalized
  // prepositions, never gerund predicates — a source attribution
  // ("... 20% of Pearl consoles, according to the technician's
  // record", Bulma P7C1) frames the rate it follows
  ('during according morning evening something anything nothing everything thing reading readings spring').split(' ')
);
// Closed-list finite forms and '-ed' past forms only (no generic '-s'
// shape): strong predicate evidence for a complete clause.
function hasStrongPredicate(text) {
  const toks = text.match(/[A-Za-z][a-z-]*/g) ?? [];
  for (const raw of toks) {
    const t = raw.toLowerCase();
    if (t.includes('-')) continue;
    if (FINITE_VERB_TOK.test(t)) return true;
    if (raw === t && /^[a-z]{4,}ed$/.test(t)) return true;
  }
  return false;
}

function hasNonfinitePredicate(text) {
  for (const m of text.toLowerCase().matchAll(/\b[a-z]{2,}ing\b/g)) {
    if (!NONFINITE_STOP.has(m[0])) return true;
  }
  return false;
}

const PRED_TAIL_STOP = new Set(
  ('the a an and or of on in at by for with to from its this that these those throughout during near beside behind').split(' ')
);
// Does a fragment assert its own proposition? A finite or nonfinite
// predicate proves it; so does a verbless (absolute) predication over a
// declared entity — entity mention followed by predicate content that
// is not just determiners/prepositions ("its buzzer silent throughout
// the check", "the pump idle"). A bare noun-phrase fragment ("the
// buzzer") asserts nothing.
function isAssertiveFragment(text, names = []) {
  // declared-name masking: a gerund/participle inside a declared event
  // name ("a flashing beacon", "a cracked seal") is noun material, not
  // a finite/nonfinite predicate; one outside the name ("the beacon
  // flashing alone") still asserts
  if (hasFiniteVerb(text, names) ||
      hasNonfinitePredicate(maskEntityNames(text, names))) return true;
  for (const n of names) {
    const m = wordRe(n).exec(text);
    if (!m) continue;
    const tail = text.slice(m.index + m[0].length).toLowerCase();
    const toks = tail.match(/[a-z][a-z-]*/g) ?? [];
    if (toks.some((t) => t.length >= 3 && !PRED_TAIL_STOP.has(t))) return true;
  }
  return false;
}

// Verbless prepositional scene/time/location adjunct: "At inspection",
// "During routine rounds", "In the morning check", "On the night-shift:".
// It frames the grounded clause it shares a sentence with and asserts no
// proposition of its own — no predicate, no declared entity, no doubt
// marker, no probability literal, no dependency wording.
export function isInertAdjunctClause(raw, names) {
  {
    const words = String(raw || '').trim().split(/\s+/);
    if (/^including$/i.test(words[0] || '') && !containsDeclaredName(raw, names)
      && !premiseDoubt(raw) && !probLikeLiteral(raw) && !/\d/.test(raw)
      && !hasRateCoactMarker(raw) && !rateInsetFiniteVerbs(raw).length
      && !/\b(activ\w*|occur\w*|firing)\b/i.test(raw)) {
      for (let k = 2; k < words.length; k++) {
        const nominal = words.slice(1, k).join(' ');
        if (/^(?:(?:the|a|an)\s+)?[a-z]+(?:\s+[a-z]+)?$/i.test(nominal)
          && isInertUncoveredInterval(words.slice(k).join(' '), names)) return true;
      }
    }
  }
  const t = raw.trim().replace(/[.!?;:,]+$/, '').trim();
  if (!t || t.length > 60) return false;
  if (!/^(?:at|in|on|during|before|after|by|within|under|across|for|near|beside|behind|inside|outside|around|above|below|between)\b/i.test(t)) return false;
  if (hasStrictFiniteVerb(t) || premiseDoubt(t) || probLikeLiteral(t)) return false;
  if (DEPENDENCY_WORDS.test(t)) return false;
  return !names.some((n) => wordRe(n).test(t));
}

// Strong clause breaks always separate independent clauses. The colon
// form requires a following space, so digit ratios ("3:1") and times
// never split; the en-dash form requires surrounding spaces, so numeric
// ranges never split. Bare coordinators (" and ", " but ", ...) split
// only when a finite predicate stands on both sides — noun-phrase
// coordination ("the H0 and H1 categories", "lamp and buzzer ...") is
// one clause, never two.
const STRONG_BREAK =
  /;|—|--|\s+–\s+|:(?=\s)|\(|\)|\[|\]|,\s+(?:while|whereas|though|although|but|yet|and)\s+/gi;
const BARE_COORD = /\s+(?:and|but|yet|while|whereas|though|although)\s+/gi;
// Round-5 clause classes the strong/bare breaks missed. (1) A
// parenthetical "(...)" or "[...]" is a strong break: its content is a
// separate assertion slot, excused only by the same positive inert
// evidence as any residual clause. (2) A comma after a COMPLETE main
// clause followed by an assertive fragment is an absolute/nonfinite
// clause boundary ("..., its buzzer silent throughout the check");
// conversely a leading subordinate fragment closed by a comma is its
// own clause. A comma after a verbless scope adjunct ("On the
// selected item, ...", "Given category X, ...") is neither — the left
// side has no predicate and asserts no subordinate proposition. (3) A
// bare subordinate introducer after a complete main clause ("... is
// lit despite the buzzer being silent", "... with its pump idle")
// opens a subordinate assertive clause of its own.
const COMMA_BREAK = /,\s+/gi;
// Bulma R7: concessive/temporal subordinators that can attach a
// counterexample clause to a PRIOR/LIK sentence ("... 20% of Pearl
// consoles even as both devices fired simultaneously", "... for 60%
// despite a documented case in which the lamp lit alone") split like
// any other subordinate break, so the appended clause must carry its
// own grounding. "after a/an" consumes its article ("after a lone
// display") while a bare temporal adjunct ("after lunch") stays part
// of its host clause; "with/without" keep their absolute guard below.
const SUBORD_BREAK =
  /\s+(?:even\s+as|despite|in\s+spite\s+of|notwithstanding|save\s+that|except\s+that|although|even\s+though|even\s+if|though|whereas|while|because|unless|following|after\s+(?:a|an)|with|without)\s+/gi;
const SUBORD_LEAD =
  /^\s*(?:even\s+as|despite|in\s+spite\s+of|notwithstanding|save\s+that|except\s+that|although|even\s+though|even\s+if|though|while|whereas|because|since|unless|following|after|before)\b/i;
// A conditional lead ("when the case is X, ...", "if ..., ...") frames
// the main clause that follows — it states the condition the main
// premise is grounded under, not a separate asserted occurrence — so
// its closing comma never opens a residual clause. Concessive leads
// (despite/although/...) are factive and are NOT exempted.
const CONDITIONAL_LEAD = /^\s*(?:when|whenever|if)\b/i;

export function subClauses(seg, names = []) {
  const pieces = [];
  let start = 0;
  for (const m of seg.text.matchAll(STRONG_BREAK)) {
    pieces.push({
      s: seg.s + start, e: seg.s + m.index,
      text: seg.text.slice(start, m.index), kind: 'strong',
    });
    start = m.index + m[0].length;
  }
  pieces.push({
    s: seg.s + start, e: seg.e, text: seg.text.slice(start), kind: 'strong',
  });
  const splitWhere = (piece, re, kind, accept) => {
    const parts = [];
    let pStart = 0;
    let via = piece.via ?? null;
    for (const m of piece.text.matchAll(re)) {
      const left = piece.text.slice(pStart, m.index);
      const right = piece.text.slice(m.index + m[0].length);
      if (accept(left, right, m[0])) {
        parts.push({
          s: piece.s + pStart, e: piece.s + m.index, text: left, kind, via,
        });
        via = m[0].trim().toLowerCase().replace(/\s+/g, ' ');
        pStart = m.index + m[0].length;
      }
    }
    parts.push({
      s: piece.s + pStart, e: piece.e, text: piece.text.slice(pStart),
      kind: parts.length ? kind : piece.kind, via,
    });
    return parts;
  };
  // Coordinated noun-phrase subject ("On night-shift panels, the lamp
  // and buzzer always activate together"): the coordinator joins two
  // declared event names into ONE subject whose verb follows the second
  // name. Splitting at 'and' there manufactures a phantom residual
  // clause out of the second subject name + shared verb (Bulma R6
  // positive-control regression). Never split when the left fragment
  // ends in a declared name, the right opens with one, and the left
  // carries no strong finite predicate of its own (a plural population
  // noun to the left — "panels" — is not a verb).
  const nameAtEnd = (t) =>
    names.some((n) => new RegExp(`\\b${escRe(n)}\\s*$`, 'i').test(t.trim()));
  const nameAtStart = (t) =>
    names.some((n) => new RegExp(`^\\s*(?:the\\s+)?${escRe(n)}\\b`, 'i').test(t));
  const coordinatedSubject = (left, right) =>
    nameAtEnd(left) && nameAtStart(right) && !hasStrongPredicate(left);
  let out = [];
  for (const piece of pieces) {
    out.push(...splitWhere(piece, BARE_COORD, 'coord',
      (left, right) => hasFiniteVerb(left, names) && hasFiniteVerb(right, names) &&
        !coordinatedSubject(left, right)));
  }
  // a left fragment is a COMPLETE clause when it carries its own finite
  // or nonfinite predicate ("... was checked with its beacon glowing"
  // leaves a gerund clause on the left of a later "despite ...").
  // Declared entity names are masked first ("filing indicator" is a
  // noun, not a gerund predicate), and a generic present-tense '-s'
  // verb counts only with a visible subject — a determiner phrase or
  // a masked entity just before it — so the plural head of a leading
  // prepositional phrase ("Among case packets with ...") is never
  // mistaken for its clause's verb.
  const SUBJECT_TOK = new Set(
    ('@ the a an this that its their his her these those every each').split(' ')
  );
  const hasOwnPredicate = (t) => {
    let masked = t;
    // a hyphenated compound containing a declared name ("lamp-lit")
    // masks as a whole, so its tail never re-emerges as a verb token
    const nameTok = new Set(names.flatMap((n) => n.toLowerCase().split(/[\s-]+/)));
    masked = masked.replace(/\b[A-Za-z]+(?:-[A-Za-z]+)+\b/g, (w) =>
      w.toLowerCase().split('-').some((p) => nameTok.has(p)) ? ' @ ' : w);
    for (const n of names) {
      masked = masked.replace(new RegExp(wordRe(n).source, 'gi'), ' @ ');
    }
    if (hasStrongPredicate(masked)) return true;
    const toks = masked.match(/@|[A-Za-z][a-z-]*/g) ?? [];
    for (let i = 1; i < toks.length; i++) {
      const raw = toks[i];
      const tok = raw.toLowerCase();
      if (tok.includes('-') || raw !== tok) continue;
      const genericVerb = tok.length >= 3 && tok.endsWith('s') && !tok.endsWith('ss') &&
        !tok.endsWith('us') && !tok.endsWith('ies') && !GENERIC_VERB_STOP.has(tok);
      const nonfiniteVerb = /^[a-z]{2,}ing$/.test(tok) && !NONFINITE_STOP.has(tok);
      if (!genericVerb && !nonfiniteVerb) continue;
      if (GENERIC_PREP_TOK.has(toks[i - 1].toLowerCase())) continue;
      // a subject marker inside the prepositional phrase itself (the
      // object of "among"/"in"/...) is not a clause subject: its own
      // preceding token must not be the governing preposition
      let hasSubject = false;
      for (let j = Math.max(0, i - 3); j < i && !hasSubject; j++) {
        if (!SUBJECT_TOK.has(toks[j].toLowerCase())) continue;
        hasSubject = j === 0 || !GENERIC_PREP_TOK.has(toks[j - 1].toLowerCase());
      }
      if (hasSubject) return true;
    }
    return false;
  };
  out = out.flatMap((piece) => splitWhere(piece, COMMA_BREAK, 'comma',
    (left, right) =>
      (!CONDITIONAL_LEAD.test(left) && hasOwnPredicate(left) && isAssertiveFragment(right, names)) ||
      (SUBORD_LEAD.test(left) && isAssertiveFragment(left, names) && hasFiniteVerb(right, names)) ||
      // leading verbless absolute ("Its pump idle, this tank was ..."):
      // no verb of its own, but a stated entity state before the main
      // clause is still an independent assertion
      (/^\s*(?:its|their|his|her|the|a|an)\s+/i.test(left) &&
        !hasFiniteVerb(left, names) && isAssertiveFragment(left, names) && hasFiniteVerb(right, names))));
  // "with/without" opens a clause only as an absolute construction over
  // a declared entity ("with its lamp lit"); a rate or instrument PP
  // ("with probability 80%") is part of its host clause. The test is
  // confined to the with-phrase head (up to the next coordinator), so
  // an entity named by a LATER coordinated clause cannot license it.
  const withClauseAssertive = (right) => {
    const head = right.split(/\s+(?:and|but)\s+|[,;]/)[0];
    return names.some((n) => wordRe(n).test(head)) && isAssertiveFragment(head, names);
  };
  // A rate fragment ("... and 20% of Pearl consoles") carries no
  // predicate of its own — the coordinator already split it from its
  // verb — yet it is a complete premise unit: a probability literal
  // on the left qualifies the split exactly as a predicate does, so a
  // concessive counterexample appended to a rate still separates.
  out = out.flatMap((piece) => splitWhere(piece, SUBORD_BREAK, 'subord',
    (left, right, intro) => (hasOwnPredicate(left) || probLikeLiteral(left)) &&
      (/with/i.test(intro) ? withClauseAssertive(right) : isAssertiveFragment(right, names))));
  return out.filter((p) => /[A-Za-z0-9]/.test(p.text));
}

// --- exact OBS span-interval subtraction (Bulma R6) ----------------------
// Connector-list clause splitting can always be defeated by one more
// connector ("even as", "following", "in spite of"): on R610 the OBS
// span covered only "At inspection the lamp on this console is lit"
// [obs.s,obs.e] inside its sentence [seg.s,seg.e], the segmenter left
// the sentence in ONE piece because "even as" is on no connector list,
// and the simultaneous-activity suffix released 0/1. For a sentence
// whose ONLY covering accounting ops are OBS (plus DET_LINK/COND_INDEP,
// whose spans cover their asserted clause) the connector lists are not
// consulted at all: the uncovered intervals of [seg.s,seg.e] minus the
// union of covering spans are computed EXACTLY, and every non-empty
// uncovered interval must be inert on its own. PRIOR/LIK spans cover
// only numeric literals, so sentences carrying any other accounting op
// are excluded — interval subtraction against a literal span would
// manufacture uncovered text out of the rate clause itself. When an
// OBS span covers the whole sentence the uncovered set is empty and
// the kernel is untouched (shipped whole-sentence OBS tests) — the
// entailment witness judges that shape via its OBS-sentence checks.
const OBS_UNCOVERED_BANNED =
  /\b(?:solo|only|both|together|silent|active|activations?|signals?|indicators?|yesterday|dawn|neither)\b/i;
const OBS_INTERVAL_PREPS = /^(?:at|in|on|during|before|after|beside|near|by|within|under|around)\b/i;

// Outer punctuation / parentheses / a leading or trailing connector are
// attachment syntax, not content: strip them before judging an interval.
function cleanIntervalText(raw) {
  let t = raw.trim().replace(/\s+/g, ' ');
  t = t.replace(/^[\s;:,()[\]]+/, '').replace(/[\s;:,()[\].!?]+$/, '');
  t = t.replace(/^(?:and|but|yet)\s+/i, '');
  t = t.replace(/^[\s;:,()[\]]+/, '').replace(/[\s;:,()[\].!?]+$/, '');
  t = t.replace(/\s+(?:and|but|yet)$/i, '');
  return t.trim();
}

// Declared-name containment that also sees hyphenated compounds: a name
// inside "lamp-only" is still a mention of the lamp.
function containsDeclaredName(text, names) {
  const norm = ` ${text.toLowerCase().replace(/-/g, ' ').replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ')} `;
  for (const n of names) {
    if (wordRe(n).test(text)) return true;
    const nn = n.toLowerCase().replace(/-/g, ' ').replace(/\s+/g, ' ').trim();
    if (nn && norm.includes(` ${nn} `)) return true;
  }
  return false;
}

// The subject shard of the OBS clause itself: an extractor may anchor
// its OBS span at the clause's verb ("This panel [was observed with
// its lamp lit]"), leaving a determiner noun-phrase prefix — optionally
// with its trailing auxiliary ("This capsule was [observed ...]") —
// formally uncovered. That prefix asserts no proposition beyond the
// OBS clause it opens: no declared entity, no lexical verb, no
// subordinator, no banned vocabulary. A prefix carrying its own verb
// or event ("Only the buzzer rang and ...") is a sister clause, never
// a shard, and fails here.
// Neutral random-sampling subject prefix of the OBS clause itself:
// "A unit drawn at random from the bin [has both a star and an amber
// notch]", "A random card from the box [has a star]". The prefix only
// states that the observed item was sampled at random from the queried
// population — zero probability literals, zero hypothesis or event
// names, zero doubt/retraction or dependency wording, and zero
// non-population restriction modifiers on the sampled or container
// noun. A restricted sampling prefix ("An Alpha unit drawn at
// random...", "A damaged unit drawn at random...", "A unit from the
// night shift drawn at random...") scopes the observation to a
// subpopulation and is never a shard.
const RANDOM_SAMPLE_VERB = '(?:drawn|picked|chosen|selected|pulled|taken|sampled)';
const RANDOM_SAMPLING_DRAWN = new RegExp(
  `^(?:a|an|one|the)\\s+(?:randomly\\s+)?([a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?)\\s+` +
    `${RANDOM_SAMPLE_VERB}\\s+at\\s+random` +
    `(?:\\s+from\\s+(?:the|this|a|an)\\s+([a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?))?$`,
  'i'
);
const RANDOM_SAMPLING_RANDOM = new RegExp(
  `^(?:a|an|one|the)\\s+random\\s+([a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?)` +
    `(?:\\s+from\\s+(?:the|this|a|an)\\s+([a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?))?$`,
  'i'
);
const SAMPLING_RESTRICT_ADJ = new Set(
  ('broken worn torn bent cracked defective faulty intact old new used spare dirty clean ' +
    'red blue green yellow black white orange purple pink brown grey gray bright dim dark ' +
    'light heavy small large big tiny short long thick thin').split(' ')
);
function samplingNpOk(npText, nameToks) {
  const toks = npText.toLowerCase().split(/[\s-]+/).filter(Boolean);
  if (toks.length === 0) return false;
  for (const mod of toks.slice(0, -1)) {
    if (nameToks.has(mod)) return false;
    if (STRATUM_MODIFIERS.has(mod)) return false;
    if (SAMPLING_RESTRICT_ADJ.has(mod)) return false;
    if (mod.length >= 4 && /(?:ed|ing|ous|ive|ful|less)$/.test(mod)) return false;
  }
  return true;
}
function isNeutralRandomSamplingPrefix(t, names) {
  if (containsDeclaredName(t, names)) return false;
  if (OBS_UNCOVERED_BANNED.test(t) || premiseDoubt(t) || probLikeLiteral(t)) return false;
  if (DEPENDENCY_WORDS.test(t)) return false;
  const m = RANDOM_SAMPLING_DRAWN.exec(t) || RANDOM_SAMPLING_RANDOM.exec(t);
  if (!m) return false;
  const nameToks = new Set(
    names.flatMap((n) => n.toLowerCase().split(/[\s-]+/).filter(Boolean))
  );
  return samplingNpOk(m[1], nameToks) &&
    (m[2] === undefined || samplingNpOk(m[2], nameToks));
}

const SHARD_AUX = new Set(['is', 'are', 'was', 'were', 'has', 'have', 'had', 'been']);
const SHARD_LEAD = new Set(
  ('a an the this that these those its their his her every each one').split(' ')
);
function isObsSubjectShard(t, names) {
  const toks = t.toLowerCase().split(/\s+/).filter(Boolean);
  if (toks.length === 0 || toks.length > 6) return false;
  if (!SHARD_LEAD.has(toks[0].replace(/['’]s$/, ''))) return false;
  if (containsDeclaredName(t, names)) return false;
  if (OBS_UNCOVERED_BANNED.test(t) || premiseDoubt(t) || probLikeLiteral(t)) return false;
  if (hasNonfinitePredicate(t)) return false;
  for (let i = 0; i < toks.length; i++) {
    const tok = toks[i].replace(/['’]s$/, '');
    if (tok.includes('-')) continue; // compound modifier ("night-shift panel")
    if (SHARD_AUX.has(tok)) {
      if (i !== toks.length - 1) return false; // aux only as the shard tail
      continue;
    }
    if (FINITE_VERB_TOK.test(tok)) return false; // lexical finite verb = own clause
    if (/^[a-z]{4,}ed$/.test(tok)) return false; // past-tense verb = own clause
    if (!/^[a-z][a-z']*$/.test(tok)) return false;
  }
  return true;
}

// The object tail of the OBS clause itself: an extractor may anchor
// twin OBS spans on subject+verb alone ("This crate triggered [both
// the tone and blink]"), leaving the coordinated object noun phrase —
// exactly the events those OBS decls assert, determiners and 'both'
// only, no verb and no state word — formally uncovered. It asserts
// nothing beyond the observations it objects.
function isObservedObjectTail(t, observedNames) {
  if (!observedNames || observedNames.length === 0) return false;
  const toks = t.toLowerCase().split(/\s+/).filter(Boolean);
  const nameToks = observedNames.map((n) => n.toLowerCase().split(/[\s-]+/).filter(Boolean));
  let i = 0;
  if (toks[i] === 'both') i++;
  let matched = 0;
  while (i < toks.length) {
    if (matched > 0) {
      if (toks[i] !== 'and') return false;
      i++;
    }
    if (toks[i] === 'the' || toks[i] === 'a' || toks[i] === 'an') i++;
    const hit = nameToks.find((nt) => nt.every((w, k) => toks[i + k] === w));
    if (!hit) return false;
    i += hit.length;
    matched++;
  }
  return matched > 0 && i === toks.length;
}

// An uncovered interval is inert only as (a) a positive inert-scene
// clause, or (b) a verbless prepositional time/place/scene adjunct with
// ZERO declared entity names (hyphenated mentions included), ZERO
// finite verbs, ZERO gerunds, and ZERO behaviour/state/quantifier
// vocabulary. A sentence-initial interval may also be (c) the subject
// shard of the OBS clause itself, and any interval may be (d) the
// object noun-phrase tail naming exactly the observed events.
// Anything else is material text no decl grounds.
export function isInertUncoveredInterval(raw, names, { sentencePrefix = false, observedNames = [] } = {}) {
  const t = cleanIntervalText(raw);
  if (!t) return true;
  if (isInertSceneSentence(t)) return true;
  if (sentencePrefix && isNeutralRandomSamplingPrefix(t, names)) return true;
  if (sentencePrefix && isObsSubjectShard(t, names)) return true;
  if (isObservedObjectTail(t, observedNames)) return true;
  if (!OBS_INTERVAL_PREPS.test(t)) return false;
  if (containsDeclaredName(t, names)) return false;
  if (hasFiniteVerb(t) || hasNonfinitePredicate(t)) return false;
  if (OBS_UNCOVERED_BANNED.test(t) || premiseDoubt(t) || probLikeLiteral(t)) return false;
  return true;
}

// Non-inert uncovered intervals of an OBS(-only) sentence, in passage
// order: [{s,e,text}]. Empty for ineligible sentences and for
// whole-sentence OBS spans.
function obsEventValues(src) {
  return src instanceof Map ? [...src.values()].filter((v) => typeof v === 'string' && v) : [];
}

function obsClausePieces(seg, names) {
  const pieces = [];
  const seen = new Set();
  const add = (s, e, via) => {
    let a = Math.max(seg.s, s);
    let b = Math.min(seg.e, e);
    while (a < b && /[\s,;]/.test(seg.text[a - seg.s] || '')) a++;
    while (b > a && /[\s,;.]/.test(seg.text[b - seg.s - 1] || '')) b--;
    const key = a + ':' + b;
    if (b > a && !seen.has(key)) {
      seen.add(key);
      pieces.push({ s: a, e: b, text: seg.text.slice(a - seg.s, b - seg.s), via: via || '' });
    }
  };
  for (const p of subClauses(seg, names) || []) {
    if (p && Number.isFinite(p.s) && Number.isFinite(p.e)) add(p.s, p.e, p.via);
  }
  const starts = new Map([[seg.s, '']]);
  let m;
  const commaRe = /[,;]\s*/g;
  while ((m = commaRe.exec(seg.text))) {
    const idx = m.index + m[0].length;
    if (idx > 0 && idx < seg.text.length) starts.set(seg.s + idx, ',');
  }
  const markerRe = /\b(even as|even though|despite|although|whereas|while|whilst|though|with|when|whenever|after|before|since|until|whose|which|but|yet|and|as)\b/gi;
  while ((m = markerRe.exec(seg.text))) starts.set(seg.s + m.index, m[1].toLowerCase());
  const ordered = [...starts.entries()].sort((x, y) => x[0] - y[0]);
  for (let i = 0; i < ordered.length; i++) {
    add(ordered[i][0], i + 1 < ordered.length ? ordered[i + 1][0] : seg.e, ordered[i][1]);
  }
  return pieces.sort((x, y) => x.s - y.s || y.e - x.e);
}

function obsAnaphoricPredicate(txt) {
  if (/\b(?:its|their)\s+[a-z][a-z-]*\s+(?:is|are|was|were|has|have|had|remain|remains|remained|stay|stays|stayed|[a-z]+(?:ed|ing|s))\b/i.test(txt)) return true;
  if (/\b(?:the\s+)?(?:other|second)\s+one\s+(?:is|are|was|were|has|have|had|remain|remains|remained|stay|stays|stayed|[a-z]+(?:ed|ing|s))\b/i.test(txt)) return true;
  return /\b[a-z][a-z-]*\s+[a-z]+(?:ed|ing|s)\b[^.]*\b(?:its|their)\s+[a-z][a-z-]*\b/i.test(txt);
}

function obsEscRe(v) {
  return String(v).replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\s+/g, '\\s+');
}

function obsVerblessStatePredicate(txt, candidateNames) {
  const t = String(txt || '');
  const state = '(?:off|on|shut|open|closed|idle|active|inactive|silent|quiet|lit|unlit|dark|bright|dead|alive|still|steady|present|absent|visible|hidden|connected|disconnected|engaged|disengaged|armed|disarmed|locked|unlocked|enabled|disabled|running|stopped)';
  const tail = '(?=$|[,;.]|\\s+(?:and\\s+' + state + '\\b|for\\b|during\\b|throughout\\b|through\\b|while\\b|when\\b|at\\b|in\\b|beside\\b|by\\b|near\\b|before\\b|after\\b|until\\b|since\\b))';
  for (const n of candidateNames || []) {
    if (n && new RegExp('\\b' + obsEscRe(n) + '\\s+' + state + '\\b' + tail, 'i').test(t)) return true;
  }
  return new RegExp('\\b(?:its|their)\\s+[a-z][a-z-]*\\s+' + state + '\\b' + tail, 'i').test(t)
    || new RegExp('\\b(?:the\\s+)?(?:other|second)\\s+one\\s+' + state + '\\b' + tail, 'i').test(t);
}

function obsMaterialNameHit(txt, candidateNames) {
  const t = String(txt || '');
  for (const n of candidateNames || []) {
    if (!n) continue;
    const re = new RegExp('\\b' + obsEscRe(n) + '\\b', 'gi');
    let m;
    while ((m = re.exec(t))) {
      const nx = t.slice(m.index + m[0].length).match(/^\s+([a-z][a-z-]*)/i);
      if (!nx) return true;
      const w = nx[1].toLowerCase();
      if (/^(?:is|are|was|were|be|been|being|has|have|had|will|would|can|could|must|may|might|shall|should|remains?|remained|stays?|stayed|seems?|seemed|off|on|shut|open|closed|idle|active|inactive|silent|quiet|lit|unlit|dark|bright|dead|alive|still|steady|present|absent|visible|hidden|connected|disconnected|engaged|disengaged|armed|disarmed|locked|unlocked|enabled|disabled|running|stopped|in|at|beside|by|near|under|over|against|behind|of|from|to|with|for|during|and|or|but|yet|while|when|as|then|now)$/.test(w)) return true;
      if (/(?:ed|ing|ly)$/.test(w) || /s$/.test(w)) return true;
      if (/^(?:rang|sang|went|came|fell|rose|began|broke|shone|struck|ran|stood|sat|spoke|woke|grew|became)$/.test(w)) return true;
    }
  }
  return false;
}

function obsPredicateSignature(txt) {
  const states = new Set('off on shut open closed idle active inactive silent quiet lit dark bright dead alive still steady present absent visible hidden connected disconnected engaged disengaged armed disarmed locked unlocked enabled disabled running stopped'.split(' '));
  const out = new Set();
  let neg = false;
  for (const raw of String(txt || '').toLowerCase().match(/[a-z][a-z-]*/g) || []) {
    if (raw === 'not' || raw === 'never' || raw === 'no' || raw === 'nt') { neg = true; continue; }
    let w = raw;
    let n = neg;
    if (!states.has(w)) {
      const un = w.match(/^(?:un|in|im|dis)([a-z]+)$/);
      if (un && (states.has(un[1]) || /(?:ed|ing)$/.test(un[1]))) { w = un[1]; n = !n; }
    }
    if (states.has(w) || /(?:ed|ing)$/.test(w)) { out.add((n ? '-' : '+') + w); neg = false; }
    else if (!/^(?:is|are|was|were|be|been|being)$/.test(raw)) neg = false;
  }
  return out;
}

function obsSameAffirmation(txt, before, observed) {
  const sameEvent = (observed || []).some((n) => n && new RegExp('\\bsame\\s+' + obsEscRe(n) + '\\b', 'i').test(txt));
  const sameTime = /\b(?:that\s+same|same)\s+(?:instant|moment|time|point)\b/i.test(txt)
    || /\bat\s+that\s+(?:instant|moment|time|point)\b/i.test(txt);
  if (!sameEvent || !sameTime) return false;
  if (/\b(?:another|different|other|earlier|later|previous|prior|last|before|after|dawn|sunrise|sunset|yesterday|tomorrow)\b/i.test(txt)) return false;
  const mainSig = obsPredicateSignature(before);
  const sig = obsPredicateSignature(txt);
  if (!sig.size || !mainSig.size) return false;
  return [...sig].every((x) => mainSig.has(x));
}

function obsHasPredicate(txt, candidateNames) {
  const t = String(txt || '').trim();
  if (!t) return false;
  if (/\b(?:is|are|was|were|be|been|being|has|have|had|will|would|can|could|must|may|might|shall|should|remains?|remained|stays?|stayed|seems?|seemed)\b/i.test(t)) return true;
  if (hasNonfinitePredicate(t) || obsAnaphoricPredicate(t)) return true;
  if (/\b[a-z][a-z-]*\s+[a-z]+(?:ed|ing)\b/i.test(t)) return true;
  if (obsVerblessStatePredicate(t, candidateNames)) return true;
  if (/^(?:beside|by|near|under|over|against|behind|at|in|on|with|from|to|of)\b/i.test(t)) return false;
  return hasFiniteVerb(t) && /\b[a-z][a-z-]*\s+[a-z]+s\b/i.test(t);
}

function isAnchorOnlyDecl(seg, d, names) {
  if (!d || !d.span || !/^(?:EVENT|HYP)(?:_DECL)?$/.test(d.op || '')) return false;
  const txt = cleanIntervalText(seg.text.slice(Math.max(seg.s, d.span.s) - seg.s, Math.min(seg.e, d.span.e) - seg.s));
  if (!txt) return true;
  const listed = Array.isArray(names) ? names : [];
  if (listed.some((n) => { const v = String(n).toLowerCase(); const w = txt.toLowerCase(); return w === v || w === 'the ' + v; })) return true;
  return containsDeclaredName(txt, names || []) && !obsHasPredicate(txt);
}

function obsNameForDecl(d, evNameMap) {
  return evNameMap ? evNameMap.get(d.ev) : undefined;
}
function obsHeadNameHit(text, allNames) {
  return allNames.find((name) => {
    const head = (String(name).toLowerCase().match(/[a-z][a-z-]*/g) || []).find((w) => !['the', 'a', 'an'].includes(w));
    return head && containsDeclaredName(text, [head]);
  }) || null;
}
function obsAcousticAssertion(txt) {
  const neg = [
    /\b(?:remains?|stays?|keeps?|kept|is|are|was|were)\s+(?:\w+ly\s+)?(?:silent|inaudible|non-?audible|mute|muted|idle)\b/i,
    /\b(?:gives?|gave|makes?|made|produces?|produced|emits?|emitted|offers?|offered)\s+(?:\w+ly\s+)?no\b/i,
    /\b(?:did|does|do)\s+not\s+(?:\w+ly\s+)?(?:ring|sound|beep|buzz|chirp|produce|emit|make)\b/i,
    /\bfailed\s+to\s+(?:ring|sound|beep|buzz|chirp)\b/i,
    /\bnot\b[^.;]{0,24}\baudible\b/i,
    /\bno\s+(?:audible\s+)?(?:sound|tone|ring|ringing|response|noise|chirp|buzz)\b/i,
    /\bwithout\s+(?:a\s+|an\s+|any\s+)?(?:audible\s+)?(?:sound|tone|ring|ringing|response|noise|chirp|buzz)\b/i,
    /\bneither\b[\s\S]*\bnor\b[\s\S]*\b(?:sound|tone|ringing|noise|chirp|buzz)\b/i,
    /\bnever\s+(?:produced|made|emitted|gave|rang|sounded)\b/i,
    /\bnot\s+(?:audibly\s+)?(?:ringing|sounding|audible|heard)\b/i,
    /\b(?:silent|inaudible|non-?audible|mute|muted)\b/i,
  ];
  if (neg.some((re) => re.test(txt))) return 'NEG';
  const pos = [
    /\baudibly\s+(?:ringing|sounding|buzzing|chirping|beeping)\b/i,
    /\b(?:is|are|was|were|can be|could be)\s+(?:audibly\s+)?(?:ringing|sounding|buzzing|chirping|beeping|heard)\b/i,
    /\b(?:is|are|was|were|can be|could be)\s+(?:\w+ly\s+)?audible\b/i,
    /\baudible\b[^.;]*\bfrom\b/i,
    /\b(?:sound|tone|noise|ringing|chirp|buzz|beep)\b[^.;]*\b(?:from|by)\b/i,
    /\bheard\b[^.;]*\b(?:sound|tone|noise|ringing|chirp|buzz|beep)\b/i,
    /\b(?:sound|tone|noise|ringing|chirp|buzz|beep)\b[^.;]*\bheard\b/i,
    /\b(?:emits?|emitted|produces?|produced|makes?|made|gives?|gave)\b[^.;]*\b(?:sound|tone|noise|ringing|chirp|buzz)\b/i,
    /\b(?:sound|tone|noise|ringing|chirp|buzz)\b[^.;]*\b(?:coming|comes|came|emanating|emanates?|emanated|issuing|issues?|issued)\s+from\b/i,
    /\b(?:rings|rang|ringing|sounds|sounded|sounding|buzzes|buzzed|buzzing|chirps|chirped|beeps|beeped)\b/i,
  ];
  if (pos.some((re) => re.test(txt))) return 'POS';
  return null;
}
function obsOrdinalBindings(text, allNames) {
  const bindings = {};
  const re = /\b(first|second)\s+(?:indicator|device)\s*(?:,|\bis\b|\bwas\b)?\s*(?:the\s+)?([a-z][a-z-]*)/gi;
  let m;
  while ((m = re.exec(String(text || '')))) {
    const name = allNames.find((n) => containsDeclaredName(n, [m[2]]));
    if (name) bindings[m[1].toLowerCase()] = name;
  }
  const re2 = /\b(first|second|former|latter)\b(?:\s+[a-z][a-z-]*){0,4}?(?:,|:|-|\bis\b|\bwas\b)\s*(?:the\s+|a\s+|an\s+)?([a-z][a-z-]*)/gi;
  while ((m = re2.exec(String(text || '')))) {
    const key = m[1].toLowerCase();
    if (bindings[key]) continue;
    const name = allNames.find((n) => containsDeclaredName(n, [m[2]]));
    if (name) bindings[key] = name;
  }
  return bindings;
}
function obsAcousticTarget(txt, before, bindings, allNames, passageText) {
  const text = String(txt || '');
  const prior = String(before || '');
  const pass = String(passageText || '');
  const bind = bindings || {};
  const headOf = (name) => (String(name).toLowerCase().match(/[a-z][a-z-]*/g) || []).find((w) => !['the', 'a', 'an'].includes(w)) || String(name).toLowerCase();
  const esc = (s) => String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const nameHits = (hay, name) => {
    const out = [];
    const cands = [String(name).toLowerCase(), headOf(name)].filter((v, i, a) => v && a.indexOf(v) === i);
    for (const c of cands) {
      const re = new RegExp('\\b' + esc(c).replace(/\s+/g, '\\s+') + '\\b', 'gi');
      let mm;
      while ((mm = re.exec(String(hay || '')))) out.push(mm.index);
    }
    return out.sort((a, b) => a - b);
  };
  const isLocativeOnly = (name) => {
    const pos = nameHits(text, name);
    if (!pos.length) return false;
    return pos.every((i) => {
      const prefix = text.slice(Math.max(0, i - 48), i).toLowerCase();
      if (/\bfrom\s+(?:the\s+|a\s+|an\s+)?$/.test(prefix)) return false;
      if (/\b(?:produced|emitted|issued|generated|made|given|sounded)\s+by\s+(?:the\s+|a\s+|an\s+)?$/.test(prefix)) return false;
      return /\b(?:near|beside|next\s+to|above|below|behind|around|at|by|on|under|over|against|towards?)\s+(?:the\s+|a\s+|an\s+)?(?:[a-z][a-z-]*\s+){0,2}$/.test(prefix)
        || /\b(?:label|tag|marking|nameplate|position|location)\b[^.,;]*$/.test(prefix);
    });
  };
  const namedList = allNames.filter((n) => containsDeclaredName(text, [n]) || nameHits(text, n).length > 0);
  const usableNamed = namedList.find((n) => !isLocativeOnly(n)) || null;
  const anyNamed = namedList.length > 0;
  const hasAmbientSubject = () => {
    const t = text.toLowerCase()
      .replace(/^(?:\s*(?:even though|even as|although|whereas|while|whilst|though|despite|with|when|whenever|and|but|yet|as)\b)+/, ' ')
      .trim();
    if (/^(?:its|their|his|her|it|they|this|that)\b/.test(t)) return false;
    const m = t.match(/^(?:a|an|the)\s+((?:[a-z][a-z-]*\s+){0,5}?[a-z][a-z-]*)(?:\s+(?:on|in|at|by|near|beside|above|below|behind|under|over|against|across|along)\s+(?:the\s+|a\s+|an\s+)?(?:[a-z][a-z-]*\s+){0,4}[a-z][a-z-]*)?\s+(?:remains?|remained|stays?|stayed|is|are|was|were|rings?|rang|sounds?|sounded|makes?|made|produces?|produced|emits?|emitted|gives?|gave|keeps?|failed|does|did|can|could|silent|inaudible|audible|mute|muted)\b/);
    if (!m) return false;
    const words = m[1].split(/\s+/);
    const head = words[words.length - 1];
    const deviceHeads = new Set([
      'indicator', 'indicators', 'device', 'devices', 'unit', 'units', 'one',
      'alarm', 'bell', 'buzzer', 'beacon', 'siren', 'horn', 'chime', 'speaker',
      'light', 'lamp', 'display', 'panel', 'signal', 'detector', 'sensor', 'annunciator',
      'sound', 'tone', 'noise', 'ring', 'ringing', 'buzz', 'chirp', 'beep', 'response',
    ]);
    if (deviceHeads.has(head)) return false;
    if (words.some((w) => ['first', 'second', 'former', 'latter', 'other', 'same'].includes(w))) return false;
    return !allNames.some((n) => headOf(n) === head || containsDeclaredName(m[1], [n]));
  };
  let selector = null;
  let ambiguous = false;
  const refPatterns = [
    /\b(?:that|the|this)\s+(same)\s+(?:[a-z][a-z-]*\s+)?(?:indicator|device|unit|one)\b/i,
    /\b(first|second|former|latter|other|remaining|another|counterpart)\b(?:\s+one)?(?:\s+[a-z][a-z-]*){0,3}?\s+of\s+(?:the\s+|these\s+|those\s+)?(?:[a-z][a-z-]*\s+){0,3}?(?:two|pair|both|couple|duo|them)\b/i,
    /\b(first|second|former|latter|other|remaining|another|counterpart)\b(?:\s+one)?\s+of\s+(?:them|these|those)\b/i,
    /\b(?:the|a|an|that|this)\s+(first|second|former|latter|other|remaining|another|counterpart)\s+one\b/i,
    /\bthe\s+(former|latter|other|remaining|counterpart)\b(?=\s*(?:[,;.]|\b(?:is|was|are|were|produced|emitted|made|gave|remains?|remained|stays?|stayed|sounds?|sounded|rings?|rang|buzz\w*|chirp\w*|beep\w*|from|at|by)\b|$))/i,
    /\b(?:the|a|an|that|this)\s+(first|second|former|latter|other|remaining|another|counterpart)\s+(?!time\b|times\b|day\b|night\b|morning\b|evening\b|hand\b|side\b|way\b|place\b)[a-z][a-z-]*\b/i,
    /\b(first|second|former|latter|other|remaining)\s+(?:indicator|device|counterpart)\b/i,
  ];
  for (const re of refPatterns) {
    const m = text.match(re);
    if (m) { selector = m[1].toLowerCase(); break; }
  }
  if (!selector) {
    const m = text.match(/\b(one|either)\b(?:\s+[a-z][a-z-]*){0,3}?\s+of\s+(?:the\s+|these\s+|those\s+)?(?:[a-z][a-z-]*\s+){0,3}?(?:two|pair|both|couple|duo|them)\b/i)
      || text.match(/\b(?:two|pair|both)\b[^.;]*\bfrom\s+(one)\b/i);
    if (m) { selector = m[1].toLowerCase(); ambiguous = true; }
  }
  if (!selector) {
    if (usableNamed) return { name: usableNamed, ref: true };
    if (anyNamed) return { name: null, ref: true };
    const stripped = text.toLowerCase().trim().replace(/^[,;\s]+/, '');
    const leadsCoord = /^(?:and|but|yet)\b/.test(stripped);
    const bare = stripped.replace(/^(?:(?:and|but|yet)\b\s*)+/, '').trim();
    const pred = bare.replace(/^(?:never|still|also|then|again|not|no\s+longer)\s+/, '');
    const priorTrim = prior.trim();
    const priorEndsCoord = /\b(?:and|but|yet)\s*$/.test(priorTrim.toLowerCase());
    const subjectless = /^(?:makes?|made|produces?|produced|emits?|emitted|issues?|issued|generates?|generated|gives?|gave|sounds?|sounded|rings?|rang|buzz\w*|chirp\w*|beep\w*|remains?|remained|stays?|stayed|keeps?|kept|continues?|continued|is|are|was|were|does|did|fails?|failed|cannot|can|could|silent|inaudible|audible|mute|muted)\b/.test(pred)
      && !/^(?:a|an|the|this|that|its|their|his|her|it|they)\b/.test(bare);
    if (subjectless && (leadsCoord || priorEndsCoord) && priorTrim && !hasAmbientSubject()) {
      const core = (priorEndsCoord
        ? priorTrim.replace(/[,;]?\s*\b(?:and|but|yet)\s*$/i, '')
        : priorTrim).replace(/[,;]\s*$/, '').trim();
      const boundaryRe = /[,;.]|\b(?:even as|even though|although|whereas|while|whilst|though|despite|when|whenever|after|before|since|until|with|and|but|yet|as)\b/gi;
      let lastStart = 0;
      let bm;
      while ((bm = boundaryRe.exec(core))) lastStart = bm.index + bm[0].length;
      const ct = core.slice(lastStart).trim();
      const anteBefore = core.slice(0, lastStart);
      if (ct && ct.toLowerCase() !== bare && anteBefore.length < prior.length) {
        const ante = obsAcousticTarget(ct, anteBefore, bind, allNames, passageText);
        if (ante && ante.name) return { name: ante.name, ref: true };
        if (ante && ante.ambient) return { name: null, ref: false, ambient: true };
        if (ante && ante.ref) return { name: null, ref: true };
      }
    }
    return { name: null, ref: false, ambient: hasAmbientSubject() };
  }
  if (selector === 'same') {
    const startsAt = [0];
    const cutRe = /[,;]|\b(?:even as|even though|although|whereas|while|whilst|though|despite|when|whenever|after|before|since|until|with|and|but|yet|as)\b/gi;
    let cm;
    while ((cm = cutRe.exec(prior))) startsAt.push(cm.index);
    const chunks = startsAt.sort((a, b) => a - b)
      .map((s, i, a) => ({ s, text: prior.slice(s, i + 1 < a.length ? a[i + 1] : prior.length) }))
      .reverse();
    const deviceRef = /\b(?:first|second|former|latter|other|remaining|another|indicator|device)\b/i;
    for (const needVerb of [true, false]) {
      for (const chunk of chunks) {
        const ct = chunk.text.trim();
        if (!ct || /^(?:[,;]\s*)?(?:beside|near|next\s+to|above|below|behind|around|at|by|on|under|over|against)\b/i.test(ct)) continue;
        const named = allNames.some((n) => nameHits(ct, n).length > 0);
        if (!named && !deviceRef.test(ct)) continue;
        if (needVerb && typeof hasFiniteVerb === 'function' && !hasFiniteVerb(ct)) continue;
        const ante = obsAcousticTarget(ct, prior.slice(0, chunk.s), bind, allNames, passageText);
        if (ante && ante.name) return { name: ante.name, ref: true };
        if (ante && ante.ref) return { name: null, ref: true };
      }
    }
    return { name: null, ref: true };
  }
  if (bind[selector]) return { name: bind[selector], ref: true };
  if (selector === 'former' && bind.first) return { name: bind.first, ref: true };
  if (selector === 'latter' && bind.second) return { name: bind.second, ref: true };
  if (ambiguous) return { name: usableNamed, ref: true };
  if (allNames.length !== 2) return { name: usableNamed, ref: true };
  const orderedFrom = (hay) => (hay && allNames.every((n) => nameHits(hay, n).length > 0)
    ? [...allNames].sort((a, b) => nameHits(hay, a)[0] - nameHits(hay, b)[0]) : null);
  const ordered = orderedFrom(pass) || orderedFrom(prior);
  if ((selector === 'first' || selector === 'former') && ordered) return { name: ordered[0], ref: true };
  if ((selector === 'second' || selector === 'latter') && ordered) return { name: ordered[1], ref: true };
  let antecedentSource = prior;
  if (pass) {
    const probe = text.toLowerCase().trim().slice(0, 30);
    const idx = probe ? pass.toLowerCase().indexOf(probe) : -1;
    if (idx >= 0 && pass.slice(0, idx).length > antecedentSource.length) antecedentSource = pass.slice(0, idx);
  }
  const hits = allNames.map((n) => ({ n, h: nameHits(antecedentSource, n) }));
  const present = hits.filter((x) => x.h.length > 0);
  let antecedent = null;
  if (present.length === 1) antecedent = present[0].n;
  else if (present.length === 2) {
    const maxFirst = Math.max(...hits.map((x) => x.h[0]));
    const top = [...hits].sort((a, b) => b.h[b.h.length - 1] - a.h[a.h.length - 1])[0];
    if (top.h[top.h.length - 1] > maxFirst) antecedent = top.n;
  }
  if (!antecedent) return { name: usableNamed, ref: true };
  if (selector === 'first' || selector === 'former') return { name: usableNamed, ref: true };
  return { name: allNames.find((n) => n.toLowerCase() !== antecedent.toLowerCase()) || null, ref: true };
}
function collectConcealedObsClauses(seg, coveredBy, names, observedNames, evNameMap, bad, passageText = '') {
  const obsSpans = coveredBy.filter((d) => d && d.op === 'OBS' && d.span);
  if (!obsSpans.length) return;
  const allNames = obsEventValues(evNameMap);
  const observed = (observedNames || []).filter((v) => typeof v === 'string' && v);
  const observedKey = new Set(observed.map((v) => v.toLowerCase()));
  const extraNames = allNames.filter((v) => !observedKey.has(v.toLowerCase()));
  const pieces = obsClausePieces(seg, names);
  if (!pieces.length) return;
  const firstS = Math.min(...pieces.map((p) => p.s));
  let trimS = seg.s;
  let trimE = seg.e;
  while (trimS < trimE && /\s/.test(seg.text[trimS - seg.s] || '')) trimS++;
  while (trimE > trimS && /[\s,;.]/.test(seg.text[trimE - seg.s - 1] || '')) trimE--;
  const startsQualifier = /^(even as|even though|despite|although|whereas|while|whilst|though|with|when|whenever|after|before|since|until|as)\b/i;
  for (const piece of pieces) {
    const raw = seg.text.slice(piece.s - seg.s, piece.e - seg.s);
    const txt = cleanIntervalText(raw);
    if (!txt || isInertUncoveredInterval(raw, names)) continue;
    const wide = obsSpans.some((d) => d.span.s <= piece.s && piece.e <= d.span.e && d.span.e - d.span.s > piece.e - piece.s);
    if (!wide) continue;
    if (piece.s <= trimS && piece.e >= trimE) continue;
    if (coveredBy.some((d) => d && d.op !== 'OBS' && !/^(?:EVENT|HYP)(?:_DECL)?$/.test(d.op || '') && !isAnchorOnlyDecl(seg, d, names) && d.span && d.span.s <= piece.s && piece.e <= d.span.e)) continue;
    const before = seg.text.slice(0, Math.max(0, piece.s - seg.s));
    const has = (list, text) => list.length > 0 && containsDeclaredName(text, list);
    const hasExtra = obsMaterialNameHit(txt, extraNames);
    const hasObserved = has(observed, txt);
    const hasAny = has(allNames, txt);
    const introducesObserved = observed.some((n) => containsDeclaredName(txt, [n]) && !containsDeclaredName(before, [n]));
    const observedBefore = has(observed, before);
    const additional = piece.s > firstS || startsQualifier.test(txt);
    const pair = /\b(both|neither|either)\b|\bthe\s+two\b/i.test(txt);
    const shifted = /\b(another|other|different|previous|prior|earlier|last)\b/i.test(txt)
      || (/^(whose|which)\b/i.test(txt) && /\b(another|other|different)\b[^,;]*$/i.test(before));
    const predicate = obsHasPredicate(txt, allNames);
    const properPart = !(piece.s <= trimS && piece.e >= trimE) && piece.e - piece.s < trimE - trimS;
    const exclusive = /\b(?:only|alone|by itself|on its own|by themselves|on their own)\b/i.test(txt);
    const contrastive = /^(even as|even though|despite|although|whereas|though|while|whilst)\b/i.test(txt);
    const sameAffirmation = !exclusive && obsSameAffirmation(txt, before, observed);
    let hit = false;
    if (hasExtra && predicate && properPart) hit = true;
    else if (shifted && additional && (hasAny || pair) && predicate) hit = true;
    else if (hasObserved && predicate && properPart && (exclusive || (!sameAffirmation && (contrastive || (additional && !introducesObserved && hasFiniteVerb(txt)))))) hit = true;
    else if (introducesObserved || hasObserved) hit = false;
    else if (pair && observed.length < 2 && additional && predicate) hit = true;
    else if (additional && observedBefore && (obsAnaphoricPredicate(txt) || obsVerblessStatePredicate(txt, allNames))) hit = true;
    if (evNameMap && properPart && additional) {
      const acousticPol = obsAcousticAssertion(txt);
      if (acousticPol) {
        const stripConnective = (t) => t.toLowerCase().replace(/\s+/g, ' ').trim()
          .replace(/^(?:(?:[,;]\s*)?(?:even as|even though|despite|although|whereas|while|whilst|though|with|when|whenever|after|before|since|until|whose|which|but|yet|and|as)\b\s*)+/, '')
          .replace(/[,\s.;]+$/, '');
        const innerAcoustic = pieces.some((q) => {
          if (q === piece || !(piece.s <= q.s && q.e <= piece.e && (q.s > piece.s || q.e < piece.e))) return false;
          const qRaw = seg.text.slice(q.s - seg.s, q.e - seg.s);
          const qTxt = cleanIntervalText(qRaw);
          if (!qTxt || isInertUncoveredInterval(qRaw, names)) return false;
          if (!obsAcousticAssertion(qTxt)) return false;
          if (stripConnective(qTxt) === stripConnective(txt)) return false;
          if (q.s <= trimS && q.e >= trimE) return false;
          if (!obsSpans.some((d) => d.span.s <= q.s && q.e <= d.span.e && d.span.e - d.span.s > q.e - q.s)) return false;
          if (coveredBy.some((d) => d && d.op !== 'OBS' && !/^(?:EVENT|HYP)(?:_DECL)?$/.test(d.op || '') && !isAnchorOnlyDecl(seg, d, names) && d.span && d.span.s <= q.s && q.e <= d.span.e)) return false;
          const qProperPart = !(q.s <= trimS && q.e >= trimE) && q.e - q.s < trimE - trimS;
          const qAdditional = q.s > firstS || startsQualifier.test(qTxt);
          return qProperPart && qAdditional;
        });
        if (innerAcoustic) {
          hit = false;
        } else {
        const preHit = hit;
        const rawPieceForTarget = seg.text.slice(piece.s - seg.s, piece.e - seg.s);
        const leadCoordForTarget = rawPieceForTarget.trim().replace(/^[\s;:,()[\]]+/, '').match(/^(?:and|but|yet)\b/i);
        const targetTxtWithCoord = leadCoordForTarget && !/^(?:and|but|yet)\b/i.test(txt) ? leadCoordForTarget[0].toLowerCase() + ' ' + txt : txt;
        const target = obsAcousticTarget(targetTxtWithCoord, before, obsOrdinalBindings(passageText || seg.text, allNames), allNames, passageText || seg.text);
        if (target.ref) {
          const grounded = !!target.name && obsSpans.some((d) => {
            const n = obsNameForDecl(d, evNameMap);
            return n && n.toLowerCase() === target.name.toLowerCase() && d.pol === acousticPol && d.span.s <= piece.s && piece.e <= d.span.e;
          });
          if (grounded && !exclusive && !pair) {
            const otherMaterial = !!target.name && allNames.some((n) => n.toLowerCase() !== target.name.toLowerCase() && obsMaterialNameHit(txt, [n]));
            const timeShift = /\b(previous|prior|earlier|previously|beforehand)\b/i.test(txt) || /\b(?:last|another|different)\s+(?:time|occasion|instance)\b/i.test(txt);
            if (!(preHit && (otherMaterial || timeShift))) hit = false;
          }
          else if (!grounded) hit = true;
        } else if (target.ambient) {
          hit = false;
        } else if (allNames.length > 1) {
          const covering = obsSpans.filter((d) => obsNameForDecl(d, evNameMap)
            && d.span.s <= piece.s && piece.e <= d.span.e);
          const licensed = covering.length >= 2 && covering.every((d) => d.pol === acousticPol);
          if (!licensed) hit = true;
        }
        }
      }
    }
    if (!hit) continue;
    if (bad.some((x) => piece.s < x.e && x.s < piece.e)) continue;
    bad.push({ s: piece.s, e: piece.e, text: txt });
  }
}

export function uncoveredObsIntervals(seg, coveredBy, names, observedNames = [], evNameMap = null, passageText = '') {
  // Only the OBS decl's OWN sentence (the one its span starts in) is
  // interval-accounted: a span that happens to overlap a neighbouring
  // sentence does not make that sentence an OBS sentence.
  if (!coveredBy.some((d) => d.op === 'OBS' && d.span.s >= seg.s && d.span.s < seg.e)) return [];
  if (!coveredBy.every((d) => d.op === 'OBS' || d.op === 'DET_LINK' || d.op === 'COND_INDEP' || isAnchorOnlyDecl(seg, d, names))) {
    return [];
  }
  const spans = coveredBy.filter((d) => !isAnchorOnlyDecl(seg, d, names))
    .map((d) => ({ s: Math.max(d.span.s, seg.s), e: Math.min(d.span.e, seg.e) }))
    .filter((sp) => sp.s < sp.e)
    .sort((a, b) => a.s - b.s);
  const bad = [];
  collectConcealedObsClauses(seg, coveredBy, names, observedNames, evNameMap, bad, passageText);
  let cur = seg.s;
  const consider = (s, e) => {
    if (s >= e) return;
    const text = seg.text.slice(s - seg.s, e - seg.s);
    const cleaned = cleanIntervalText(text);
    if (cleaned === 'and' && spans.some((p) => p.e <= s) && spans.some((p) => p.s >= e)) return;
    if (!cleaned) return;
    // A span boundary falling INSIDE a word leaves that word's tail as
    // a one-token uncovered fragment — boundary slop in the span, not
    // a separate assertion (a multi-word interval never qualifies).
    const relS = s - seg.s;
    if (relS > 0 && /[A-Za-z0-9]/.test(seg.text[relS - 1]) && /^[A-Za-z0-9]/.test(text) &&
        /^[A-Za-z0-9][A-Za-z0-9'-]*$/.test(cleaned)) return;
    if (!isInertUncoveredInterval(text, names, { sentencePrefix: s === seg.s, observedNames })) {
      bad.push({ s, e, text });
    }
  };
  for (const sp of spans) {
    consider(cur, sp.s);
    cur = Math.max(cur, sp.e);
  }
  consider(cur, seg.e);
  return bad;
}

// --- rate-sentence tail accounting (Bulma R7) ---------------------------
// Exact OBS interval accounting left PRIOR/LIK sentences open: a LIK span
// covers only its numeric literal ("20% of Pearl"), so a counterexample
// appended after the unit noun ("... consoles even as both devices fired
// simultaneously") sat in the same sub-clause piece as the rate and no
// interval check ever saw it. Structural rule, connector-free: in a
// sentence carrying PRIOR/LIK/JOINT_LIK/PART_EXHAUSTIVE spans, the text
// after the RIGHTMOST rate span — skipping at most the rate's own
// unit/population noun phrase ("consoles", "of Flint cases", "in the
// Flint and Pearl population") — must be inert: a verbless population
// or time/place/source adjunct ("in the survey", "on Monday") with zero
// event names, zero finite or gerund predicates and zero co-/solo-
// activation vocabulary, an inert scene clause, or the rate clause's
// own simple predicate naming exactly the entity that rate asserts
// ("of cases are Eta", "of Epsilon cases trigger the relay alarm").
// A concessive marker in the leading frame before the first rate span
// is judged the same way. Anything else is material text no decl
// grounds, however it is attached.
export const CONCESSIVE_MARKER =
  /\b(?:even\s+as|despite|in\s+spite\s+of|notwithstanding|although|even\s+though|save\s+that|except\s+that)\b/i;
const RATE_OPS = new Set(['PRIOR', 'LIK', 'JOINT_LIK', 'PART_EXHAUSTIVE']);
const RATE_TAIL_BANNED =
  /\b(?:both|devices?|outputs?|signals?|indicators?|simultaneously|together|alone|solo|silent|off|idle|inactive|documented|despite|notwithstanding|even|case|fired|lit|rang|sounded|occurred|remained|stayed|was|were|is|are)\b/i;
const RATE_NP_GLUE = new Set(
  ('of in on at among for the a an and per within under across between from').split(' ')
);

// Verbless unit/population/adjunct noun phrase over HYP names only:
// glue words and plain nounish tokens, no strong/gerund predicate, no
// event name, no banned activation vocabulary.
function isRateNounPhrase(text, hypNames, eventNames) {
  const t = cleanIntervalText(text);
  if (!t) return true;
  if (RATE_TAIL_BANNED.test(t) || CONCESSIVE_MARKER.test(t)) return false;
  let masked = t;
  for (const n of hypNames) masked = masked.replace(new RegExp(wordRe(n).source, 'gi'), ' @ ');
  if (hasNonfinitePredicate(masked)) return false;
  const toks = masked.toLowerCase().match(/@|[a-z][a-z'-]*/g) ?? [];
  if (toks.length === 0) return false;
  // Event mentions: a single-token '-ed' event name directly before
  // the noun it modifies ("80% of flagged cases", the event being
  // "flagged") is the unit population's participial adjective, not a
  // stated occurrence; any other event mention — including inside a
  // hyphenated compound — is content no rate decl ledgers.
  const evToks = eventNames.map((n) => n.toLowerCase().split(/[\s-]+/).filter(Boolean));
  const evParts = new Set(evToks.flat());
  for (let i = 0; i < toks.length; i++) {
    if (toks[i].includes('-')) {
      if (toks[i].split('-').some((p) => evParts.has(p))) return false;
      continue;
    }
    for (const nt of evToks) {
      if (nt.length === 0 || !nt.every((w, k) => toks[i + k] === w)) continue;
      const adjective = nt.length === 1 && /^[a-z]{4,}ed$/.test(nt[0]) &&
        i + 1 < toks.length && !RATE_NP_GLUE.has(toks[i + 1]);
      if (!adjective) return false;
    }
  }
  // An '-ed' token directly before the noun it modifies ("of flagged
  // cases") is a participial adjective inside the noun phrase; in
  // predicate position (final, or before a determiner/preposition) it
  // is a past-tense verb and the phrase is a clause, not an NP. The
  // banned list above already rejects the report forms ("documented
  // case", "fired", ...).
  return toks.every((tok, i) => {
    if (tok === '@' || RATE_NP_GLUE.has(tok) || tok.includes('-')) return true;
    if (FINITE_VERB_TOK.test(tok)) return false;
    if (/^[a-z]{4,}ed$/.test(tok)) {
      return i < toks.length - 1 && !RATE_NP_GLUE.has(toks[i + 1]);
    }
    return true;
  });
}

// The rate clause's own predicate tail: [unit NP] + one verb + optional
// determiner + exactly one name the rightmost rate decl asserts, and
// nothing after it. A second name, an adverb, or any state word makes
// it a new assertion, never the rate's own tail.
function isRatePredicateTail(text, hypNames, eventNames, assertedNames) {
  if (!assertedNames.length) return false;
  const toks = cleanIntervalText(text).toLowerCase().match(/[a-z][a-z'-]*/g) ?? [];
  const nameToks = assertedNames.map((n) => n.toLowerCase().split(/[\s-]+/).filter(Boolean));
  for (let v = 0; v < toks.length; v++) {
    const prefix = toks.slice(0, v).join(' ');
    if (prefix && !isRateNounPhrase(prefix, hypNames, eventNames)) continue;
    const verb = toks[v];
    if (!verb || RATE_NP_GLUE.has(verb) || CONCESSIVE_MARKER.test(verb)) continue;
    if (/\b(?:both|alone|solo|simultaneously|together|despite|even)\b/.test(verb)) continue;
    let i = v + 1;
    if (toks[i] === 'the' || toks[i] === 'a' || toks[i] === 'an') i++;
    if (nameToks.some((nt) => nt.every((w, k) => toks[i + k] === w) && i + nt.length === toks.length)) {
      return true;
    }
  }
  return false;
}

function hasRateCoactMarker(text) {
  const t = String(text || '');
  return /\b(simultane\w+|concurren\w+|together|jointly|co-?occur\w*|co-?activ\w*)\b/i.test(t)
    || /\b(in\s+unison|in\s+tandem|at\s+once|same\s+(?:instant|moment|time)|very\s+instant)\b/i.test(t)
    || /\bjoint\s+(?:activation|occurrence|firing)\b/i.test(t)
    || /\bboth\b[^,;]*\b(?:on|fired|activat\w*|lit|sounded|flashed)\b/i.test(t);
}
function isRecordRateVerb(word) {
  return /^(not|record|tabul|transcrib|log|list|compil|collect|document|enter|writ|count|measur|survey|report|observ|catalog|archiv)/i.test(word || '');
}
function rateInsetFiniteVerbs(text) {
  const irregular = new Set(['lit', 'saw', 'seen', 'rang', 'sang', 'was', 'were', 'is', 'are', 'has', 'had', 'became', 'came', 'shone', 'struck', 'began', 'wrote', 'written', 'given', 'taken', 'known', 'shown']);
  return (String(text || '').toLowerCase().match(/[a-z]+/g) || [])
    .filter((w) => irregular.has(w) || (w.length > 3 && /ed$/.test(w) && w !== 'included'));
}
function isNeutralRateSource(source, hypNames, eventNames) {
  const rawSource = String(source || '').trim();
  const strippedSource = rateStripBenignLocativeRelative(rawSource);
  const s = strippedSource === null ? rawSource : strippedSource.trim();
  if (!s || premiseDoubt(s) || probLikeLiteral(s)) return false;
  if (containsDeclaredName(s, eventNames) || hasRateCoactMarker(s) || /\bwithout\b/i.test(s)) return false;
  if ((s.match(/\b(?:who|which|that|where|whose)\b/gi) || []).length > 1) return false;
  const relVerb = s.match(/\b(?:who|that|which)\s+([a-z]+)/i);
  if (relVerb && !isRecordRateVerb(relVerb[1])) return false;
  return !rateInsetFiniteVerbs(s).some((w) => !isRecordRateVerb(w));
}
const RATE_EVENT_VERB_STEMS = new Set(['activate', 'appear', 'begin', 'blink', 'buzz', 'chime', 'chirp', 'click', 'emit', 'fire', 'flash', 'flicker', 'glow', 'hum', 'illuminate', 'light', 'occur', 'operate', 'peal', 'pulse', 'radiate', 'resonate', 'ring', 'shine', 'signal', 'sing', 'sound', 'start', 'strike', 'toll', 'trigger', 'vibrate', 'warble', 'whir', 'whistle']);
const RATE_IRREGULAR_VERB_STEMS = { lit: 'light', rang: 'ring', rung: 'ring', shone: 'shine', struck: 'strike', sang: 'sing', sung: 'sing', began: 'begin', begun: 'begin', became: 'become', came: 'come' };
const RATE_EVENT_STATE_WORDS = new Set(['idle', 'silent', 'lit', 'unlit', 'dark', 'ringing', 'glowing', 'shining', 'flashing', 'blinking', 'buzzing', 'sounding', 'audible', 'inaudible', 'active', 'inactive', 'dormant', 'extinguished', 'illuminated', 'quiet', 'off']);
const RATE_EVENT_STATE_NOUNS = new Set(['sound', 'tone', 'noise', 'ringing', 'response', 'light', 'glow', 'flash', 'signal', 'illumination', 'activation', 'occurrence']);
function rateWordStems(word) {
  const w = String(word || '').toLowerCase();
  const stems = new Set(w ? [w] : []);
  if (RATE_IRREGULAR_VERB_STEMS[w]) stems.add(RATE_IRREGULAR_VERB_STEMS[w]);
  if (/ies$/.test(w) && w.length > 4) stems.add(`${w.slice(0, -3)}y`);
  if (/(ches|shes|sses|xes|zes|oes)$/.test(w)) stems.add(w.slice(0, -2));
  else if (/s$/.test(w) && !/ss$/.test(w) && w.length > 3) stems.add(w.slice(0, -1));
  if (/ing$/.test(w) && w.length > 5) {
    const base = w.slice(0, -3);
    stems.add(base); stems.add(`${base}e`);
    if (base.length > 2 && base[base.length - 1] === base[base.length - 2]) stems.add(base.slice(0, -1));
  }
  if (/ed$/.test(w) && w.length > 4) {
    const base = w.slice(0, -2);
    stems.add(base); stems.add(`${base}e`);
    if (base.length > 2 && base[base.length - 1] === base[base.length - 2]) stems.add(base.slice(0, -1));
  }
  return stems;
}
function rateClauseTokens(text) {
  return String(text || '').toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g) || [];
}
function rateIsBenignLocativeRelative(rel) {
  const raw = String(rel || '').trim();
  if (!raw || /[,;:]/.test(raw)) return false;
  const toks = rateClauseTokens(raw.replace(/[.!?]+$/, ''));
  if (!toks.length || toks.some((t) => ['while', 'when', 'whenever', 'whilst', 'during', 'and', 'but', 'or', 'not', 'never', 'no', 'without', 'if', 'because'].includes(t) || /n't$/.test(t) || /ly$/.test(t))) return false;
  const locVerbs = new Set(['is', 'are', 'was', 'were', 'sits', 'sit', 'sat', 'stands', 'stand', 'stood', 'lies', 'lie', 'lay', 'rests', 'rest', 'rested']);
  let verbAt = -1;
  let subject = [];
  if (toks[0] === 'whose') {
    verbAt = toks.findIndex((t, i) => i >= 2 && locVerbs.has(t));
    if (verbAt < 2 || verbAt > 4) return false;
    subject = toks.slice(1, verbAt);
  } else if (toks[0] === 'which' || toks[0] === 'that') {
    if (!locVerbs.has(toks[1])) return false;
    verbAt = 1;
    subject = [toks[0]];
  } else return false;
  if (!subject.length || subject.some((t) => RATE_EVENT_STATE_WORDS.has(t) || /(ing|ed)$/.test(t))) return false;
  if (subject.length > 1 && subject.slice(0, -1).some((t) => [...rateWordStems(t)].some((stem) => RATE_EVENT_VERB_STEMS.has(stem)))) return false;
  let compAt = -1;
  if (toks[verbAt + 1] === 'next' && toks[verbAt + 2] === 'to') compAt = verbAt + 3;
  else if (['beside', 'near', 'by', 'at', 'in', 'on', 'under', 'behind', 'against', 'alongside'].includes(toks[verbAt + 1])) compAt = verbAt + 2;
  else return false;
  const comp = toks.slice(compAt);
  if (comp.length < 2 || comp.length > 4) return false;
  if (!['its', 'the', 'a', 'an', 'this', 'that', 'their', 'his', 'her'].includes(comp[0])) return false;
  const compNouns = comp.slice(1);
  if (compNouns.some((t) => RATE_EVENT_STATE_WORDS.has(t) || locVerbs.has(t) || /(ing|ed)$/.test(t))) return false;
  if (compNouns.length > 1 && compNouns.slice(0, -1).some((t) => [...rateWordStems(t)].some((stem) => RATE_EVENT_VERB_STEMS.has(stem)))) return false;
  return true;
}
function rateStripBenignLocativeRelative(text) {
  const raw = String(text || '');
  const re = /\b(?:whose|which|that)\b/gi;
  let m;
  let found = null;
  while ((m = re.exec(raw))) {
    if (rateIsBenignLocativeRelative(raw.slice(m.index))) found = raw.slice(0, m.index);
  }
  return found;
}
function rateEventVerbStems(eventNames) {
  const stems = new Set();
  for (const name of eventNames || []) {
    const toks = String(name).toLowerCase().match(/[a-z][a-z-]*/g) || [];
    toks.forEach((tok, idx) => {
      for (const stem of rateWordStems(tok)) {
        if (RATE_EVENT_VERB_STEMS.has(stem)) stems.add(stem);
        if (toks.length > 1 && idx === toks.length - 1 && stem.length > 2) stems.add(stem);
      }
    });
  }
  return stems;
}
function rateClauseReferencesEvent(clause, eventNames) {
  if (containsDeclaredName(clause, eventNames)) return true;
  if (/\b(?:first|second|other|remaining|its|the)\s+(?:indicator|device|light|lamp|beacon|buzzer)\b/i.test(clause)) return true;
  const eventStems = new Set();
  for (const name of eventNames || []) {
    for (const tok of String(name).toLowerCase().match(/[a-z][a-z-]*/g) || []) {
      if (tok.length > 2 && !['the', 'and', 'with', 'its'].includes(tok)) {
        for (const stem of rateWordStems(tok)) eventStems.add(stem);
      }
    }
  }
  return rateClauseTokens(clause).some((tok) => [...rateWordStems(tok)].some((stem) => eventStems.has(stem)));
}
function rateHasEventVerb(text, eventNames) {
  const declared = rateEventVerbStems(eventNames);
  return rateClauseTokens(text).some((tok) => !isRecordRateVerb(tok)
    && [...rateWordStems(tok)].some((stem) => declared.has(stem) || RATE_EVENT_VERB_STEMS.has(stem)));
}
function rateHasTemporalEventOverlap(clause, eventNames) {
  const m = String(clause || '').match(/\b(while|whenever|when|whilst|during)\b/i);
  if (!m) return false;
  const after = String(clause).slice(m.index + m[0].length);
  return rateClauseReferencesEvent(after, eventNames) || rateHasEventVerb(after, eventNames);
}
function rateHasAbsencePresentation(clause, eventNames) {
  const toks = rateClauseTokens(clause);
  const declared = rateEventVerbStems(eventNames);
  for (let i = 0; i < toks.length; i++) {
    const strong = ['no', 'without', 'absent'].includes(toks[i]);
    const neg = strong || ['not', 'never', 'fail', 'fails', 'failed', 'failing'].includes(toks[i]) || /n't$/.test(toks[i]);
    if (!neg) continue;
    const win = toks.slice(i + 1, i + 6);
    if (win.some((tok) => [...rateWordStems(tok)].some((stem) => declared.has(stem) || RATE_EVENT_VERB_STEMS.has(stem) || RATE_EVENT_STATE_NOUNS.has(stem) || RATE_EVENT_STATE_WORDS.has(stem)))) return true;
    if (strong && rateClauseReferencesEvent(win.join(' '), eventNames)) return true;
  }
  return /\bno\s+(?:audible\s+)?(?:sound|tone|response|ringing|noise)\b/i.test(clause);
}
function rateEventAssertionMaterial(clause, eventNames) {
  const strippedClause = rateStripBenignLocativeRelative(clause);
  const c = strippedClause === null ? clause : strippedClause;
  if (!String(c || '').trim() || !rateClauseReferencesEvent(c, eventNames)) return false;
  const pastVerb = rateInsetFiniteVerbs(c).some((w) => !isRecordRateVerb(w) && !['is', 'are', 'was', 'were'].includes(w));
  return pastVerb
    || rateHasEventVerb(c, eventNames)
    || rateHasTemporalEventOverlap(c, eventNames)
    || rateHasAbsencePresentation(c, eventNames)
    || rateClauseTokens(c).some((tok) => RATE_EVENT_STATE_WORDS.has(tok))
    || /\bnon-?audible\b/i.test(c);
}
function rateWhoseMaterial(inset, eventNames) {
  const at = inset.search(/\b(?:whose|which|who|that|where)\b/i);
  if (at < 0) return false;
  return rateEventAssertionMaterial(inset.slice(at), eventNames);
}
function classifyRateInset(text, hypNames, eventNames) {
  let inset = String(text || '').trim();
  if (!inset) return null;
  const strippedInset = rateStripBenignLocativeRelative(inset);
  if (strippedInset !== null) inset = strippedInset.trim().replace(/[\s,;:.]+$/, '');
  if (!inset) return 'neutral';
  if (hasRateCoactMarker(inset)) return 'material';
  if (rateWhoseMaterial(inset, eventNames)) return 'material';
  if (rateEventAssertionMaterial(inset, eventNames)) return 'material';
  const sourceAt = inset.search(/\baccording\s+to\b/i);
  if (sourceAt >= 0) {
    const source = inset.slice(sourceAt).replace(/^according\s+to\b/i, '');
    return isNeutralRateSource(source, hypNames, eventNames) ? 'neutral' : 'material';
  }
  const verbs = rateInsetFiniteVerbs(inset);
  const nonRecord = verbs.filter((w) => !isRecordRateVerb(w));
  if (/\bwithout\b/i.test(inset) && (verbs.length || containsDeclaredName(inset, eventNames))) return 'material';
  if (nonRecord.length && nonRecord.every((w) => ['is', 'are', 'was', 'were'].includes(w))
    && /\b(beside|near|next\s+to|by|at|in|on|under|behind)\b/i.test(inset)
    && !containsDeclaredName(inset, eventNames)) return 'neutral';
  if (nonRecord.length) return 'material';
  if (isInertUncoveredInterval(inset, [...hypNames, ...eventNames]) || isInertSceneSentence(inset)) return 'neutral';
  return 'neutral';
}
function rateBridgeStart(text, from, hypNames, eventNames) {
  let best = -1;
  const re = /\band\b/gi;
  re.lastIndex = from;
  let m;
  while ((m = re.exec(text))) {
    const suffix = text.slice(m.index);
    const clean = suffix.replace(/[\s,;:.]+$/, '');
    const afterAnd = clean.replace(/^and\b/i, '').trim();
    if (!afterAnd) { best = m.index; continue; }
    if (hasRateCoactMarker(suffix) || !containsDeclaredName(suffix, [...hypNames, ...eventNames])) continue;
    const bridgeTrailer = /\b(for|a|an|the|is|are)$/i.test(clean)
      || /\b(holds?|accounts?|sounds?|rings?|glows?)(\s+for)?$/i.test(clean);
    if (!bridgeTrailer && rateEventAssertionMaterial(afterAnd, eventNames)) continue;
    if (bridgeTrailer || afterAnd.split(/\s+/).filter(Boolean).length <= 2) best = m.index;
  }
  return best;
}
export function isInertRateTail(raw, hypNames, eventNames, assertedNames = []) {
  const t = cleanIntervalText(raw);
  if (!t) return true;
  if (isInertSceneSentence(t)) return true;
  if (premiseDoubt(t) || probLikeLiteral(t)) return false;
  if (isRateNounPhrase(t, hypNames, eventNames)) return true;
  // A verbless coordinated-arm object: the whole tail is exactly the
  // rightmost rate's own asserted event as a bare noun phrase ("... and
  // one in four a stripe" — the shared verb is elided). It names the
  // event the bound LIK itself asserts and predicates nothing further.
  for (const n of assertedNames) {
    if (!eventNames.includes(n)) continue;
    const alts = inflectedNameVariants(n)
      .map((v) => escRe(v).replace(/[\s-]+/g, '[\\s-]+')).join('|');
    if (new RegExp(`^(?:(?:the|a|an)\\s+)?(?:${alts})$`, 'i').test(t)) return true;
  }
  const sourceSplit = t.match(/^(.*?)\baccording\s+to\b\s*([\s\S]+)$/i);
  if (sourceSplit && isNeutralRateSource(sourceSplit[2], hypNames, eventNames)) {
    const beforeSource = cleanIntervalText(sourceSplit[1]).replace(/^of\s+/i, '');
    if (!beforeSource || isRateNounPhrase(beforeSource, hypNames, eventNames)) return true;
  }
  return isRatePredicateTail(t, hypNames, eventNames, assertedNames);
}

// Non-inert leading/trailing intervals of a rate sentence, in passage
// order: [{s,e,text}]. Empty when no rate span starts in the sentence
// or another accounting span already covers the tail.
export function uncoveredRateIntervals(seg, coveredBy, X, prog, names) {
  const clamp = (d) => ({ s: Math.max(d.span.s, seg.s), e: Math.min(d.span.e, seg.e) });
  const rateDecls = coveredBy.filter(
    (d) => RATE_OPS.has(d.op) && d.span.s >= seg.s && d.span.s < seg.e
  );
  if (rateDecls.length === 0) return [];
  const spans = rateDecls.map(clamp).filter((sp) => sp.s < sp.e);
  if (spans.length === 0) return [];
  const maxE = Math.max(...spans.map((sp) => sp.e));
  const minS = Math.min(...spans.map((sp) => sp.s));
  const hypNameById = declaredHypNames(X, prog);
  const evNameById = eventNameMap(X, prog);
  const hypNames = [...hypNameById.values()];
  const eventNames = [...evNameById.values()];
  const rightmost = rateDecls.reduce((a, d) => (clamp(d).e > clamp(a).e ? d : a));
  const assertedNames = [];
  if (rightmost.op === 'PRIOR' && hypNameById.get(rightmost.hyp)) assertedNames.push(hypNameById.get(rightmost.hyp));
  if (rightmost.op === 'PART_EXHAUSTIVE') {
    for (const h of rightmost.hyps ?? []) if (hypNameById.get(h)) assertedNames.push(hypNameById.get(h));
  }
  if (rightmost.op === 'LIK') {
    if (evNameById.get(rightmost.ev)) assertedNames.push(evNameById.get(rightmost.ev));
    // "20% are Fir" — the rate clause's own copula naming its GIVEN
    // hypothesis is still the rate's own predicate (whether the
    // binding direction is sound is the gate's question, not the
    // tail's)
    if (hypNameById.get(rightmost.hyp)) assertedNames.push(hypNameById.get(rightmost.hyp));
  }
  if (rightmost.op === 'JOINT_LIK') {
    for (const s of rightmost.spec ?? []) if (evNameById.get(s.ev)) assertedNames.push(evNameById.get(s.ev));
    if (hypNameById.get(rightmost.hyp)) assertedNames.push(hypNameById.get(rightmost.hyp));
  }
  const bad = [];
  // leading frame: only a concessive lead is suspect — ordinary rate
  // frames ("Given category X, ...", "On the night shift, ...") are
  // the rate clause's own subject/conditioner
  for (const d of coveredBy) {
    if (d.op !== 'PRIOR' && !RATE_OPS.has(d.op)) continue;
    const cs = clamp(d);
    if (cs && cs.e > cs.s && !spans.some((sp) => sp.s === cs.s && sp.e === cs.e)) spans.push(cs);
  }
  const orderedRateSpans = [...spans].sort((a, b) => a.s - b.s || a.e - b.e);
  let prevRateSpan = null;
  for (const curRateSpan of orderedRateSpans) {
    if (prevRateSpan && curRateSpan.s > prevRateSpan.e) {
      const gapS = prevRateSpan.e;
      const gapE = curRateSpan.s;
      const gapText = seg.text.slice(gapS - seg.s, gapE - seg.s);
      const introAt = gapText.search(/\b(including|included|according\s+to|on\s+which|whose|where|with|in)\b/i);
      if (introAt < 0) {
        if (hasRateCoactMarker(gapText) && !bad.some((f) => f.s < gapE && gapS < f.e)) {
          bad.push({ s: gapS, e: gapE, text: gapText });
        }
      } else {
        let insetEnd = gapText.length;
        const bridgeAt = rateBridgeStart(gapText, introAt, hypNames, eventNames);
        if (bridgeAt > introAt) insetEnd = bridgeAt;
        const bareAnd = /[,;\s]*\band\s*$/i.exec(gapText.slice(introAt, insetEnd));
        if (bridgeAt < 0 && bareAnd) insetEnd -= bareAnd[0].length;
        const partStarts = [introAt];
        const partRe = /[,;]|\b(?:including|according\s+to)\b/gi;
        partRe.lastIndex = introAt + 1;
        let pm;
        while ((pm = partRe.exec(gapText)) && pm.index < insetEnd) {
          partStarts.push(pm[0] === ',' || pm[0] === ';' ? pm.index + 1 : pm.index);
        }
        for (let pi = 0; pi < partStarts.length; pi++) {
          const partS = partStarts[pi];
          const partE = pi + 1 < partStarts.length ? partStarts[pi + 1] : insetEnd;
          const insetText = gapText.slice(partS, partE);
          if (!insetText.trim()) continue;
          const kind = classifyRateInset(insetText, hypNames, eventNames);
          if (kind) {
            const insetS = gapS + partS;
            const insetE = gapS + partE;
            if (kind === 'material' && !bad.some((f) => f.s < insetE && insetS < f.e)) {
              bad.push({ s: insetS, e: insetE, text: insetText });
            }
            if (insetE > insetS) spans.push({ s: insetS, e: insetE, text: insetText });
          }
        }
      }
    }
    prevRateSpan = prevRateSpan && curRateSpan.s <= prevRateSpan.e
      ? { s: prevRateSpan.s, e: Math.max(prevRateSpan.e, curRateSpan.e) }
      : curRateSpan;
  }
  spans.sort((a, b) => a.s - b.s || a.e - b.e);
  const lead = seg.text.slice(0, minS - seg.s);
  if (CONCESSIVE_MARKER.test(lead)) {
    const leadClause = lead.split(',')[0];
    if (!isInertSceneSentence(cleanIntervalText(leadClause)) &&
        !isRateNounPhrase(leadClause, hypNames, eventNames)) {
      bad.push({ s: seg.s, e: seg.s + leadClause.length, text: leadClause });
    }
  }
  // Trailing interval, minus any later ASSERTION span sharing the
  // sentence. HYP/EVENT spans are name anchors, not assertions: one
  // anchored at the first mention INSIDE the smuggled clause itself
  // must never ledger that clause, so anchors are not subtracted.
  const covering = coveredBy
    .filter((d) => d.op !== 'HYP' && d.op !== 'EVENT')
    .map(clamp).filter((sp) => sp.s < sp.e && sp.e > maxE)
    .sort((x, y) => x.s - y.s);
  const gaps = [];
  let cur = maxE;
  for (const sp of covering) {
    if (sp.s > cur) gaps.push({ s: cur, e: sp.s });
    cur = Math.max(cur, sp.e);
  }
  if (cur < seg.e) gaps.push({ s: cur, e: seg.e });
  for (const gap of gaps) {
    const tail = seg.text.slice(gap.s - seg.s, gap.e - seg.s);
    const cleaned = cleanIntervalText(tail);
    const relS = gap.s - seg.s;
    const wordSlop = relS > 0 && /[A-Za-z0-9]/.test(seg.text[relS - 1]) &&
      /^[A-Za-z0-9]/.test(tail) && /^[A-Za-z0-9][A-Za-z0-9'-]*$/.test(cleaned);
    if (cleaned && !wordSlop && !isInertRateTail(tail, hypNames, eventNames, assertedNames)) {
      bad.push({ s: gap.s, e: gap.e, text: tail });
    }
  }
  return bad;
}

// First residual sub-clause of an accounted sentence that no accounting
// span touches and that asserts material content of its own, or null.
// A clause is accounted when any ACCOUNTING_OPS span overlaps IT (not
// merely its sentence); a residual clause is excused only by positive
// inert evidence — the static-scene grammar (via sentenceMateriality)
// or the verbless adjunct form above.
function unaccountedResidualClause(row, names) {
  const pieces = subClauses(row.seg, names);
  if (pieces.length < 2) return null;
  for (const piece of pieces) {
    if (row.coveredBy.some((d) => d.span.s < piece.e && piece.s < d.span.e)) continue;
    if (isInertAdjunctClause(piece.text, names)) continue;
    const m = sentenceMateriality(piece, names);
    if (m.kind !== 'material') continue;
    // a fragment asserting no proposition of its own — a noun phrase
    // split off by a coordinator, or the subject shard before a
    // parenthetical insertion ("Cedar (the 40% share ...) made up
    // 60%") — belongs to its grounded host clause, unless it carries
    // doubt or a probability literal of its own
    if (!isAssertiveFragment(piece.text, names) &&
        !premiseDoubt(piece.text) && !probLikeLiteral(piece.text)) continue;
    return { piece, materiality: m };
  }
  return null;
}

// Residual/adjacent clauses inside a premise sentence, for the
// entailment witness: kernel accounting treats a clause as covered when
// ANY accounting span overlaps it, but a span stretched over a whole
// compound sentence semantically binds only the clause it is anchored
// in (the clause containing its span start). A material assertive
// clause with no decl anchored in it — a parenthetical counterexample,
// a comma-absolute or concessive clause stating the OTHER event's
// state inside the OBS sentence — is adjacent content no decl asserts,
// whether or not some wide span happens to overlap it. Returns
// [{segIdx, piece, materiality, overlapping:[decl]}] in passage order;
// inert adjuncts and statically inert clauses are excused as above.
export function adjacentClauseFindings(X, prog) {
  const segs = sentences(X);
  const names = entityNames(X, prog);
  const evNameById = eventNameMap(X, prog);
  const hypNameById = declaredHypNames(X, prog);
  // Entity names a decl actually asserts, for the integrated
  // "with"-absolute payload test below.
  const assertedNames = (d) => {
    const out = new Set();
    const ev = (id) => { const n = evNameById.get(id); if (n) out.add(n.toLowerCase()); };
    const hyp = (id) => { const n = hypNameById.get(id); if (n) out.add(n.toLowerCase()); };
    if (d.ev) ev(d.ev);
    if (d.a) ev(d.a);
    if (d.b) ev(d.b);
    if (Array.isArray(d.evs)) d.evs.forEach(ev);
    if (Array.isArray(d.spec)) d.spec.forEach((s) => ev(s.ev));
    if (d.op === 'EVENT') ev(d.id);
    if (d.op === 'HYP') hyp(d.id);
    if (d.hyp && d.hyp !== '*') hyp(d.hyp);
    if (Array.isArray(d.hyps)) d.hyps.forEach(hyp);
    return out;
  };
  const out = [];
  for (const seg of segs) {
    const coveredBy = prog.decls.filter(
      (d) => ACCOUNTING_OPS.has(d.op) && d.span && d.span.s < seg.e && seg.s < d.span.e
    );
    if (coveredBy.length === 0) continue;
    // Exact span-interval subtraction on OBS sentences (Bulma R6): text
    // inside the sentence but outside every covering span is adjacent
    // content no decl asserts, however it is attached — the witness
    // must see it even when the connector lists leave the sentence in
    // one piece. Whole-sentence OBS spans yield no interval here.
    const observedNames = coveredBy
      .filter((d) => d.op === 'OBS' && evNameById.get(d.ev))
      .map((d) => evNameById.get(d.ev));
    for (const piece of uncoveredObsIntervals(seg, coveredBy, names, observedNames, evNameById, X)) {
      out.push({
        segIdx: seg.idx,
        piece,
        materiality: { kind: 'material', reason: 'text outside the OBS span inside its sentence' },
        overlapping: coveredBy.filter((d) => d.span.s < piece.e && piece.s < d.span.e),
      });
    }
    // Rate-sentence tails (Bulma R7): a counterexample appended to a
    // PRIOR/LIK sentence is adjacent content no decl asserts, exactly
    // like text outside an OBS span.
    for (const piece of uncoveredRateIntervals(seg, coveredBy, X, prog, names)) {
      out.push({
        segIdx: seg.idx,
        piece,
        materiality: { kind: 'material', reason: 'text beyond the rate span inside its sentence' },
        overlapping: coveredBy.filter((d) => d.span.s < piece.e && piece.s < d.span.e),
      });
    }
    const pieces = subClauses(seg, names);
    if (pieces.length < 2) continue;
    for (const piece of pieces) {
      if (isInertAdjunctClause(piece.text, names)) continue;
      const m = sentenceMateriality(piece, names);
      if (m.kind !== 'material') continue;
      // same non-assertion exemption as kernel accounting: a noun-phrase
      // shard of a grounded clause is not adjacent asserted content
      if (!isAssertiveFragment(piece.text, names) &&
          !premiseDoubt(piece.text) && !probLikeLiteral(piece.text)) continue;
      const overlapping = coveredBy.filter(
        (d) => d.span.s < piece.e && piece.s < d.span.e
      );
      // A sub-piece lying inside (or overlapping) a DET_LINK /
      // COND_INDEP span is the license's own asserted clause, never an
      // unentailed residual — the splitter can shard a coordinated
      // license subject ("the lamp and buzzer ...") so the span start
      // lands in an earlier piece, but the span still binds the whole
      // clause it covers (Bulma R6 positive-control regression).
      if (overlapping.some((d) => d.op === 'DET_LINK' || d.op === 'COND_INDEP')) continue;
      // An interval-subtraction finding above already names this text.
      if (out.some((f) => f.segIdx === seg.idx && f.piece.s < piece.e && piece.s < f.piece.e)) {
        continue;
      }
      if (overlapping.length === 0) {
        out.push({ segIdx: seg.idx, piece, materiality: m, overlapping });
        continue;
      }
      // A primary-factor quote that includes its leading coordinator
      // ("and a cracked seal", "and 4/11 by Beta") starts just BEFORE
      // the sub-clause the coordinator splitter produced from it, so
      // the strict anchor test (span start inside the piece) misses
      // it. It still anchors the piece when ONLY whitespace/comma +
      // "and"/"or" separate the span start from the piece, the span
      // covers the piece's whole non-punctuation text, the piece
      // carries no premise doubt, and every declared name the piece
      // mentions is one this decl asserts. An OBS piece must
      // additionally carry no literal and no verb outside its asserted
      // names ("and the seal cracked", "and the buzzer fired alone"
      // stay residual). A numeric factor piece (PRIOR / LIK /
      // POSTERIOR_STATED / JOINT_LIK) must carry at most one
      // probability-like literal, equal to the decl's own value when
      // present, and no co-activation, dependency, concessive, or
      // solo/activation counterexample wording — its own rate verb
      // inside the covering span ("4/11 were handled by Beta") is the
      // decl's own predicate, already bound by literal ownership.
      const leadingCoordAnchored = (d) => {
        if (!['OBS', 'PRIOR', 'LIK', 'POSTERIOR_STATED', 'JOINT_LIK'].includes(d.op)) {
          return false;
        }
        if (!(d.span.s < piece.s)) return false;
        if (!/^[\s,]*(?:and|or)\s+$/i.test(X.slice(d.span.s, piece.s))) return false;
        const core = piece.text.trimEnd().replace(/[.!?;:,]+$/, '');
        if (d.span.e < piece.s + core.length) return false;
        const asserted = assertedNames(d);
        const mentioned = names.filter((n) => wordRe(n).test(piece.text));
        if (!mentioned.every((n) => asserted.has(n.toLowerCase()))) return false;
        if (premiseDoubt(piece.text)) return false;
        if (d.op === 'OBS') {
          if (probLikeLiteral(piece.text)) return false;
          const dNames = [...asserted];
          if (hasFiniteVerb(piece.text, dNames)) return false;
          if (hasNonfinitePredicate(maskEntityNames(piece.text, dNames))) return false;
          return true;
        }
        if (!d.value) return false;
        const lits = scanLiterals(piece.text).filter((l) => !/^\d+$/.test(l.text));
        if (lits.length > 1) return false;
        if (lits.length === 1 && !R.eq(lits[0].value, d.value)) return false;
        if (hasRateCoactMarker(piece.text)) return false;
        if (DEPENDENCY_WORDS.test(piece.text) || CONCESSIVE_MARKER.test(piece.text)) return false;
        if (/\b(?:solo|alone|activations?|co-?activ\w*|co-?occur\w*)\b/i.test(piece.text)) {
          return false;
        }
        return true;
      };
      const anchored = overlapping.filter(
        (d) => (d.span.s >= piece.s && d.span.s < piece.e) || leadingCoordAnchored(d)
      );
      // An integrated "with"-absolute can BE the overlapping decl's own
      // payload ("was observed with its lamp lit" — the OBS asserts
      // exactly the event state the with-clause states, of the same
      // item at the same time). It is residual only when it asserts an
      // entity no overlapping decl asserts. Parenthetical, coordinate,
      // comma and concessive slots are separate assertion slots and
      // get no such exemption.
      if (piece.via === 'with' || piece.via === 'without') {
        const captured = new Set();
        for (const d of overlapping) for (const n of assertedNames(d)) captured.add(n);
        const mentioned = names.filter((n) => wordRe(n).test(piece.text));
        if (mentioned.every((n) => captured.has(n.toLowerCase()))) continue;
      }
      if (anchored.length === 0 && isAssertiveFragment(piece.text, names)) {
        out.push({ segIdx: seg.idx, piece, materiality: m, overlapping });
      }
    }
  }
  return out;
}

/**
 * checkSentenceAccounting(X, prog) -> { ok:true, tags:[{idx,tag,covered}] }
 *                                | { ok:false, code:'UNACCOUNTED_MATERIAL_SENTENCE', detail }
 * The sentence-partition ledger is UNCONDITIONAL whenever X is supplied:
 *   1. SENT_TAG decls must cover every sentence index exactly once
 *      (indices 0..m-1, unique), and their spans must match the segmentation.
 *      A wholly absent ledger is an invalid ledger.
 *   2. Tag semantics are checked against the text, never trusted:
 *        QUERY         — the sentence must actually carry the query
 *        PREMISE       — some decl must ground the sentence
 *        NON_ASSERTED  — a NON_ASSERTED decl must cover the sentence
 *        CONTEXT_SCOPE — the tag must carry a CtxId the program declares
 *        FLUFF         — the sentence must not be material or the query
 *   3. Every material sentence is grounded by at least one premise decl
 *      span — a FLUFF-tagged material sentence or an untagged material
 *      sentence refuses identically.
 *   4. Grounding is per sub-clause: a decl covering only one clause of a
 *      compound sentence does not account for a material residual clause
 *      sharing the sentence (see unaccountedResidualClause).
 */
export function checkSentenceAccounting(X, prog) {
  const segs = sentences(X);
  const tags = prog.decls.filter((d) => d.op === 'SENT_TAG');
  const report = segs.map((seg) => ({
    idx: seg.idx,
    seg,
    tag: null,
    tagDecl: null,
    coveredBy: [],
    materiality: null,
  }));
  const fail = (detail, span = null) => ({
    ok: false,
    code: 'UNACCOUNTED_MATERIAL_SENTENCE',
    detail,
    span,
  });

  const seen = new Set();
  for (const t of tags) {
    if (!Number.isInteger(t.idx) || t.idx < 0 || t.idx >= segs.length) {
      return fail(`SENT_TAG idx ${t.idx} outside sentence range 0..${segs.length - 1}`);
    }
    if (seen.has(t.idx)) {
      return fail(`SENT_TAG idx ${t.idx} ledgered twice`);
    }
    seen.add(t.idx);
    if (!SENT_TAGS.has(t.tag)) {
      return fail(`SENT_TAG idx ${t.idx} has unknown tag "${t.tag}"`);
    }
    if (t.span && (t.span.s !== segs[t.idx].s || t.span.e !== segs[t.idx].e)) {
      return fail(
        `SENT_TAG idx ${t.idx} span [${t.span.s},${t.span.e}] does not match ` +
          `sentence bounds [${segs[t.idx].s},${segs[t.idx].e}]`
      );
    }
    report[t.idx].tag = t.tag;
    report[t.idx].tagDecl = t;
  }
  // unconditional coverage: every sentence index must be ledgered exactly
  // once whenever X is supplied — a wholly absent ledger fails here too
  if (seen.size !== segs.length) {
    const missing = report.filter((r) => !seen.has(r.idx)).map((r) => r.idx);
    return fail(
      `sentence partition incomplete: indices {${missing.join(',')}} have no SENT_TAG`
    );
  }

  // CtxIds the program actually uses — a CONTEXT_SCOPE tag must bind to one.
  // SENT_TAG decls do NOT count: a scope tag may not create its own context.
  const usedCtxs = new Set([prog.query.ctx]);
  for (const d of prog.decls) {
    if (d.ctx && d.op !== 'SENT_TAG') usedCtxs.add(d.ctx);
  }

  const names = entityNames(X, prog);
  const excl = new Set(names.map((n) => n.toLowerCase()));
  for (const r of report) {
    r.coveredBy = prog.decls.filter(
      (d) => ACCOUNTING_OPS.has(d.op) && d.span && d.span.s < r.seg.e && r.seg.s < d.span.e
    );
    r.materiality = sentenceMateriality(r.seg, names);
    // An unchecked-attribute remark about an unobserved, unqueried feature
    // is not dropped evidence: it grounds nothing the query needs. The
    // same holds for an unrecorded-tally remark in a zero-prior program
    // and for a qualitative sentence scoped to an unqueried stratum.
    r.irrelevantUnchecked = irrelevantUncheckedSeg(X, prog, r.seg);
    r.irrelevantUnrecordedPrior = irrelevantUnrecordedPriorSeg(X, prog, r.seg);
    r.irrelevantOtherStratum = irrelevantOtherStratumSeg(X, prog, r.seg);
    if ((r.irrelevantUnchecked || r.irrelevantUnrecordedPrior ||
         r.irrelevantOtherStratum) && r.materiality.kind === 'material') {
      r.materiality = { kind: 'fluff' };
    }
    const t = r.seg.text.trim().slice(0, 90);
    switch (r.tag) {
      case 'QUERY':
        if (r.materiality.kind !== 'query') {
          return fail(
            `sentence ${r.idx} carries no query but is tagged QUERY: "${t}"`,
            { s: r.seg.s, e: r.seg.e }
          );
        }
        break;
      case 'PREMISE':
        if (r.coveredBy.length === 0 && !r.irrelevantUnchecked &&
            !r.irrelevantUnrecordedPrior && !r.irrelevantOtherStratum) {
          return fail(
            `sentence ${r.idx} tagged PREMISE but grounded by no decl: "${t}"`,
            { s: r.seg.s, e: r.seg.e }
          );
        }
        break;
      case 'NON_ASSERTED':
        if (!r.coveredBy.some((d) => d.op === 'NON_ASSERTED')) {
          return fail(
            `sentence ${r.idx} tagged NON_ASSERTED but no NON_ASSERTED decl covers it: "${t}"`,
            { s: r.seg.s, e: r.seg.e }
          );
        }
        break;
      case 'CONTEXT_SCOPE': {
        const bound = r.tagDecl.ctx ?? null;
        if (!bound || !usedCtxs.has(bound)) {
          return fail(
            `sentence ${r.idx} tagged CONTEXT_SCOPE but carries no declared CtxId: "${t}"`,
            { s: r.seg.s, e: r.seg.e }
          );
        }
        // the bound context must be grounded in the passage text itself:
        // the tagged sentence must carry the population label(s) whose
        // canonical CtxId it binds ("On the night-shift, ..." grounds
        // ctx_night_shift). A scope tag on an unlabeled sentence — or one
        // whose labels name a different context — invents its own context.
        const grounded = canonicalCtxId(r.seg.text, excl);
        if (grounded !== bound) {
          return fail(
            `sentence ${r.idx} tagged CONTEXT_SCOPE ${bound} but the text grounds ` +
              `${grounded ?? 'no context'}: "${t}"`,
            { s: r.seg.s, e: r.seg.e }
          );
        }
        break;
      }
      case 'FLUFF':
        break;
      default:
        break;
    }
    if (r.materiality.kind === 'query' && r.tag !== 'QUERY') {
      return fail(`sentence ${r.idx} carries the query but is tagged ${r.tag}`, {
        s: r.seg.s,
        e: r.seg.e,
      });
    }
    if (r.materiality.kind === 'material' && r.tag === 'FLUFF') {
      return fail(
        `sentence ${r.idx} is material (${r.materiality.reason}) but tagged FLUFF: "${t}"`,
        { s: r.seg.s, e: r.seg.e }
      );
    }
    if (r.coveredBy.length === 0 && r.materiality.kind === 'material') {
      return fail(
        `sentence ${r.idx} is material (${r.materiality.reason}) but grounded by no decl: "${t}"`,
        { s: r.seg.s, e: r.seg.e }
      );
    }
    // exact OBS span-interval subtraction (Bulma R6): on a sentence
    // covered only by OBS/DET_LINK/COND_INDEP spans, every interval of
    // the sentence outside the covering spans must be inert — no
    // connector list mediates this check
    if (r.coveredBy.length > 0) {
      const evNames = eventNameMap(X, prog);
      const observedNames = r.coveredBy
        .filter((d) => d.op === 'OBS' && evNames.get(d.ev))
        .map((d) => evNames.get(d.ev));
      const uncovered = uncoveredObsIntervals(r.seg, r.coveredBy, names, observedNames, evNames, X);
      if (uncovered.length > 0) {
        const c = uncovered[0].text.trim().replace(/\s+/g, ' ').slice(0, 90);
        return fail(
          `sentence ${r.idx} carries material text outside its OBS span ` +
            `grounded by no decl: "${c}"`,
          { s: uncovered[0].s, e: uncovered[0].e }
        );
      }
      // rate-sentence tail accounting (Bulma R7): text beyond the
      // rightmost PRIOR/LIK/JOINT_LIK/PART_EXHAUSTIVE span (or under a
      // concessive leading frame) must be inert on its own — a
      // counterexample appended to a rate refuses like one appended
      // to an observation
      const rateBad = uncoveredRateIntervals(r.seg, r.coveredBy, X, prog, names);
      if (rateBad.length > 0) {
        const c = rateBad[0].text.trim().replace(/\s+/g, ' ').slice(0, 90);
        return fail(
          `sentence ${r.idx} carries material text beyond its rate span ` +
            `grounded by no decl: "${c}"`,
          { s: rateBad[0].s, e: rateBad[0].e }
        );
      }
    }
    // per-clause coverage: an accounted (or query) sentence may still
    // smuggle an ungrounded material clause past a decl that grounds
    // only its sister clause
    if (r.coveredBy.length > 0 || r.tag === 'QUERY') {
      const residual = unaccountedResidualClause(r, names);
      if (residual) {
        const c = residual.piece.text.trim().slice(0, 90);
        return fail(
          `sentence ${r.idx} carries a material sub-clause ` +
            `(${residual.materiality.reason}) grounded by no decl: "${c}"`,
          { s: residual.piece.s, e: residual.piece.e }
        );
      }
    }
  }
  return { ok: true, tags: report };
}

/**
 * autoSentTags(X, prog, usedCtxs?) -> SENT_TAG decl objects covering every
 * sentence. The compiler-owned ledger derivation used by the live extractor
 * and the synthetic generator: query sentences -> QUERY, NON_ASSERTED-covered
 * -> NON_ASSERTED, material -> PREMISE, unlabeled inert -> FLUFF — and a
 * sentence whose only content is a population label the program actually
 * uses ("On the night-shift:") -> CONTEXT_SCOPE bound to that CtxId. Never
 * trusts the model's own tags — the checker re-verifies them anyway.
 */
export function autoSentTags(X, prog, usedCtxs = null) {
  const segs = sentences(X);
  const names = entityNames(X, prog);
  const excl = new Set(names.map((n) => n.toLowerCase()));
  const ctxSet =
    usedCtxs ??
    new Set(
      prog.decls.map((d) => d.ctx).filter(Boolean).concat([prog.query?.ctx].filter(Boolean))
    );
  const naSpans = prog.decls.filter((d) => d.op === 'NON_ASSERTED' && d.span).map((d) => d.span);
  return segs.map((seg) => {
    const covered = prog.decls.some(
      (d) => ACCOUNTING_OPS.has(d.op) && d.span && d.span.s < seg.e && seg.s < d.span.e
    );
    const naCovered = naSpans.some((sp) => sp.s < seg.e && seg.s < sp.e);
    const m = sentenceMateriality(seg, names);
    let tag;
    let ctx = null;
    if ((irrelevantUncheckedSeg(X, prog, seg) ||
         irrelevantUnrecordedPriorSeg(X, prog, seg) ||
         irrelevantOtherStratumSeg(X, prog, seg)) && m.kind !== 'query') {
      tag = 'FLUFF';
    } else if (m.kind === 'query') {
      tag = 'QUERY';
    } else if (naCovered) {
      tag = 'NON_ASSERTED';
    } else if (covered || m.kind === 'material') {
      tag = 'PREMISE';
    } else {
      tag = 'FLUFF';
      // pure scope-establishing sentence: an unlabeled-population FLUFF
      // stays FLUFF, but a sentence carrying a population label the program
      // uses is a context-scope declaration and binds that CtxId
      const cid = canonicalCtxId(seg.text, excl);
      if (cid && ctxSet.has(cid)) {
        tag = 'CONTEXT_SCOPE';
        ctx = cid;
      }
    }
    return { op: 'SENT_TAG', idx: seg.idx, tag, ctx, span: { s: seg.s, e: seg.e } };
  });
}
