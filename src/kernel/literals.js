// prism-kernel :: numeric literal scanner.
// Finds and parses every numeric literal inside a text span, in document order:
//   "5/8", "62.5%", "0.625", "80", "five eighths", "seven twelfths",
//   "a third", "one half", "half", "two fifths", "3 in 10", "3 out of 5".
// Deterministic. Used for pointer-grounding checks (slot 2 of the B=4 gate)
// and by the synth renderer to round-trip word-form rationals.

import * as R from './rational.js';

const NUM_WORDS = new Map([
  ['a', 1n], ['an', 1n], ['zero', 0n], ['one', 1n], ['two', 2n], ['three', 3n], ['four', 4n],
  ['five', 5n], ['six', 6n], ['seven', 7n], ['eight', 8n], ['nine', 9n],
  ['ten', 10n], ['eleven', 11n], ['twelve', 12n], ['thirteen', 13n],
  ['fourteen', 14n], ['fifteen', 15n], ['sixteen', 16n], ['seventeen', 17n],
  ['eighteen', 18n], ['nineteen', 19n], ['twenty', 20n],
]);

const DEN_WORDS = new Map([
  ['half', 2n], ['halves', 2n],
  ['third', 3n], ['thirds', 3n],
  ['quarter', 4n], ['quarters', 4n], ['fourth', 4n], ['fourths', 4n],
  ['fifth', 5n], ['fifths', 5n],
  ['sixth', 6n], ['sixths', 6n],
  ['seventh', 7n], ['sevenths', 7n],
  ['eighth', 8n], ['eighths', 8n],
  ['ninth', 9n], ['ninths', 9n],
  ['tenth', 10n], ['tenths', 10n],
  ['eleventh', 11n], ['elevenths', 11n],
  ['twelfth', 12n], ['twelfths', 12n],
  ['thirteenth', 13n], ['thirteenths', 13n],
  ['fourteenth', 14n], ['fourteenths', 14n],
  ['fifteenth', 15n], ['fifteenths', 15n],
  ['sixteenth', 16n], ['sixteenths', 16n],
  ['seventeenth', 17n], ['seventeenths', 17n],
  ['eighteenth', 18n], ['eighteenths', 18n],
  ['nineteenth', 19n], ['nineteenths', 19n],
  ['twentieth', 20n], ['twentieths', 20n],
]);

const NUMWORD_ALT = [...NUM_WORDS.keys()].join('|');
const DENWORD_ALT = [...DEN_WORDS.keys()].join('|');
// Frequency-ratio cardinals: digits or number words (never the articles
// 'a'/'an', which would turn ordinary prose into ratios).
const CARD_ALT = [...NUM_WORDS.keys()].filter((w) => w !== 'a' && w !== 'an').join('|');
const CARD = `(?:\\d+|${CARD_ALT})`;
// Words that can never be the simple noun inside "<card> <noun> in <card>":
// number/denominator words and function words. Anything else is treated as
// a counted noun ("three cards in ten").
const FREQ_NOUN_STOP = new Set([
  ...NUM_WORDS.keys(), ...DEN_WORDS.keys(),
  'in', 'out', 'of', 'every', 'the', 'and', 'or', 'to', 'per', 'is', 'are',
  'was', 'were', 'on', 'at', 'by', 'for', 'with', 'from', 'each',
]);

function cardValue(tok) {
  const t = String(tok).toLowerCase();
  if (/^\d+$/.test(t)) return BigInt(t);
  return NUM_WORDS.get(t) ?? null;
}

// A 4-digit token in the modern-year range is a date, not a frequency
// count ("3 in 2024" is a calendar statement, never a ratio).
const isYearToken = (tok) =>
  /^\d{4}$/.test(tok) && BigInt(tok) >= 1900n && BigInt(tok) <= 2099n;

function freqRational(numTok, denTok) {
  if (isYearToken(numTok) || isYearToken(denTok)) return null;
  const n = cardValue(numTok);
  const d = cardValue(denTok);
  if (n === null || d === null || d <= 0n || n < 0n) return null;
  return R.make(n, d);
}

// Ordered alternation; longer/more-specific patterns first. Frequency forms
// use horizontal whitespace only, so a ratio never spans sentences/lines.
const LITERAL_RE = new RegExp(
  [
    `\\b(${NUMWORD_ALT})\\s+(${DENWORD_ALT})\\b`,                 // five eighths
    `\\b(${CARD})[ \\t]+([a-z]+(?:-[a-z]+)?(?:[ \\t]+[a-z]+(?:-[a-z]+)?)?)[ \\t]+in[ \\t]+(${CARD})\\b`, // three cards in ten
    `\\b(${CARD})[ \\t]+in[ \\t]+(${CARD})\\b`,                   // one in ten / 3 in 10
    `\\b(${CARD})[ \\t]+out[ \\t]+of[ \\t]+(?:every[ \\t]+)?(${CARD})\\b`, // three out of (every) ten
    `\\b(${DENWORD_ALT})\\b`,                                      // bare "half"/"third"
    '\\b(\\d+)\\s*\\/\\s*(\\d+)\\b',                               // 5/8
    '\\b(\\d+(?:\\.\\d+)?)\\s*%',                                  // 62.5%
    '\\b\\d+\\.\\d+\\b',                                           // 0.625
    '\\b\\d+\\b',                                                  // 80
  ].join('|'),
  'gi'
);

// A literal site: {s, e, text, value:{n,d}} with value a normalized rational.
export function scanLiterals(text) {
  const sites = [];
  const re = new RegExp(LITERAL_RE.source, 'gi');
  let m;
  while ((m = re.exec(text)) !== null) {
    const full = m[0];
    const s = m.index;
    const e = s + full.length;
    const value = parseLiteral(full);
    if (value !== null) sites.push({ s, e, text: full, value });
  }
  return sites;
}

// Parse a single literal substring to a rational, or null.
export function parseLiteral(str) {
  const orig = String(str).trim();
  const s = orig.toLowerCase();
  let m;
  if ((m = new RegExp(`^(${NUMWORD_ALT})\\s+(${DENWORD_ALT})$`).exec(s))) {
    const n = NUM_WORDS.get(m[1]);
    const d = DEN_WORDS.get(m[2]);
    return R.make(n, d);
  }
  // worded frequency ratios: "<card> [noun(s)] in <card>",
  // "<card> out of [every] <card>". The counted noun (1-2 simple lowercase
  // words, exactly as written) may sit between numerator and "in".
  if ((m = new RegExp(`^(${CARD})\\s+(?:([a-z][a-z-]*(?:\\s+[a-z][a-z-]*)?)\\s+)?in\\s+(${CARD})$`).exec(s))) {
    if (m[2] !== undefined) {
      const nounToks = m[2].split(/\s+/);
      if (nounToks.length < 1 || nounToks.length > 2) return null;
      if (nounToks.some((t) => t.split('-').some((p) => FREQ_NOUN_STOP.has(p)))) return null;
      // the noun must be lowercase as written (a capitalised token is a
      // name/heading, not a counted common noun)
      const origToks = orig.split(/\s+/);
      const origNoun = origToks.slice(1, 1 + nounToks.length);
      if (origNoun.some((t) => t !== t.toLowerCase())) return null;
    }
    return freqRational(m[1], m[3]);
  }
  if ((m = new RegExp(`^(${CARD})\\s+out\\s+of\\s+(?:every\\s+)?(${CARD})$`).exec(s))) {
    return freqRational(m[1], m[2]);
  }
  if (DEN_WORDS.has(s)) return R.make(1n, DEN_WORDS.get(s));
  if ((m = /^(\d+)\s*\/\s*(\d+)$/.exec(s))) {
    if (BigInt(m[2]) === 0n) return null;
    return R.make(BigInt(m[1]), BigInt(m[2]));
  }
  if ((m = /^(\d+(?:\.\d+)?)\s*%$/.exec(s))) {
    const v = m[1].includes('.') ? R.parseDecimal(m[1]) : R.fromInt(BigInt(m[1]));
    return R.div(v, R.fromInt(100n));
  }
  if (/^\d+\.\d+$/.test(s)) return R.parseDecimal(s);
  if (/^\d+$/.test(s)) return R.fromInt(BigInt(s));
  return null;
}

// Literals strictly inside span [s,e) of text.
export function literalsInSpan(text, span) {
  return scanLiterals(text).filter((l) => l.s >= span.s && l.e <= span.e);
}

// Minimal literal site inside a span whose value equals `value`, else null.
export function literalAtSpanForValue(text, span, value) {
  const inside = literalsInSpan(text, span);
  return inside.find((l) => R.eq(l.value, value)) ?? null;
}
