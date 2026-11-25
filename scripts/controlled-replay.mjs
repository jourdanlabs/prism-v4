// controlled-replay :: replays Bulma's V2-gate published fixtures against the
// fixed public gate. Every fixture uses the published passage as X and the
// published bytecode, with a compiler-derived sentence ledger appended
// (autoSentTags — exactly what the wired live extractor now emits).
//   usage: node scripts/controlled-replay.mjs [--json]
import { solve } from '../src/kernel/solve.js';
import { solveCredal } from '../src/kernel/credal.js';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags } from '../src/kernel/sentences.js';
import * as R from '../src/kernel/rational.js';

const fmt = (v) => (v ? `${v.n}/${v.d}` : null);

/** Append a compiler-derived SENT_TAG ledger before ASK_PROB. */
function withLedger(X, bytecode) {
  const prog = toProgram(bytecode);
  const tags = autoSentTags(X, prog);
  const lines = tags.map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return bytecode.replace(/(^|\n)(ASK_PROB)/, `$1${lines.join('\n')}\n$2`);
}

const CASES = [];

// ---------------------------------------------------------------------------
// V2 published fixtures (bytecode verbatim from BULMA-GATE-PRISM-V2-CREDAL)
// ---------------------------------------------------------------------------

const X_IMPOSSIBLE =
  'Every capsule is either Elm or Fir. Elm accounts for 40% of capsules and Fir accounts for 60%. The lamp lights for 0% of Elm capsules and 0% of Fir capsules. Among lamp-lit capsules, 80% are Elm. This capsule was observed with its lamp lit. What is the probability it is Elm?';
const BC_IMPOSSIBLE = `HYP h0 [24,27]
HYP h1 [31,34]
PART_EXHAUSTIVE (h0 h1) [24,34]
EVENT e0 [99,103]
PRIOR h0 2/5 ctx0 [53,56]
PRIOR h1 3/5 ctx0 [90,93]
LIK e0 POS GIVEN h0 0/1 ctx0 [115,117]
LIK e0 POS GIVEN h1 0/1 ctx0 [138,140]
POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 [183,186]
OBS e0 POS ctx0 [196,239]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
CASES.push({
  id: 'stated-posterior-impossible-evidence',
  X: X_IMPOSSIBLE,
  bc: BC_IMPOSSIBLE,
  expect: { refuse: 'DEGENERATE_PARTITION' },
  was: '4/5',
});

const X_POSSZERO =
  'Every capsule is either Elm or Fir. Elm accounts for 40% of capsules and Fir accounts for 60%. Among lamp-lit capsules, 80% are Elm. This capsule was observed with its lamp lit. What is the probability it is Elm?';
const BC_POSSZERO = `HYP h0 [24,27]
HYP h1 [31,34]
PART_EXHAUSTIVE (h0 h1) [24,34]
EVENT e0 [101,105]
PRIOR h0 2/5 ctx0 [53,56]
PRIOR h1 3/5 ctx0 [90,93]
POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0 [120,123]
OBS e0 POS ctx0 [133,176]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
CASES.push({
  id: 'stated-posterior-possible-zero',
  X: X_POSSZERO,
  bc: BC_POSSZERO,
  expect: { refuse: 'ZERO_SUPPORT_POSSIBLE' },
  was: '4/5',
});

// context guard: the ONLY stated posterior lives at a different CtxId
const BC_CTX_GUARD = BC_POSSZERO.replace('POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx0', 'POSTERIOR_STATED h0 GIVEN e0 POS 4/5 ctx_night');
CASES.push({
  id: 'stated-posterior-context-guard',
  X: X_POSSZERO,
  bc: BC_CTX_GUARD,
  expect: { refuse: 'CONTEXT_MISMATCH' },
  was: '4/5',
});

const X_QUIET =
  'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. The lamp lights for 80% of Ash panels and 20% of Fir panels. The buzzer sounds for 80% of Ash panels and 20% of Fir panels. The room was quiet. This panel was observed with its lamp lit. What is the probability its buzzer sounded?';
const BC_QUIET = `HYP h0 [22,25]
HYP h1 [29,32]
PART_EXHAUSTIVE (h0 h1) [22,32]
EVENT e0 [95,99]
EVENT e1 [156,162]
PRIOR h0 2/5 ctx0 [51,54]
PRIOR h1 3/5 ctx0 [86,89]
LIK e0 POS GIVEN h0 4/5 ctx0 [111,114]
LIK e0 POS GIVEN h1 1/5 ctx0 [133,136]
LIK e1 POS GIVEN h0 4/5 ctx0 [174,177]
LIK e1 POS GIVEN h1 1/5 ctx0 [196,199]
OBS e0 POS ctx0 [235,276]
DET_LINK e0 EQUIV e1 ctx0 [215,233]
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
CASES.push({
  id: 'invented-link-bound-to-quiet-room',
  X: X_QUIET,
  bc: BC_QUIET,
  expect: { refuse: true }, // license invalid -> unlicensed pair; must NOT release 1/1
  was: '1/1',
});

const X_RELAY =
  'Every panel is either Elm or Fir. Elm accounts for 40% of panels and Fir accounts for 60%. The lamp lights for 90% of Elm panels and 10% of Fir panels. The buzzer sounds for 90% of Elm panels and 10% of Fir panels. The two outputs share a relay. These outputs therefore always activate together. Under each panel type, room temperature and humidity are independent. This panel was observed with its lamp lit. What is the probability its buzzer sounded?';
const BC_RELAY_MISLABELED = `HYP h0 [22,25]
HYP h1 [29,32]
PART_EXHAUSTIVE (h0 h1) [22,32]
EVENT e0 [95,99]
EVENT e1 [156,162]
PRIOR h0 2/5 ctx0 [51,54]
PRIOR h1 3/5 ctx0 [86,89]
LIK e0 POS GIVEN h0 9/10 ctx0 [111,114]
LIK e0 POS GIVEN h1 1/10 ctx0 [133,136]
LIK e1 POS GIVEN h0 9/10 ctx0 [174,177]
LIK e1 POS GIVEN h1 1/10 ctx0 [196,199]
COND_INDEP (e0 e1) GIVEN * ctx0 [296,364]
OBS e0 POS ctx0 [366,407]
SENT_TAG 0 PREMISE [0,33]
SENT_TAG 1 PREMISE [33,90]
SENT_TAG 2 PREMISE [90,151]
SENT_TAG 3 PREMISE [151,214]
SENT_TAG 4 FLUFF [214,245]
SENT_TAG 5 FLUFF [245,295]
SENT_TAG 6 PREMISE [295,365]
SENT_TAG 7 PREMISE [365,408]
SENT_TAG 8 QUERY [408,452]
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
CASES.push({
  id: 'anaphoric-link-wrong-license-tagged-fluff',
  X: X_RELAY,
  bc: BC_RELAY_MISLABELED, // ledger already in fixture — do not regenerate
  ledger: false,
  expect: { refuse: true }, // was 11/14 — material relay sentences masked FLUFF + unrelated license
  was: '11/14',
});

// faithful variant of the same text: two EQUIV links bound to the two relay
// sentences, the temperature/humidity sentence ledgered FLUFF (it asserts a
// relation about undeclared quantities and is genuinely inert for this program)
const BC_RELAY_FAITHFUL = `HYP h0 [22,25]
HYP h1 [29,32]
PART_EXHAUSTIVE (h0 h1) [22,32]
EVENT e0 [95,99]
EVENT e1 [156,162]
PRIOR h0 2/5 ctx0 [51,54]
PRIOR h1 3/5 ctx0 [86,89]
LIK e0 POS GIVEN h0 9/10 ctx0 [111,114]
LIK e0 POS GIVEN h1 1/10 ctx0 [133,136]
LIK e1 POS GIVEN h0 9/10 ctx0 [174,177]
LIK e1 POS GIVEN h1 1/10 ctx0 [196,199]
DET_LINK e0 EQUIV e1 ctx0 [214,245]
DET_LINK e0 EQUIV e1 ctx0 [245,295]
OBS e0 POS ctx0 [366,407]
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
CASES.push({
  id: 'anaphoric-link-faithful-ledger',
  X: X_RELAY,
  bc: BC_RELAY_FAITHFUL,
  expect: { answer: '1/1' },
});

// ---------------------------------------------------------------------------
// R4 gate replays (BULMA-GATE-PRISM-V2-R4-37defe1): substring cue + deny-list
// released 1/1 on each of these. R5's anchored full-sentence grammar must
// refuse all four wrong-release forms while keeping both controls.
// ---------------------------------------------------------------------------
const X_R4 = (link) =>
  'Every panel is either Ash or Fir. Ash accounts for 40% of panels and Fir accounts for 60%. ' +
  'The lamp lights for 40% of Ash panels and 20% of Fir panels. ' +
  'The buzzer sounds for 40% of Ash panels and 20% of Fir panels. ' +
  `${link} ` +
  'This panel was observed with its lamp lit. ' +
  'What is the probability its buzzer sounded?';
const bcR4 = (X, linkSentence) => {
  const at = (s) => {
    const i = X.indexOf(s);
    return `[${i},${i + s.length}]`;
  };
  const bz = X.indexOf('The buzzer sounds');
  return `HYP h0 ${at('Ash')}
HYP h1 ${at('Fir')}
PART_EXHAUSTIVE (h0 h1) ${at('Ash or Fir')}
EVENT e0 ${at('lamp')}
EVENT e1 ${at('buzzer')}
PRIOR h0 2/5 ctx0 ${at('40% of panels')}
PRIOR h1 3/5 ctx0 ${at('60%.')}
LIK e0 POS GIVEN h0 2/5 ctx0 ${at('40% of Ash')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Fir')}
LIK e1 POS GIVEN h0 2/5 ctx0 [${X.indexOf('40% of Ash', bz)},${X.indexOf('40% of Ash', bz) + 10}]
LIK e1 POS GIVEN h1 1/5 ctx0 [${X.indexOf('20% of Fir', bz)},${X.indexOf('20% of Fir', bz) + 10}]
DET_LINK e0 EQUIV e1 ctx0 ${at(linkSentence)}
OBS e0 POS ctx0 ${at('was observed with its lamp lit')}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
};
for (const [id, link, expect] of [
  [
    'r4-subset-universal-wrong-release',
    'On a subset of panels, the lamp and buzzer always activate together, and elsewhere both solo patterns occur.',
    { refuse: true }, // was 1/1 — subset clause wraps the 'always' cue
  ],
  [
    'r4-existential-wiring-wrong-release',
    'At least one panel has a lamp and buzzer that are wired to the same switch.',
    { refuse: true }, // was 1/1 — existential quantifier wraps the wiring cue
  ],
  [
    'r4-modal-wiring-wrong-release',
    'It is possible that the lamp and buzzer are wired to the same switch on every panel.',
    { refuse: true }, // was 1/1 — modal frame wraps the universal claim
  ],
  [
    'r4-hypothesis-scoped-wrong-release',
    'The lamp and buzzer always activate together on Ash panels.',
    { refuse: true }, // was 1/1 — Ash-only link is not population-wide
  ],
  [
    'r4-affirmative-control',
    'The lamp and buzzer always activate together.',
    { answer: '1/1' },
  ],
  // ---------------------------------------------------------------
  // R5 gate replays (BULMA-GATE-PRISM-V2-R5-3e0f168): the every/all
  // quantifier head accepted ANY trailing noun phrase — adjectives,
  // prepositional restrictions, exceptions, and foreign head nouns all
  // slipped through as "universal". R6 grounds the quantified noun to
  // the queried population (panel) exactly: no wildcards.
  // ---------------------------------------------------------------
  [
    'r5-qualified-universal-every-with',
    'On every panel with a common relay, the lamp and buzzer always activate together.',
    { refuse: true }, // was 1/1 — prepositional restriction inside the quantified domain
  ],
  [
    'r5-qualified-universal-all-intact',
    'The lamp and buzzer always activate together on all intact panels.',
    { refuse: true }, // was 1/1 — adjective restricts the quantified domain
  ],
  [
    'r5-both-hyp-qualified',
    'On Ash and Fir panels fitted with a common relay, the lamp and buzzer always activate together.',
    { refuse: true }, // was 1/1 — names every hypothesis but carries a trailing restriction
  ],
  [
    'r5-explicit-exception',
    'On every panel except damaged panels, the lamp and buzzer always activate together.',
    { refuse: true }, // was 1/1 — explicit exception inside the quantified domain
  ],
  [
    'r5-possessive-prototype',
    "Every prototype's lamp and buzzer are wired to the same switch.",
    { refuse: true }, // was 1/1 — 'prototype' is not the queried population noun
  ],
  // unseen variants of the same wildcard hole
  [
    'r6-scope-every-standard',
    'On every standard panel, the lamp and buzzer always activate together.',
    { refuse: true },
  ],
  [
    'r6-scope-without-defects',
    'The lamp and buzzer always activate together on all panels without defects.',
    { refuse: true },
  ],
  [
    'r6-scope-passes-inspection',
    'On every panel that passes inspection, the lamp and buzzer always activate together.',
    { refuse: true },
  ],
  [
    'r6-scope-each-undamaged',
    'On each undamaged panel, the lamp and buzzer always activate together.',
    { refuse: true },
  ],
  [
    'r6-possessive-tested-unit',
    "Each tested unit's lamp and buzzer are wired to the same switch.",
    { refuse: true },
  ],
  // population-grounded controls that must still release
  [
    'r6-possessive-panel-control',
    "Every panel's lamp and buzzer are wired to the same switch.",
    { answer: '1/1' },
  ],
  [
    'r6-scope-all-panels-control',
    'The lamp and buzzer always activate together on all panels.',
    { answer: '1/1' },
  ],
  [
    'r6-scope-for-all-panels-control',
    'For all panels, the lamp and buzzer always activate together.',
    { answer: '1/1' },
  ],
  [
    'r6-scope-every-hyp-panel-control',
    'On every Ash and Fir panel, the lamp and buzzer always activate together.',
    { answer: '1/1' },
  ],
]) {
  const X = X_R4(link);
  CASES.push({ id, X, bc: bcR4(X, link), expect });
}

// ---------------------------------------------------------------------------
// R3 anaphoric population replay (verbatim from BULMA-GATE-PRISM-PHASE0-R3)
// ---------------------------------------------------------------------------
const X_DAYNIGHT =
  'Every day-shift item is either Pebble or Moss. On the day-shift, Pebble accounts for 25% of items and Moss accounts for 75%. On the night-shift, the alarm sounds for 90% of Pebble items and 30% of Moss items. A day-shift item was observed with its alarm sounding. What is the probability this item is Pebble?';
const BC_DAYNIGHT = `HYP h0 [31,37]
HYP h1 [41,45]
PART_EXHAUSTIVE (h0 h1) [31,45]
PRIOR h0 1/4 ctx0 [85,88]
PRIOR h1 3/4 ctx0 [120,123]
EVENT e0 [149,154]
LIK e0 POS GIVEN h0 9/10 ctx0 [166,169]
LIK e0 POS GIVEN h1 3/10 ctx0 [190,193]
OBS e0 POS ctx0 [209,262]
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;
CASES.push({
  id: 'lowercase-anaphoric-population',
  X: X_DAYNIGHT,
  bc: BC_DAYNIGHT,
  expect: { refuse: 'CONTEXT_MISMATCH' },
  was: '1/2',
});

// same-shift control — genuinely same-population rates make 1/2 valid
const X_DAYSAME = X_DAYNIGHT.replace('On the night-shift, the alarm sounds', 'On the day-shift, the alarm sounds');
CASES.push({
  id: 'same-population-control',
  X: X_DAYSAME,
  bc: BC_DAYNIGHT,
  expect: { answer: '1/2' },
});

// ---------------------------------------------------------------------------
// raw credal false point (published degenerate-pivot case) — 0/1 released,
// oracle interval [0/1, 4/9]
// ---------------------------------------------------------------------------
const BC_FALSEPOINT = `HYP h0 [0,1]
HYP h1 [0,1]
EVENT e0 [0,1]
EVENT e1 [0,1]
PART_EXHAUSTIVE (h0 h1) [0,1]
LIK e0 POS GIVEN h0 0/1 q [0,1]
POSTERIOR_STATED h0 GIVEN e1 NEG 5/9 q [0,1]
JOINT_LIK (e0:POS e1:POS) GIVEN h0 0/1 q [0,1]
JOINT_LIK (e0:POS e1:POS) GIVEN h1 0/1 q [0,1]
ASK_PROB e0 GIVEN () q
END`;
CASES.push({
  id: 'raw-credal-false-point',
  bc: BC_FALSEPOINT,
  credalOnly: true,
  expect: { refuse: 'UNDERDETERMINED_INTERVAL', interval: ['0/1', '4/9'] },
  was: '0/1',
});

// ---------------------------------------------------------------------------
// ledger-validity battery on a solvable passage
// ---------------------------------------------------------------------------
const X_BASE =
  'Every capsule is either Elm or Fir. Elm accounts for 40% of capsules and Fir accounts for 60%. The lamp lights for 80% of Elm capsules and 20% of Fir capsules. The room was quiet. This capsule was observed with its lamp lit. What is the probability it is Elm?';
const span = (X, needle) => {
  const s = X.indexOf(needle);
  return `[${s},${s + needle.length}]`;
};
const BC_BASE = `HYP h0 ${span(X_BASE, 'Elm')}
HYP h1 ${span(X_BASE, 'Fir')}
PART_EXHAUSTIVE (h0 h1) ${span(X_BASE, 'Elm or Fir')}
EVENT e0 ${span(X_BASE, 'lamp')}
PRIOR h0 2/5 ctx0 ${span(X_BASE, '40%')}
PRIOR h1 3/5 ctx0 ${span(X_BASE, '60%')}
LIK e0 POS GIVEN h0 4/5 ctx0 ${span(X_BASE, '80%')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${span(X_BASE, '20%')}
OBS e0 POS ctx0 ${span(X_BASE, 'observed with its lamp lit')}
ASK_PROB h0 GIVEN (e0:POS) ctx0
END`;

const tagsFor = (X, bc) => autoSentTags(X, toProgram(bc));
const ledgerLines = (tags) =>
  tags.map((t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`).join('\n');
const ledgerOf = (X, bc) => bc.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(tagsFor(X, bc))}\n$2`);

// answer: 0.8*0.4/(0.8*0.4+0.2*0.6) = 8/11
CASES.push({ id: 'sentence-complete-control', X: X_BASE, bc: ledgerOf(X_BASE, BC_BASE), ledger: false, expect: { answer: '8/11' } });
CASES.push({ id: 'sentence-no-ledger-at-all', X: X_BASE, bc: BC_BASE, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' }, was: '8/11' });
{
  // partial ledger: drop one tag
  const tags = tagsFor(X_BASE, BC_BASE);
  const partial = tags.filter((t) => t.idx !== 1);
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(partial)}\n$2`);
  CASES.push({ id: 'sentence-partial-ledger', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' } });
}
{
  // duplicate index
  const tags = tagsFor(X_BASE, BC_BASE);
  const dup = [...tags, { ...tags[0] }];
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(dup)}\n$2`);
  CASES.push({ id: 'sentence-duplicate-index', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' } });
}
{
  // shifted span on one tag
  const tags = tagsFor(X_BASE, BC_BASE).map((t) =>
    t.idx === 0 ? { ...t, span: { s: t.span.s + 1, e: t.span.e } } : t
  );
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(tags)}\n$2`);
  CASES.push({ id: 'sentence-shifted-span', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' } });
}
{
  // FLUFF masks a numeric premise
  const tags = tagsFor(X_BASE, BC_BASE).map((t) => (t.idx === 1 ? { ...t, tag: 'FLUFF' } : t));
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(tags)}\n$2`);
  CASES.push({ id: 'sentence-fluff-masks-priors', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' } });
}
{
  // CONTEXT_SCOPE tag with no CtxId binding
  const tags = tagsFor(X_BASE, BC_BASE).map((t) => (t.idx === 3 ? { ...t, tag: 'CONTEXT_SCOPE', ctx: null } : t));
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(tags)}\n$2`);
  CASES.push({ id: 'sentence-unbound-context-tag', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' }, was: '8/11' });
}
{
  // PREMISE tag on the genuine filler sentence (no decl covers it)
  const tags = tagsFor(X_BASE, BC_BASE).map((t) => (t.tag === 'FLUFF' ? { ...t, tag: 'PREMISE' } : t));
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(tags)}\n$2`);
  CASES.push({ id: 'sentence-ungrounded-premise-tag', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' }, was: '8/11' });
}
{
  // QUERY tag on a non-query sentence
  const tags = tagsFor(X_BASE, BC_BASE).map((t) => (t.tag === 'FLUFF' ? { ...t, tag: 'QUERY' } : t));
  const bc = BC_BASE.replace(/(^|\n)(ASK_PROB)/, `$1${ledgerLines(tags)}\n$2`);
  CASES.push({ id: 'sentence-spurious-query-tag', X: X_BASE, bc, ledger: false, expect: { refuse: 'UNACCOUNTED_MATERIAL_SENTENCE' }, was: '8/11' });
}
{
  // blank-line control: extra blank lines inside X leave the ledger valid
  const XB = X_BASE.replace('The room was quiet.', 'The room was quiet.\n\n');
  const bc = ledgerOf(XB, BC_BASE);
  CASES.push({ id: 'sentence-blank-line-control', X: XB, bc, ledger: false, expect: { answer: '8/11' } });
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------
import { fileURLToPath } from 'node:url';

export function runReplay() {
  const results = [];
  let wrong = 0;
  for (const c of CASES) {
    const bc = c.ledger === false || c.credalOnly ? c.bc : withLedger(c.X, c.bc);
    const d = c.credalOnly ? solveCredal(bc) : solve(bc, c.X);
    let pass = false;
    let detail = '';
    if (d.status === 'ANSWER') {
      const v = fmt(d.value);
      pass = c.expect.answer === v;
      detail = `released ${v}`;
    } else {
      if (c.expect.refuse === true) pass = true;
      else if (c.expect.refuse === d.code) {
        pass = true;
        if (c.expect.interval && d.details?.interval) {
          const iv = d.details.interval;
          pass =
            fmt(typeof iv.min === 'string' ? R.parse(iv.min) : iv.min) === c.expect.interval[0] &&
            fmt(typeof iv.max === 'string' ? R.parse(iv.max) : iv.max) === c.expect.interval[1];
          detail = ` interval [${iv.min},${iv.max}]`;
        }
      }
      detail = `refused ${d.code}${detail}`;
    }
    if (!pass) wrong++;
    results.push({ id: c.id, expect: c.expect, got: detail, pass, was: c.was ?? null, decision: d });
  }
  return results;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const results = runReplay();
  const wrong = results.filter((r) => !r.pass).length;
  for (const r of results) {
    console.log(
      `${r.pass ? 'PASS' : 'FAIL'} ${r.id.padEnd(46)} want=${JSON.stringify(r.expect)} got=${r.got}${r.was ? ` (was ${r.was})` : ''}`
    );
  }
  console.log(`\n${results.length - wrong}/${results.length} oracle-correct; ${wrong} discrepant`);
  if (process.argv.includes('--json')) console.log(JSON.stringify(results, null, 2));
  process.exit(wrong ? 1 : 0);
}
