// prism-v4 :: offline unit tests — deterministic mock extractor +
// mock cross-model witness (global fetch stub). No live Ollama calls.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as R from '../src/kernel/rational.js';
import { solve } from '../src/kernel/solve.js';
import {
  compileV4,
  normCtxV4,
  normSwapName,
  verifyV4,
  checkV4SemanticEntailment,
  verifyV4Witness,
  solveV4,
  buildV4Checks,
  coherentSwapsForProgram,
  alignEntityNames,
} from '../src/v4/index.js';

const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);

// ---------- mock extractors -------------------------------------------------
function cloneProg(prog) {
  return JSON.parse(
    JSON.stringify(prog, (k, v) =>
      v && typeof v === 'object' && 'n' in v && 'd' in v
        ? { n: String(v.n), d: String(v.d) }
        : v),
    (k, v) =>
      v && typeof v === 'object' && 'n' in v && 'd' in v && typeof v.n === 'string'
        ? { n: BigInt(v.n), d: BigInt(v.d) }
        : v
  );
}

// Genuine semantic binding: on a counterfactual, re-binds each factor to
// the literal now at its site (via gate info), spans remapped.
function semanticExtractor(json, X0) {
  const c = compileV4(JSON.parse(JSON.stringify(json)), X0);
  assert.deepEqual(c.errors, [], 'template must compile');
  const template = c.program;
  return async (X, info = {}) => {
    const prog = cloneProg(template);
    if (info.slot === 'swap' && info.remapSpan) {
      for (const d of prog.decls) {
        if (d.span) d.span = { ...info.remapSpan(d.span) };
      }
      const numeric = prog.decls.filter((d) => NUMERIC_OPS.has(d.op));
      numeric.forEach((d, i) => {
        const rs = info.remappedSites?.[info.declSiteIdx?.[i]];
        if (rs) d.value = rs.value;
      });
    }
    prog.hypNames = { ...c.hypNames };
    prog.evNames = { ...c.evNames };
    return prog;
  };
}

// The positional bug: spans remapped, values memorized — must refuse.
function positionalExtractor(json, X0) {
  const c = compileV4(JSON.parse(JSON.stringify(json)), X0);
  assert.deepEqual(c.errors, [], 'template must compile');
  const template = c.program;
  return async (X, info = {}) => {
    const prog = cloneProg(template);
    if (info.slot === 'swap' && info.remapSpan) {
      for (const d of prog.decls) {
        if (d.span) d.span = { ...info.remapSpan(d.span) };
      }
    }
    prog.hypNames = { ...c.hypNames };
    prog.evNames = { ...c.evNames };
    return prog;
  };
}

// Semantic binding with a caller-supplied mutation of the swap program
// (e.g. morphological name variants or a "base" ctx on the swap). With
// trackValues:false the values stay memorized — a genuine binding bug.
function swapMutatingExtractor(json, X0, { trackValues = true, mutate = null } = {}) {
  const c = compileV4(JSON.parse(JSON.stringify(json)), X0);
  assert.deepEqual(c.errors, [], 'template must compile');
  const template = c.program;
  return async (X, info = {}) => {
    const prog = cloneProg(template);
    prog.hypNames = { ...c.hypNames };
    prog.evNames = { ...c.evNames };
    if (info.slot === 'swap' && info.remapSpan) {
      for (const d of prog.decls) {
        if (d.span) d.span = { ...info.remapSpan(d.span) };
      }
      if (trackValues) {
        const numeric = prog.decls.filter((d) => NUMERIC_OPS.has(d.op));
        numeric.forEach((d, i) => {
          const rs = info.remappedSites?.[info.declSiteIdx?.[i]];
          if (rs) d.value = rs.value;
        });
      }
      if (mutate) mutate(prog);
    }
    return prog;
  };
}

// ---------- mock witness transport ------------------------------------------
// Approves every emitted check unless told to fail/drop some ids. The
// reply still travels the real sealing path in checkV4SemanticEntailment.
function mockWitnessFetch(t, { failIds = new Set(), dropIds = new Set() } = {}) {
  return t.mock.method(globalThis, 'fetch', async (url, init) => {
    const body = JSON.parse(init.body);
    const prompt = body.messages[0].content;
    const ids = /CHECK_IDS: (.+)/.exec(prompt)[1].split(',');
    const checks = ids
      .filter((id) => !dropIds.has(id))
      .map((id) => ({ id, entailed: !failIds.has(id), reason: 'mock' }));
    const content = JSON.stringify({
      checks,
      entailed: failIds.size === 0,
      code: failIds.size === 0 ? null : 'NOT_ENTAILED',
      reason: 'mock',
    });
    return Response.json({ message: { content } });
  });
}

function forbidFetch(t) {
  return t.mock.method(globalThis, 'fetch', async () => {
    throw new Error('unexpected fetch');
  });
}

// ---------- passages + extractions ------------------------------------------
const STD_X =
  'Every relay in the trial was either a repaired unit or a new replacement. ' +
  'Refurbished modules accounted for three in ten of the trial. ' +
  'New replacements made up 70% of the trial. ' +
  'Among repaired units, the amber notch appears in 0.46 of cases. ' +
  'Among new replacements, the amber notch appears in 0.10 of cases. ' +
  'A relay was observed with an amber notch. ' +
  'What is the probability it is a repaired unit?';
const STD_J = {
  hyps: [
    { id: 'h0', name: 'repaired unit', span: 'either a repaired unit or a new replacement' },
    { id: 'h1', name: 'new replacement', span: 'either a repaired unit or a new replacement' },
  ],
  events: [{ id: 'e0', name: 'amber notch', span: 'the amber notch appears in 0.46' }],
  priors: [
    { hyp: 'h0', value: '3/10', span: 'Refurbished modules accounted for three in ten', ctx: 'base' },
    { hyp: 'h1', value: '7/10', span: 'New replacements made up 70%', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '23/50', span: 'the amber notch appears in 0.46', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/10', span: 'the amber notch appears in 0.10', ctx: 'base' },
  ],
  posteriors: [],
  obs: [{ ev: 'e0', pol: 'POS', span: 'observed with an amber notch', ctx: 'base' }],
  condIndep: [],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
};
// (3/10*23/50) / (3/10*23/50 + 7/10*1/10) = 69/104
const STD_ANSWER = '69/104';

const CI_X =
  'Every panel is either Elm or Fir. Elm accounts for 1/2 of panels and Fir accounts for 1/2. ' +
  'For Elm panels, the lamp lights in 3/4 of cases and the buzzer sounds in 1/2 of cases. ' +
  'For Fir panels, the lamp lights in 1/4 of cases and the buzzer sounds in 1/2 of cases. ' +
  'The chance of the buzzer is unchanged when the lamp occurs. ' +
  'A panel was observed with the lamp lit and the buzzer sounding. ' +
  'What is the probability it is Elm?';
const CI_J = {
  hyps: [
    { id: 'h0', name: 'Elm', span: 'either Elm or Fir' },
    { id: 'h1', name: 'Fir', span: 'either Elm or Fir' },
  ],
  events: [
    { id: 'e0', name: 'lamp', span: 'the lamp lights in 3/4' },
    { id: 'e1', name: 'buzzer', span: 'the buzzer sounds in 1/2' },
  ],
  priors: [
    { hyp: 'h0', value: '1/2', span: 'Elm accounts for 1/2', ctx: 'base' },
    { hyp: 'h1', value: '1/2', span: 'Fir accounts for 1/2', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '3/4', span: 'the lamp lights in 3/4', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/2', span: 'Elm panels, the lamp lights in 3/4 of cases and the buzzer sounds in 1/2', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/4', span: 'the lamp lights in 1/4', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h1', value: '1/2', span: 'Fir panels, the lamp lights in 1/4 of cases and the buzzer sounds in 1/2', ctx: 'base' },
  ],
  posteriors: [],
  obs: [
    { ev: 'e0', pol: 'POS', span: 'the lamp lit', ctx: 'base' },
    { ev: 'e1', pol: 'POS', span: 'the buzzer sounding', ctx: 'base' },
  ],
  condIndep: [
    { evs: ['e0', 'e1'], span: 'The chance of the buzzer is unchanged when the lamp occurs', ctx: 'base' },
  ],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, { ev: 'e1', pol: 'POS' }], ctx: 'base' },
};
// (1/2*3/4*1/2) / (1/2*3/4*1/2 + 1/2*1/4*1/2) = 3/4
const CI_ANSWER = '3/4';

const twoEvents = (likSpans) => ({
  events: [
    { id: 'e0', name: 'lamp', span: likSpans[0] },
    { id: 'e1', name: 'buzzer', span: likSpans[1] },
  ],
});

const EQUIV_X =
  'Every capsule is either Elm or Fir. Elm accounts for 2/5 of capsules and Fir accounts for 3/5. ' +
  'For Elm capsules, the lamp lights in 1/2 of cases and the buzzer sounds in 1/2 of cases. ' +
  'For Fir capsules, the lamp lights in 1/5 of cases and the buzzer sounds in 1/5 of cases. ' +
  'The lamp and the buzzer are wired to the same switch, so they always activate together. ' +
  'A capsule was observed with the lamp lit and the buzzer sounding. ' +
  'What is the probability it is Elm?';
const EQUIV_J = {
  hyps: [
    { id: 'h0', name: 'Elm', span: 'either Elm or Fir' },
    { id: 'h1', name: 'Fir', span: 'either Elm or Fir' },
  ],
  ...twoEvents(['the lamp lights in 1/2', 'buzzer sounds in 1/2']),
  priors: [
    { hyp: 'h0', value: '2/5', span: 'Elm accounts for 2/5', ctx: 'base' },
    { hyp: 'h1', value: '3/5', span: 'Fir accounts for 3/5', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/2', span: 'Elm capsules, the lamp lights in 1/2', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/2', span: 'Elm capsules, the lamp lights in 1/2 of cases and the buzzer sounds in 1/2', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/5', span: 'Fir capsules, the lamp lights in 1/5', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h1', value: '1/5', span: 'Fir capsules, the lamp lights in 1/5 of cases and the buzzer sounds in 1/5', ctx: 'base' },
  ],
  posteriors: [],
  obs: [
    { ev: 'e0', pol: 'POS', span: 'the lamp lit', ctx: 'base' },
    { ev: 'e1', pol: 'POS', span: 'the buzzer sounding', ctx: 'base' },
  ],
  condIndep: [],
  detLinks: [
    { a: 'e0', rel: 'EQUIV', b: 'e1', span: 'The lamp and the buzzer are wired to the same switch', ctx: 'base' },
  ],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, { ev: 'e1', pol: 'POS' }], ctx: 'base' },
};
// (2/5*1/2) / (2/5*1/2 + 3/5*1/5) = 5/8
const EQUIV_ANSWER = '5/8';

const IMPL_X =
  'Every unit is either Alpha or Beta. Alpha accounts for 1/3 of units and Beta accounts for 2/3. ' +
  'For Alpha units, the star appears in 3/5 of cases and the notch appears in 4/5 of cases. ' +
  'For Beta units, the star appears in 1/5 of cases and the notch appears in 3/5 of cases. ' +
  'Every unit that has a star also has a notch. ' +
  'A unit was observed with a star and a notch. ' +
  'What is the probability it is Alpha?';
const IMPL_J = {
  hyps: [
    { id: 'h0', name: 'Alpha', span: 'either Alpha or Beta' },
    { id: 'h1', name: 'Beta', span: 'either Alpha or Beta' },
  ],
  events: [
    { id: 'e0', name: 'star', span: 'the star appears in 3/5' },
    { id: 'e1', name: 'notch', span: 'the notch appears in 4/5' },
  ],
  priors: [
    { hyp: 'h0', value: '1/3', span: 'Alpha accounts for 1/3', ctx: 'base' },
    { hyp: 'h1', value: '2/3', span: 'Beta accounts for 2/3', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '3/5', span: 'the star appears in 3/5', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h0', value: '4/5', span: 'the notch appears in 4/5', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/5', span: 'the star appears in 1/5', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h1', value: '3/5', span: 'Beta units, the star appears in 1/5 of cases and the notch appears in 3/5', ctx: 'base' },
  ],
  posteriors: [],
  obs: [
    { ev: 'e0', pol: 'POS', span: 'with a star', ctx: 'base' },
    { ev: 'e1', pol: 'POS', span: 'and a notch', ctx: 'base' },
  ],
  condIndep: [],
  detLinks: [
    { a: 'e0', rel: 'IMPLIES', b: 'e1', span: 'Every unit that has a star also has a notch', ctx: 'base' },
  ],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, { ev: 'e1', pol: 'POS' }], ctx: 'base' },
};
// star implies notch, so P(star and notch | h) = P(star | h):
// (1/3*3/5) / (1/3*3/5 + 2/3*1/5) = 3/5
const IMPL_ANSWER = '3/5';

const MUTEX_X =
  'Every unit is either Alpha or Beta. Alpha accounts for 3/5 of units and Beta accounts for 2/5. ' +
  'For Alpha units, the star appears in 1/4 of cases and the notch appears in 1/2 of cases. ' +
  'For Beta units, the star appears in 3/4 of cases and the notch appears in 1/4 of cases. ' +
  'No unit ever has both a star and a notch. ' +
  'A unit was observed with a star and no notch. ' +
  'What is the probability it is Alpha?';
const MUTEX_J = {
  hyps: [
    { id: 'h0', name: 'Alpha', span: 'either Alpha or Beta' },
    { id: 'h1', name: 'Beta', span: 'either Alpha or Beta' },
  ],
  events: [
    { id: 'e0', name: 'star', span: 'the star appears in 1/4' },
    { id: 'e1', name: 'notch', span: 'the notch appears in 1/2' },
  ],
  priors: [
    { hyp: 'h0', value: '3/5', span: 'Alpha accounts for 3/5', ctx: 'base' },
    { hyp: 'h1', value: '2/5', span: 'Beta accounts for 2/5', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/4', span: 'Alpha units, the star appears in 1/4', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h0', value: '1/2', span: 'the notch appears in 1/2', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '3/4', span: 'the star appears in 3/4', ctx: 'base' },
    { ev: 'e1', pol: 'POS', hyp: 'h1', value: '1/4', span: 'Beta units, the star appears in 3/4 of cases and the notch appears in 1/4', ctx: 'base' },
  ],
  posteriors: [],
  obs: [
    { ev: 'e0', pol: 'POS', span: 'with a star', ctx: 'base' },
    { ev: 'e1', pol: 'NEG', span: 'and no notch', ctx: 'base' },
  ],
  condIndep: [],
  detLinks: [
    { a: 'e0', rel: 'MUTEX', b: 'e1', span: 'No unit ever has both a star and a notch', ctx: 'base' },
  ],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }, { ev: 'e1', pol: 'NEG' }], ctx: 'base' },
};
// mutex, so P(star and no notch | h) = P(star | h):
// (3/5*1/4) / (3/5*1/4 + 2/5*3/4) = 1/3
const MUTEX_ANSWER = '1/3';

const POST_X =
  'Every case is either Alpha or Beta. Among cases with a green mark, 7/11 are Alpha and 4/11 are Beta. ' +
  'A case was observed with a green mark. What is the probability it is Alpha?';
const POST_J = {
  hyps: [
    { id: 'h0', name: 'Alpha', span: 'either Alpha or Beta' },
    { id: 'h1', name: 'Beta', span: 'either Alpha or Beta' },
  ],
  events: [{ id: 'e0', name: 'green mark', span: 'with a green mark, 7/11' }],
  priors: [],
  liks: [],
  posteriors: [
    { hyp: 'h0', ev: 'e0', pol: 'POS', value: '7/11', span: 'with a green mark, 7/11 are Alpha', ctx: 'base' },
    { hyp: 'h1', ev: 'e0', pol: 'POS', value: '4/11', span: '4/11 are Beta', ctx: 'base' },
  ],
  obs: [{ ev: 'e0', pol: 'POS', span: 'observed with a green mark', ctx: 'base' }],
  condIndep: [],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
};

const STRATA_X =
  'Every crate in the east room is either Cedar or Pine, and every crate in the west room is either Cedar or Pine. ' +
  'In the east room, Cedar accounts for 3/4 of crates and Pine accounts for 1/4. ' +
  'In the west room, Cedar accounts for 1/4 of crates and Pine accounts for 3/4. ' +
  'In the east room, Cedar crates show a seal in 2/3 of cases and Pine crates show a seal in 1/3 of cases. ' +
  'In the west room, Cedar crates show a seal in 1/6 of cases and Pine crates show a seal in 5/6 of cases. ' +
  'An east room crate was observed with a seal. What is the probability it is Cedar?';
const STRATA_J = {
  hyps: [
    { id: 'h0', name: 'Cedar', span: 'either Cedar or Pine' },
    { id: 'h1', name: 'Pine', span: 'either Cedar or Pine' },
  ],
  events: [{ id: 'e0', name: 'seal', span: 'show a seal in 2/3' }],
  priors: [
    { hyp: 'h0', value: '3/4', span: 'In the east room, Cedar accounts for 3/4', ctx: 'east room' },
    { hyp: 'h1', value: '1/4', span: 'east room, Cedar accounts for 3/4 of crates and Pine accounts for 1/4', ctx: 'east room' },
    { hyp: 'h0', value: '1/4', span: 'In the west room, Cedar accounts for 1/4', ctx: 'west room' },
    { hyp: 'h1', value: '3/4', span: 'west room, Cedar accounts for 1/4 of crates and Pine accounts for 3/4', ctx: 'west room' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '2/3', span: 'east room, Cedar crates show a seal in 2/3', ctx: 'east room' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/3', span: 'Pine crates show a seal in 1/3', ctx: 'east room' },
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/6', span: 'west room, Cedar crates show a seal in 1/6', ctx: 'west room' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '5/6', span: 'Pine crates show a seal in 5/6', ctx: 'west room' },
  ],
  posteriors: [],
  obs: [{ ev: 'e0', pol: 'POS', span: 'east room crate was observed with a seal', ctx: 'east room' }],
  condIndep: [],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'east room' },
};
// east only: (3/4*2/3) / (3/4*2/3 + 1/4*1/3) = 6/7
const STRATA_ANSWER = '6/7';

const BG_X =
  'The workshop keeps 320 records in a cabinet by the door. Every tag is either Brass or Zinc. ' +
  'Brass accounts for 4/5 of tags and Zinc accounts for 1/5. ' +
  'Brass tags carry a stamp in 9/10 of cases. Zinc tags carry a stamp in 1/10 of cases. ' +
  'The cabinet was painted last spring. A tag was observed with a stamp. ' +
  'What is the probability it is Brass?';
const BG_J = {
  hyps: [
    { id: 'h0', name: 'Brass', span: 'either Brass or Zinc' },
    { id: 'h1', name: 'Zinc', span: 'either Brass or Zinc' },
  ],
  events: [{ id: 'e0', name: 'stamp', span: 'carry a stamp in 9/10' }],
  priors: [
    { hyp: 'h0', value: '4/5', span: 'Brass accounts for 4/5', ctx: 'base' },
    { hyp: 'h1', value: '1/5', span: 'Zinc accounts for 1/5', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '9/10', span: 'Brass tags carry a stamp in 9/10', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '1/10', span: 'Zinc tags carry a stamp in 1/10', ctx: 'base' },
  ],
  posteriors: [],
  obs: [{ ev: 'e0', pol: 'POS', span: 'observed with a stamp', ctx: 'base' }],
  condIndep: [],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
};
// (4/5*9/10) / (4/5*9/10 + 1/5*1/10) = 36/37
const BG_ANSWER = '36/37';

const COORD_X =
  'Every relay is either a repaired unit or a new replacement. Repaired units made up 1/2 of relays and new replacements made up 1/2. ' +
  'The stamp rates for repaired and new relays were 0.34 and 0.62, respectively. ' +
  'A relay was observed with a stamp. What is the probability it is a repaired unit?';
const COORD_J = {
  hyps: [
    { id: 'h0', name: 'repaired unit', span: 'either a repaired unit or a new replacement' },
    { id: 'h1', name: 'new replacement', span: 'either a repaired unit or a new replacement' },
  ],
  events: [{ id: 'e0', name: 'stamp', span: 'observed with a stamp' }],
  priors: [
    { hyp: 'h0', value: '1/2', span: 'Repaired units made up 1/2', ctx: 'base' },
    { hyp: 'h1', value: '1/2', span: 'new replacements made up 1/2', ctx: 'base' },
  ],
  liks: [
    { ev: 'e0', pol: 'POS', hyp: 'h0', value: '17/50', span: 'were 0.34 and 0.62, respectively', ctx: 'base' },
    { ev: 'e0', pol: 'POS', hyp: 'h1', value: '31/50', span: 'were 0.34 and 0.62, respectively', ctx: 'base' },
  ],
  posteriors: [],
  obs: [{ ev: 'e0', pol: 'POS', span: 'observed with a stamp', ctx: 'base' }],
  condIndep: [],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
};
// (1/2*17/50) / (1/2*17/50 + 1/2*31/50) = 17/48
const COORD_ANSWER = '17/48';

// ---------- compileV4 ---------------------------------------------------------
test('compileV4: exact rationals from words, percents, decimals, fractions', () => {
  const c = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  assert.deepEqual(c.errors, []);
  const byOp = (op) => c.program.decls.filter((d) => d.op === op);
  // "three in ten" -> 3/10, "70%" -> 7/10, "0.46" -> 23/50, "0.10" -> 1/10
  assert.deepEqual(byOp('PRIOR').map((d) => R.toString(d.value)), ['3/10', '7/10']);
  assert.deepEqual(byOp('LIK').map((d) => R.toString(d.value)), ['23/50', '1/10']);
  assert.equal(c.bound.length, 4);
  assert.equal(new Set(c.bound.map((b) => `${b.site.s}:${b.site.e}`)).size, 4);
});

test('compileV4: word ratio "two thirds" binds exactly', () => {
  const X =
    'Every slip is either Red or Blue. Red makes up two thirds of slips and Blue makes up one third. ' +
    'Red slips carry a dot in 1/4 of cases. Blue slips carry a dot in 3/4 of cases. ' +
    'A slip was observed with a dot. What is the probability it is Red?';
  const j = {
    hyps: [
      { id: 'h0', name: 'Red', span: 'either Red or Blue' },
      { id: 'h1', name: 'Blue', span: 'either Red or Blue' },
    ],
    events: [{ id: 'e0', name: 'dot', span: 'carry a dot in 1/4' }],
    priors: [
      { hyp: 'h0', value: '2/3', span: 'makes up two thirds', ctx: 'base' },
      { hyp: 'h1', value: '1/3', span: 'makes up one third', ctx: 'base' },
    ],
    liks: [
      { ev: 'e0', pol: 'POS', hyp: 'h0', value: '1/4', span: 'carry a dot in 1/4', ctx: 'base' },
      { ev: 'e0', pol: 'POS', hyp: 'h1', value: '3/4', span: 'carry a dot in 3/4', ctx: 'base' },
    ],
    posteriors: [],
    obs: [{ ev: 'e0', pol: 'POS', span: 'observed with a dot', ctx: 'base' }],
    condIndep: [], detLinks: [], nonAsserted: [],
    partition: ['h0', 'h1'],
    query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
  };
  const c = compileV4(j, X);
  assert.deepEqual(c.errors, []);
  const priors = c.program.decls.filter((d) => d.op === 'PRIOR');
  assert.deepEqual(priors.map((d) => R.toString(d.value)), ['2/3', '1/3']);
});

test('compileV4: literal-site uniqueness refuses one figure grounding two factors', () => {
  const j = JSON.parse(JSON.stringify(STD_J));
  j.posteriors = [
    { hyp: 'h0', ev: 'e0', pol: 'POS', value: '3/10', span: 'Refurbished modules accounted for three in ten', ctx: 'base' },
  ];
  const c = compileV4(j, STD_X);
  assert.ok(c.errors.some((e) => /shared literal/.test(e)), c.errors.join('; '));
});

test('compileV4: unresolvable and value-mismatched spans refuse', () => {
  const badSpan = JSON.parse(JSON.stringify(STD_J));
  badSpan.priors[0].span = 'this text is nowhere in the passage';
  assert.ok(compileV4(badSpan, STD_X).errors.some((e) => /unresolvable span/.test(e)));

  const badValue = JSON.parse(JSON.stringify(STD_J));
  badValue.priors[0].value = '1/4'; // span holds 3/10, not 1/4
  assert.ok(compileV4(badValue, STD_X).errors.some((e) => /no literal equal/.test(e)));

  const outOfRange = JSON.parse(JSON.stringify(STD_J));
  outOfRange.priors[0].value = '3/2';
  assert.ok(compileV4(outOfRange, STD_X).errors.some((e) => /outside \[0,1\]/.test(e)));
});

test('compileV4: NON_ASSERTED figures are excluded from active premises', () => {
  const j = JSON.parse(JSON.stringify(STD_J));
  j.nonAsserted = [
    { span: 'Refurbished modules accounted for three in ten', reason: 'RETRACTED' },
  ];
  const c = compileV4(j, STD_X);
  assert.deepEqual(c.errors, []);
  const priors = c.program.decls.filter((d) => d.op === 'PRIOR');
  assert.deepEqual(priors.map((d) => d.hyp), ['h1']);
  assert.equal(c.program.decls.filter((d) => d.op === 'NON_ASSERTED').length, 1);
});

test('compileV4: COND_INDEP on the DET_LINK sentence is deduplicated', () => {
  const j = JSON.parse(JSON.stringify(IMPL_J));
  j.condIndep = [
    { evs: ['e0', 'e1'], span: 'Every unit that has a star also has a notch', ctx: 'base' },
  ];
  const c = compileV4(j, IMPL_X);
  assert.deepEqual(c.errors, []);
  assert.equal(c.program.decls.filter((d) => d.op === 'COND_INDEP').length, 0);
  assert.equal(c.program.decls.filter((d) => d.op === 'DET_LINK').length, 1);
});

test('compileV4: stratum ctx labels stay distinct; base collapses to ctx0', () => {
  assert.equal(normCtxV4('base'), 'ctx0');
  assert.equal(normCtxV4(''), 'ctx0');
  assert.equal(normCtxV4('East Room'), 'ctx_east_room');
  assert.equal(normCtxV4('ctx_east_room'), 'ctx_east_room');

  const strata = compileV4(JSON.parse(JSON.stringify(STRATA_J)), STRATA_X);
  assert.deepEqual(strata.errors, []);
  const ctxs = new Set(
    strata.program.decls.filter((d) => d.ctx).map((d) => d.ctx)
  );
  assert.deepEqual([...ctxs].sort(), ['ctx_east_room', 'ctx_west_room']);
  assert.equal(strata.program.query.ctx, 'ctx_east_room');
});

test('compileV4: a single non-ctx0 population label canonicalizes to ctx0', () => {
  const j = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j.priors, j.liks, j.obs]) {
    for (const e of list) e.ctx = 'the intake';
  }
  j.query.ctx = 'the intake';
  const c = compileV4(j, STD_X);
  assert.deepEqual(c.errors, []);
  const ctxs = new Set(c.program.decls.filter((d) => d.ctx).map((d) => d.ctx));
  assert.deepEqual([...ctxs], ['ctx0']);
  assert.equal(c.program.query.ctx, 'ctx0');
});

test('compileV4: hypothesis-name echo in LIK ctx normalizes to the shared ctx', () => {
  // "Alpha" / "Alpha lots"-style echo: own-hyp tokens in the LIK ctx
  const j = JSON.parse(JSON.stringify(STD_J));
  j.liks[0].ctx = 'repaired'; // h0 = "repaired unit"
  j.liks[1].ctx = 'new replacement lots'; // h1 = "new replacement"
  const c = compileV4(j, STD_X);
  assert.deepEqual(c.errors, []);
  assert.deepEqual(
    c.program.decls.filter((d) => d.op === 'LIK').map((d) => d.ctx),
    ['ctx0', 'ctx0']
  );

  // spec example: ctx_starter_lots on h0 = "commercial starter"
  const j2 = JSON.parse(JSON.stringify(STD_J));
  j2.hyps[0].name = 'commercial starter';
  j2.hyps[1].name = 'commercial standard';
  j2.liks[0].ctx = 'starter lots';
  j2.liks[1].ctx = 'standard lots';
  const c2 = compileV4(j2, STD_X);
  assert.deepEqual(c2.errors, []);
  assert.deepEqual(
    c2.program.decls.filter((d) => d.op === 'LIK').map((d) => d.ctx),
    ['ctx0', 'ctx0']
  );

  // echo onto a shared non-base population: binds there, then the
  // single-population canonicalization collapses everything to ctx0
  const j3 = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j3.priors, j3.obs]) {
    for (const e of list) e.ctx = 'audit intake';
  }
  j3.query.ctx = 'audit intake';
  j3.liks[0].ctx = 'repaired';
  j3.liks[1].ctx = 'new replacement';
  const c3 = compileV4(j3, STD_X);
  assert.deepEqual(c3.errors, []);
  const ctxs3 = new Set(c3.program.decls.filter((d) => d.ctx).map((d) => d.ctx));
  assert.deepEqual([...ctxs3], ['ctx0']);
  assert.equal(c3.program.query.ctx, 'ctx0');
});

test('compileV4: hypothesis-name echo in POSTERIOR ctx ("Alpha"/"Beta lots") normalizes', () => {
  const j = JSON.parse(JSON.stringify(POST_J));
  j.posteriors[0].ctx = 'Alpha'; // h0 = "Alpha"
  j.posteriors[1].ctx = 'Beta lots'; // h1 = "Beta"
  const c = compileV4(j, POST_X);
  assert.deepEqual(c.errors, []);
  assert.deepEqual(
    c.program.decls.filter((d) => d.op === 'POSTERIOR_STATED').map((d) => d.ctx),
    ['ctx0', 'ctx0']
  );
});

test('compileV4: echo guard leaves genuine strata and query-named ctx untouched', () => {
  // genuine multi-stratum ctxs share no token with a hypothesis name
  const strata = compileV4(JSON.parse(JSON.stringify(STRATA_J)), STRATA_X);
  assert.deepEqual(strata.errors, []);
  const ctxs = new Set(strata.program.decls.filter((d) => d.ctx).map((d) => d.ctx));
  assert.deepEqual([...ctxs].sort(), ['ctx_east_room', 'ctx_west_room']);
  assert.equal(strata.program.query.ctx, 'ctx_east_room');

  // a ctx token that is also in the query ctx is NOT an echo: priors in
  // the east room, query about the "repaired room", LIK ctx "repaired"
  // stays exactly as stated (and >=2 ctxs means no canonicalization)
  const j = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j.priors, j.obs]) {
    for (const e of list) e.ctx = 'east room';
  }
  j.liks[0].ctx = 'repaired'; // h0 token, but also in query ctx below
  j.liks[1].ctx = 'east room';
  j.query.ctx = 'repaired room';
  const c = compileV4(j, STD_X);
  assert.deepEqual(c.errors, []);
  assert.deepEqual(
    c.program.decls.filter((d) => d.op === 'LIK').map((d) => d.ctx),
    ['ctx_repaired', 'ctx_east_room']
  );
  assert.equal(c.program.query.ctx, 'ctx_repaired_room');
});

test('compileV4: single-population arbitrary ctx canonicalizes; multi-stratum never collapses', () => {
  const j = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j.priors, j.liks, j.obs]) {
    for (const e of list) e.ctx = 'signaling trial';
  }
  j.query.ctx = 'signaling trial';
  const c = compileV4(j, STD_X);
  assert.deepEqual(c.errors, []);
  const ctxs = new Set(c.program.decls.filter((d) => d.ctx).map((d) => d.ctx));
  assert.deepEqual([...ctxs], ['ctx0']);
  assert.equal(c.program.query.ctx, 'ctx0');

  // the same passage stated with "base" compiles to the identical ctx,
  // so a swap emitting "base" cannot vanish a proposition
  const base = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  assert.deepEqual(
    c.program.decls.filter((d) => d.ctx).map((d) => d.ctx),
    base.program.decls.filter((d) => d.ctx).map((d) => d.ctx)
  );
  assert.equal(c.program.query.ctx, base.program.query.ctx);

  // multi-stratum: two distinct ctxs anywhere => nothing collapses
  const j2 = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j2.priors, j2.liks, j2.obs]) {
    for (const e of list) e.ctx = 'east room';
  }
  j2.query.ctx = 'west room';
  const c2 = compileV4(j2, STD_X);
  assert.deepEqual(c2.errors, []);
  const ctxs2 = new Set(c2.program.decls.filter((d) => d.ctx).map((d) => d.ctx));
  assert.deepEqual([...ctxs2], ['ctx_east_room']);
  assert.equal(c2.program.query.ctx, 'ctx_west_room');
});

test('compileV4: invalid ids and empty names refuse', () => {
  const badId = JSON.parse(JSON.stringify(STD_J));
  badId.hyps[0].id = 'hypA';
  assert.ok(compileV4(badId, STD_X).errors.length > 0);
  const badName = JSON.parse(JSON.stringify(STD_J));
  badName.events[0].name = '  ';
  assert.ok(compileV4(badName, STD_X).errors.length > 0);
});

test('compileV4: coordinated "0.34 and 0.62, respectively" binds distinct sites', () => {
  const c = compileV4(JSON.parse(JSON.stringify(COORD_J)), COORD_X);
  assert.deepEqual(c.errors, []);
  const liks = c.program.decls.filter((d) => d.op === 'LIK');
  assert.deepEqual(liks.map((d) => R.toString(d.value)), ['17/50', '31/50']);
  const likSites = c.bound
    .filter((b) => b.decl.op === 'LIK')
    .map((b) => `${b.site.s}:${b.site.e}`);
  assert.equal(new Set(likSites).size, 2);
});

// ---------- verifyV4: the six solvable families ------------------------------
const FAMILIES = [
  ['Standard Bayes', STD_X, STD_J, STD_ANSWER],
  ['Conditional Independence', CI_X, CI_J, CI_ANSWER],
  ['Deterministic Linkage EQUIV', EQUIV_X, EQUIV_J, EQUIV_ANSWER],
  ['Deterministic Linkage IMPLIES', IMPL_X, IMPL_J, IMPL_ANSWER],
  ['Deterministic Linkage MUTEX', MUTEX_X, MUTEX_J, MUTEX_ANSWER],
  ['Direct Posterior', POST_X, POST_J, '7/11'],
  ['Paired Population Strata', STRATA_X, STRATA_J, STRATA_ANSWER],
  ['Benign Background', BG_X, BG_J, BG_ANSWER],
  ['Coordinated respectively rates', COORD_X, COORD_J, COORD_ANSWER],
];

for (const [name, X, j, expected] of FAMILIES) {
  test(`verifyV4 PASS: ${name}`, async (t) => {
    mockWitnessFetch(t);
    const res = await verifyV4({ X, extract: semanticExtractor(j, X) });
    assert.equal(res.verdict, 'PASS', JSON.stringify({ code: res.code, details: res.details }));
    assert.equal(res.code, null);
    assert.equal(R.toString(res.value), expected);
    assert.equal(res.rational, expected);
    assert.equal(typeof res.rational, 'string');
    assert.ok(R.eq(res.interval.min, res.interval.max));
    assert.equal(res.decision.status, 'ANSWER');
    assert.ok(res.swaps.length >= 1 && res.swaps.every((s) => s.ok), 'swaps ran and passed');
    assert.equal(res.witness.entailed, true);
    assert.ok(res.compiled.program);
    assert.equal(typeof res.elapsedMs, 'number');
  });
}

test('verifyV4: raw-JSON extractor path (maxSwaps 0) passes Standard Bayes', async (t) => {
  mockWitnessFetch(t);
  const res = await verifyV4({
    X: STD_X,
    extract: async () => JSON.parse(JSON.stringify(STD_J)),
    maxSwaps: 0,
  });
  assert.equal(res.verdict, 'PASS');
  assert.equal(R.toString(res.value), STD_ANSWER);
  assert.deepEqual(res.swaps, []);
});

test('verifyV4 PASS: hypothesis-name echo in LIK ctx solves (no false CONTEXT_MISMATCH)', async (t) => {
  mockWitnessFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  j.liks[0].ctx = 'repaired unit'; // h0 name echoed into ctx
  j.liks[1].ctx = 'new replacement lots'; // h1 name + generic noun
  const res = await verifyV4({ X: STD_X, extract: semanticExtractor(j, STD_X) });
  assert.equal(res.verdict, 'PASS', JSON.stringify({ code: res.code, details: res.details }));
  assert.equal(res.rational, STD_ANSWER);

  // direct-posterior echo: "Alpha" / "Beta lots" on POSTERIOR_STATED
  const jp = JSON.parse(JSON.stringify(POST_J));
  jp.posteriors[0].ctx = 'Alpha';
  jp.posteriors[1].ctx = 'Beta lots';
  const resP = await verifyV4({ X: POST_X, extract: semanticExtractor(jp, POST_X) });
  assert.equal(resP.verdict, 'PASS', JSON.stringify({ code: resP.code, details: resP.details }));
  assert.equal(resP.rational, '7/11');
});

test('verifyV4 PASS: single-population ctx slug survives a swap emitting "base"', async (t) => {
  mockWitnessFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j.priors, j.liks, j.obs]) {
    for (const e of list) e.ctx = 'signaling trial';
  }
  j.query.ctx = 'signaling trial';
  // on the counterfactual the extractor emits the compiled "base" ctx
  const res = await verifyV4({
    X: STD_X,
    extract: swapMutatingExtractor(j, STD_X, {
      mutate: (prog) => {
        for (const d of prog.decls) if (d.ctx) d.ctx = 'ctx0';
        prog.query.ctx = 'ctx0';
      },
    }),
  });
  assert.equal(res.verdict, 'PASS', JSON.stringify({ code: res.code, details: res.details }));
  assert.ok(res.swaps.length >= 1 && res.swaps.every((s) => s.ok), 'swaps ran and passed');
  assert.equal(res.rational, STD_ANSWER);
});

test('verifyV4 PASS: morphological name variants on swap track covariantly', async (t) => {
  mockWitnessFetch(t);
  const res = await verifyV4({
    X: STD_X,
    extract: swapMutatingExtractor(STD_J, STD_X, {
      mutate: (prog) => {
        prog.hypNames = { h0: 'repaired units', h1: 'a new replacement' };
        prog.evNames = { e0: 'amber-notch' };
      },
    }),
  });
  assert.equal(res.verdict, 'PASS', JSON.stringify({ code: res.code, details: res.details }));
  assert.ok(res.swaps.length >= 1 && res.swaps.every((s) => s.ok), 'swaps ran and passed');
  assert.equal(res.rational, STD_ANSWER);
});

test('verifyV4 REFUSED: morphological name variants do not excuse unswapped values', async (t) => {
  const fetch = forbidFetch(t);
  const res = await verifyV4({
    X: STD_X,
    extract: swapMutatingExtractor(STD_J, STD_X, {
      trackValues: false, // values memorized: genuinely wrong on swap
      mutate: (prog) => {
        prog.hypNames = { h0: 'repaired units', h1: 'the new replacements' };
        prog.evNames = { e0: 'amber-notch' };
      },
    }),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.ok(res.swaps.some((s) => !s.ok));
  assert.match(res.details?.detail ?? '', /did not track literal permutation/);
  assert.equal(fetch.mock.callCount(), 0);
});

test('normSwapName: articles, hyphens, and trailing plurals fold together', () => {
  assert.equal(normSwapName('repaired units'), normSwapName('Repaired Unit'));
  assert.equal(normSwapName('the repaired unit'), normSwapName('a repaired units'));
  assert.equal(normSwapName('fixed lens'), normSwapName('fixed-lens'));
  assert.equal(normSwapName('fixed_lens'), normSwapName('fixed  lens'));
  assert.equal(normSwapName('trial bodies'), 'trial body');
  assert.equal(normSwapName('sealed boxes'), 'sealed box');
  assert.equal(normSwapName('amber notches'), 'amber notch');
  // conservative: ss / us endings are not singularized away
  assert.equal(normSwapName('glass'), 'glass');
  assert.equal(normSwapName('status'), 'status');
  // genuinely different names stay different
  assert.notEqual(normSwapName('repaired unit'), normSwapName('new replacement'));
});

// ---------- verifyV4: trap families ------------------------------------------
test('verifyV4 REFUSED: MISSING_BASE_RATE when priors are absent', async (t) => {
  const fetch = forbidFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  j.priors = [];
  const res = await verifyV4({ X: STD_X, extract: semanticExtractor(j, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'MISSING_BASE_RATE');
  assert.equal(res.value, null);
  assert.equal(res.rational, null);
  assert.equal(fetch.mock.callCount(), 0, 'witness must not run on a refused solve');
});

test('verifyV4 REFUSED: INCOMPLETE_FACTORIZATION when a likelihood is missing', async (t) => {
  forbidFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  j.liks = j.liks.slice(0, 1);
  const res = await verifyV4({ X: STD_X, extract: semanticExtractor(j, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'INCOMPLETE_FACTORIZATION');
});

test('verifyV4 REFUSED: UNLICENSED_INDEPENDENCE without a license', async (t) => {
  forbidFetch(t);
  const j = JSON.parse(JSON.stringify(CI_J));
  j.condIndep = [];
  const res = await verifyV4({ X: CI_X, extract: semanticExtractor(j, CI_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNLICENSED_INDEPENDENCE');
});

test('verifyV4 REFUSED: CONTEXT_MISMATCH when the query stratum has no premises', async (t) => {
  forbidFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  for (const list of [j.priors, j.liks, j.obs]) {
    for (const e of list) e.ctx = 'east room';
  }
  j.query.ctx = 'west room';
  const res = await verifyV4({ X: STD_X, extract: semanticExtractor(j, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'CONTEXT_MISMATCH');
});

test('verifyV4 REFUSED: UNSTABLE_BINDING when values do not track swaps', async (t) => {
  const fetch = forbidFetch(t);
  const res = await verifyV4({ X: STD_X, extract: positionalExtractor(STD_J, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.ok(res.swaps.some((s) => !s.ok));
  assert.equal(fetch.mock.callCount(), 0, 'witness must not run after a failed swap');
});

test('verifyV4 REFUSED: SEMANTIC_NOT_ENTAILED when the witness rejects a check', async (t) => {
  mockWitnessFetch(t, { failIds: new Set(['L1']) });
  const res = await verifyV4({ X: STD_X, extract: semanticExtractor(STD_J, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'SEMANTIC_NOT_ENTAILED');
  assert.equal(res.witness.entailed, false);
});

test('verifyV4 REFUSED: SEMANTIC_NOT_ENTAILED when the witness drops a check id', async (t) => {
  // the mock still claims top-level entailed:true; the strict AND must refuse
  mockWitnessFetch(t, { dropIds: new Set(['G1']) });
  const res = await verifyV4({ X: STD_X, extract: semanticExtractor(STD_J, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'SEMANTIC_NOT_ENTAILED');
});

test('verifyV4 REFUSED: forged witness receipt is unauthenticated', async (t) => {
  forbidFetch(t);
  const forged = {
    witnessModel: 'hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M',
    witnessHost: 'http://127.0.0.1:11434',
    passageSha256: 'deadbeef',
    programCanonicalSha256: 'deadbeef',
    entailed: true,
    code: null,
    reason: 'trust me',
    checks: [],
  };
  const res = await verifyV4({
    X: STD_X,
    extract: semanticExtractor(STD_J, STD_X),
    witnessCheck: async () => forged,
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED');
});

test('verifyV4 REFUSED: EXTRACTION_FAILED on structurally broken extraction', async (t) => {
  forbidFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  j.priors[0].span = 'nowhere in the passage at all';
  const res = await verifyV4({
    X: STD_X,
    extract: async () => j,
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'EXTRACTION_FAILED');
});

// ---------- witness unit surface ---------------------------------------------
test('buildV4Checks: itemized ids cover every claim family', () => {
  const std = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  assert.deepEqual(
    buildV4Checks(std.program, STD_X, std).map((c) => c.id),
    ['PART', 'P1', 'P2', 'L1', 'L2', 'O1', 'G1']
  );
  const ci = compileV4(JSON.parse(JSON.stringify(CI_J)), CI_X);
  assert.deepEqual(
    buildV4Checks(ci.program, CI_X, ci).map((c) => c.id),
    ['PART', 'P1', 'P2', 'L1', 'L2', 'L3', 'L4', 'I1', 'O1', 'G1']
  );
  const post = compileV4(JSON.parse(JSON.stringify(POST_J)), POST_X);
  const postChecks = buildV4Checks(post.program, POST_X, post);
  assert.deepEqual(postChecks.map((c) => c.id), ['PART', 'D1', 'D2', 'O1', 'G1']);
  assert.match(postChecks.find((c) => c.id === 'G1').text, /ONLY direct posteriors/);
});

test('witness: genuine receipt verifies; spread clone is unauthenticated', async (t) => {
  mockWitnessFetch(t);
  const c = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  const witness = await checkV4SemanticEntailment(c.program, STD_X, c);
  assert.equal(witness.entailed, true);
  assert.deepEqual(verifyV4Witness(witness, c.program, STD_X), { ok: true });

  const clone = { ...witness, checks: [...witness.checks] };
  const v = verifyV4Witness(clone, c.program, STD_X);
  assert.equal(v.ok, false);
  assert.equal(v.code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED');
});

test('solveV4: exact answer releases behind a verified witness', async (t) => {
  mockWitnessFetch(t);
  const c = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  const res = await solveV4(c.program, STD_X, c);
  assert.equal(res.status, 'ANSWER');
  assert.equal(R.toString(res.value), STD_ANSWER);
  assert.equal(res.witness.entailed, true);
});

test('solveV4: forged precomputed witness refuses unauthenticated', async (t) => {
  forbidFetch(t);
  const c = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  const res = await solveV4(c.program, STD_X, {
    ...c,
    witness: { entailed: true, code: null, reason: '', checks: [] },
  });
  assert.equal(res.status, 'REFUSED');
  assert.equal(res.code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED');
});

// ---------- R3 Fix 1: query-embedded observation grounding -------------------
// The observed event is stated inside the question sentence itself; the
// extractor puts it in query.given with obs: []. compileV4 must ground an
// OBS on the query sentence span — and ONLY there.
const QOBS_X =
  'Every case is either Alpha or Beta. Among cases with a green mark, 7/11 are Alpha and 4/11 are Beta. ' +
  'A case with a green mark is drawn at random; what is the probability it is Alpha?';
const QOBS_J = {
  hyps: [
    { id: 'h0', name: 'Alpha', span: 'either Alpha or Beta' },
    { id: 'h1', name: 'Beta', span: 'either Alpha or Beta' },
  ],
  events: [{ id: 'e0', name: 'green mark', span: 'with a green mark, 7/11' }],
  priors: [],
  liks: [],
  posteriors: [
    { hyp: 'h0', ev: 'e0', pol: 'POS', value: '7/11', span: 'with a green mark, 7/11 are Alpha', ctx: 'base' },
    { hyp: 'h1', ev: 'e0', pol: 'POS', value: '4/11', span: '4/11 are Beta', ctx: 'base' },
  ],
  obs: [],
  condIndep: [],
  detLinks: [],
  nonAsserted: [],
  partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
};

test('compileV4 (R3 Fix 1): query-sentence observation grounds OBS on the ? span', () => {
  const c = compileV4(JSON.parse(JSON.stringify(QOBS_J)), QOBS_X);
  assert.deepEqual(c.errors, []);
  const obs = c.program.decls.filter((d) => d.op === 'OBS');
  assert.equal(obs.length, 1);
  assert.equal(obs[0].ev, 'e0');
  assert.equal(obs[0].pol, 'POS');
  assert.equal(
    QOBS_X.slice(obs[0].span.s, obs[0].span.e),
    'A case with a green mark is drawn at random; what is the probability it is Alpha?'
  );
  const decision = solve(c.program, null);
  assert.equal(decision.status, 'ANSWER');
  assert.equal(R.toString(decision.value), '7/11');
});

test('compileV4 (R3 Fix 1) negative: unmentioned given still refuses MISSING_OBSERVATION_WITNESS', () => {
  // same stated posteriors, but the question sentence never mentions the
  // green mark — a hallucinated given must NOT be grounded
  const X =
    'Every case is either Alpha or Beta. Among cases with a green mark, 7/11 are Alpha and 4/11 are Beta. ' +
    'What is the probability it is Alpha?';
  const c = compileV4(JSON.parse(JSON.stringify(QOBS_J)), X);
  assert.deepEqual(c.errors, []);
  assert.equal(c.program.decls.filter((d) => d.op === 'OBS').length, 0);
  const decision = solve(c.program, null);
  assert.equal(decision.status, 'REFUSED');
  assert.equal(decision.code, 'MISSING_OBSERVATION_WITNESS');
});

test('verifyV4 PASS (R3 Fix 1): query-embedded posterior observation solves end-to-end', async (t) => {
  mockWitnessFetch(t); // O1 still audits query.given against raw X
  const res = await verifyV4({ X: QOBS_X, extract: semanticExtractor(QOBS_J, QOBS_X) });
  assert.equal(res.verdict, 'PASS', JSON.stringify({ code: res.code, details: res.details }));
  assert.equal(res.rational, '7/11');
  assert.equal(res.witness.entailed, true);
});

test('verifyV4 REFUSED (R3 Fix 1): hallucinated given refuses MISSING_OBSERVATION_WITNESS', async (t) => {
  const fetch = forbidFetch(t);
  const X =
    'Every case is either Alpha or Beta. Among cases with a green mark, 7/11 are Alpha and 4/11 are Beta. ' +
    'What is the probability it is Alpha?';
  const res = await verifyV4({ X, extract: semanticExtractor(QOBS_J, X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'MISSING_OBSERVATION_WITNESS');
  assert.equal(fetch.mock.callCount(), 0);
});

// ---------- R3 Fix 2: coherent within-group counterfactual swaps -------------
test('coherent swaps (R3 Fix 2): PRIOR<->PRIOR then LIK<->LIK, never cross-group', () => {
  const c = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  assert.deepEqual(c.errors, []);
  const { bound, sites, assigns } = coherentSwapsForProgram(c.program, STD_X, 2);
  const opOfSite = new Map(bound.map((b) => [b.siteIdx, b.decl.op]));
  const declOfSite = new Map(bound.map((b) => [b.siteIdx, b.decl]));
  assert.equal(assigns.length, 2);
  const movedOps = assigns.map((assign) => {
    const moved = assign.map((src, i) => (src === i ? null : i)).filter((v) => v !== null);
    assert.equal(moved.length, 2, 'each swap moves exactly two sites');
    assert.equal(opOfSite.get(moved[0]), opOfSite.get(moved[1]), 'same owning factor op');
    return opOfSite.get(moved[0]);
  });
  assert.deepEqual(movedOps, ['PRIOR', 'LIK']);
  // the LIK swap is the same event across distinct hypotheses
  const likMoved = assigns[1].map((src, i) => (src === i ? null : i)).filter((v) => v !== null);
  assert.equal(declOfSite.get(likMoved[0]).ev, declOfSite.get(likMoved[1]).ev);
  assert.notEqual(declOfSite.get(likMoved[0]).hyp, declOfSite.get(likMoved[1]).hyp);

  // every generatable swap preserves sum(PRIOR) === 1 and stays in-group
  const all = coherentSwapsForProgram(c.program, STD_X, 99).assigns;
  assert.ok(all.length >= 2);
  const priorIdx = bound.filter((b) => b.decl.op === 'PRIOR').map((b) => b.siteIdx);
  for (const assign of all) {
    const sum = priorIdx
      .map((i) => sites[assign[i]].value)
      .reduce((acc, v) => R.add(acc, v), R.zero());
    assert.ok(R.eq(sum, R.one()), 'sum(PRIOR) stays 1 under every swap');
    for (let i = 0; i < assign.length; i++) {
      if (assign[i] !== i) {
        assert.equal(opOfSite.get(i), opOfSite.get(assign[i]), 'no cross-group site move');
      }
    }
  }
});

test('coherent swaps (R3 Fix 2): POSTERIOR pairs stay in-group; single literal needs none', () => {
  const post = compileV4(JSON.parse(JSON.stringify(POST_J)), POST_X);
  assert.deepEqual(post.errors, []);
  const { bound, assigns } = coherentSwapsForProgram(post.program, POST_X, 2);
  const opOfSite = new Map(bound.map((b) => [b.siteIdx, b.decl.op]));
  assert.equal(assigns.length, 1);
  const moved = assigns[0].map((src, i) => (src === i ? null : i)).filter((v) => v !== null);
  assert.deepEqual(moved.map((i) => opOfSite.get(i)), ['POSTERIOR_STATED', 'POSTERIOR_STATED']);

  // one single literal (one stated posterior) -> zero swaps
  const singleJ = JSON.parse(JSON.stringify(POST_J));
  singleJ.posteriors = singleJ.posteriors.slice(0, 1);
  const single = compileV4(singleJ, POST_X);
  assert.deepEqual(single.errors, []);
  assert.deepEqual(coherentSwapsForProgram(single.program, POST_X, 2).assigns, []);
});

test('verifyV4 REFUSED (R3 Fix 2): stale PRIOR value on a coherent swap still refuses', async (t) => {
  const fetch = forbidFetch(t);
  const res = await verifyV4({ X: STD_X, extract: positionalExtractor(STD_J, STD_X) });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.match(res.details?.detail ?? '', /PRIOR\|.*did not track literal permutation/);
  assert.equal(fetch.mock.callCount(), 0);
});

// ---------- R3 Fix 3: unambiguous token-subset name alignment ----------------
test('alignEntityNames (R3 Fix 3): unique token-subset expansions align; ambiguous fails closed', () => {
  const m = (obj) => new Map(Object.entries(obj));
  // ids reordered AND names expanded ("toxic" -> "toxic specimen")
  const aligned = alignEntityNames(
    m({ h0: 'toxic', h1: 'benign' }),
    m({ h0: 'benign specimen', h1: 'toxic specimen' })
  );
  assert.deepEqual(Object.fromEntries(aligned), { h0: 'h1', h1: 'h0' });
  // trimmed modifier ("sound cable section" -> "sound section")
  const trimmed = alignEntityNames(
    m({ h0: 'sound cable section', h1: 'frayed cable section' }),
    m({ h0: 'sound section', h1: 'frayed section' })
  );
  assert.deepEqual(Object.fromEntries(trimmed), { h0: 'h0', h1: 'h1' });
  // ambiguous: "red" alone matches both "red seal" and "red tag"
  assert.equal(
    alignEntityNames(m({ h0: 'red seal', h1: 'red tag' }), m({ h0: 'red', h1: 'blue' })),
    null
  );
  // disjoint names never align; count mismatch never aligns
  assert.equal(alignEntityNames(m({ h0: 'alpha' }), m({ h0: 'omega' })), null);
  assert.equal(alignEntityNames(m({ h0: 'alpha' }), m({ h0: 'alpha', h1: 'beta' })), null);
});

// STD shape with short names that gain a modifier on the counterfactual
function shortNamedStd() {
  const j = JSON.parse(JSON.stringify(STD_J));
  j.hyps[0].name = 'toxic';
  j.hyps[1].name = 'benign';
  j.events[0].name = 'mislabelled';
  return j;
}

test('verifyV4 PASS (R3 Fix 3): "toxic"->"toxic specimen" aligns when values track', async (t) => {
  mockWitnessFetch(t);
  const res = await verifyV4({
    X: STD_X,
    extract: swapMutatingExtractor(shortNamedStd(), STD_X, {
      mutate: (prog) => {
        prog.hypNames = { h0: 'toxic specimen', h1: 'benign specimen' };
        prog.evNames = { e0: 'mislabelled container' };
      },
    }),
  });
  assert.equal(res.verdict, 'PASS', JSON.stringify({ code: res.code, details: res.details }));
  assert.ok(res.swaps.length >= 1 && res.swaps.every((s) => s.ok), 'swaps ran and passed');
  assert.equal(res.rational, STD_ANSWER);
});

test('verifyV4 REFUSED (R3 Fix 3): aligned names do not excuse unswapped values', async (t) => {
  const fetch = forbidFetch(t);
  const res = await verifyV4({
    X: STD_X,
    extract: swapMutatingExtractor(shortNamedStd(), STD_X, {
      trackValues: false,
      mutate: (prog) => {
        prog.hypNames = { h0: 'toxic specimen', h1: 'benign specimen' };
        prog.evNames = { e0: 'mislabelled container' };
      },
    }),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.match(res.details?.detail ?? '', /did not track literal permutation/);
  assert.equal(fetch.mock.callCount(), 0);
});

test('verifyV4 REFUSED (R3 Fix 3): ambiguous shared-token names fail closed', async (t) => {
  const fetch = forbidFetch(t);
  const j = JSON.parse(JSON.stringify(STD_J));
  j.hyps[0].name = 'red seal';
  j.hyps[1].name = 'red tag';
  const res = await verifyV4({
    X: STD_X,
    extract: swapMutatingExtractor(j, STD_X, {
      mutate: (prog) => {
        // "red" could be either hypothesis — no unambiguous alignment
        prog.hypNames = { h0: 'red', h1: 'blue' };
      },
    }),
  });
  assert.equal(res.verdict, 'REFUSED');
  assert.equal(res.code, 'UNSTABLE_BINDING');
  assert.match(res.details?.detail ?? '', /vanished or ambiguous/);
  assert.equal(fetch.mock.callCount(), 0);
});
