// synth-bench :: seeded reverse-synthetic corpus generator (ARCH-SPEC §5.1-5.2
// Phase-0 scale). Formal factor graphs are sampled first, solved exactly by
// prism-kernel, then rendered into diverse prose with exact span bookkeeping —
// so every emitted bytecode program is pointer-grounded by construction.
// 200 solvable multi-hop items (multi-event joints, NEG polarity,
// marginalization targets, fraction-word literals) + 14 trap families.

import * as R from '../kernel/rational.js';
import { solve } from '../kernel/solve.js';
import { toProgram } from '../kernel/parser.js';
import { autoSentTags } from '../kernel/sentences.js';
import { makeRng } from './rng.js';
import { literalForms } from './words.js';

// The sentence-partition ledger is unconditional whenever X is supplied —
// generated programs carry a compiler-derived SENT_TAG ledger by
// construction. Tag semantics come from autoSentTags; the kernel re-verifies
// them against the text at solve time.
function withLedger(X, program) {
  const prog = toProgram(program);
  const tagLines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  const lines = program.split('\n');
  const askIdx = lines.findIndex((l) => l.startsWith('ASK_PROB'));
  lines.splice(askIdx < 0 ? lines.length - 1 : askIdx, 0, ...tagLines);
  return lines.join('\n');
}

// ---------- span-tracking text builder ----------
class XBuilder {
  constructor() {
    this.parts = [];
    this.pos = 0;
  }
  add(t) {
    this.parts.push(t);
    this.pos += t.length;
    return this;
  }
  marked(t) {
    const s = this.pos;
    this.add(t);
    return { s, e: this.pos };
  }
  mark() {
    return this.pos;
  }
  text() {
    return this.parts.join('');
  }
}

const HYP_NAMES = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta'];
const EV_NAMES = [
  'flag test', 'relay alarm', 'screening panel', 'audit marker',
  'sensor ping', 'quality gate', 'triage tag', 'watchdog alert',
];
const REGISTERS = ['clinical', 'market', 'sensor', 'defect', 'contract'];
const INTROS = {
  clinical: 'The clinical registry partitions cases into disjoint cohorts.',
  market: 'The quarterly audit partitioned accounts by segment.',
  sensor: 'The postmortem review bucketed incidents by subsystem.',
  defect: 'The line audit grouped units by origin line.',
  contract: 'The contract review split claims by clause type.',
};
const DISTRACTORS = [
  'The review covered 320 records across 4 sites.',
  'Two analysts verified each entry twice.',
  'The study ran for 6 weeks in total.',
  'An earlier phase sampled 1024 documents.',
  'Each site used the same 8 step protocol.',
  'The export took 12 minutes to generate.',
  'Nine reviewers rotated through the queue.',
  'A checksum pass ran on all 7 batches.',
];

function renderLiteral(rng, r, forcedKind = null) {
  const forms = literalForms(r);
  if (forcedKind) {
    const f = forms.find((x) => x.kind === forcedKind);
    if (f) return f;
  }
  const weights = forms.map((f) => [
    f,
    f.kind === 'words' ? 0.34 : f.kind === 'percent' ? 0.2 : f.kind === 'fraction' ? 0.18 : 0.14,
  ]);
  return rng.weighted(weights);
}

// ---------- clause emitters (each returns recorded spans) ----------
function emitPriorClause(b, rng, name, litText) {
  const v = rng.int(0, 3);
  let nameSpan, litSpan;
  if (v === 0) {
    nameSpan = b.marked(name);
    b.add(' accounts for ');
    litSpan = b.marked(litText);
  } else if (v === 1) {
    nameSpan = b.marked(name);
    b.add(' holds a ');
    litSpan = b.marked(litText);
    b.add(' share');
  } else if (v === 2) {
    litSpan = b.marked(litText);
    b.add(' of cases are ');
    nameSpan = b.marked(name);
  } else {
    nameSpan = b.marked(name);
    b.add(' represents ');
    litSpan = b.marked(litText);
    b.add(' of the population');
  }
  return { nameSpan, litSpan };
}

function emitLikClause(b, rng, evName, hypName, litText, firstEv) {
  const v = rng.int(0, 3);
  let hypSpan, litSpan, evSpan = null;
  if (v === 0) {
    b.add('when the case is ');
    hypSpan = b.marked(hypName);
    b.add(', the ');
    evSpan = b.marked(evName);
    b.add(' is positive ');
    litSpan = b.marked(litText);
    b.add(' of the time');
  } else if (v === 1) {
    b.add('under ');
    hypSpan = b.marked(hypName);
    b.add(', the ');
    evSpan = b.marked(evName);
    b.add(' fires ');
    litSpan = b.marked(litText);
  } else if (v === 2) {
    litSpan = b.marked(litText);
    b.add(' of ');
    hypSpan = b.marked(hypName);
    b.add(' cases trigger the ');
    evSpan = b.marked(evName);
  } else {
    b.add('given ');
    hypSpan = b.marked(hypName);
    b.add(', ');
    evSpan = b.marked(evName);
    b.add(' reads positive ');
    litSpan = b.marked(litText);
  }
  return { hypSpan, litSpan, evSpan };
}

function emitObsClause(b, rng, evName, pol) {
  const s = b.mark();
  let evSpan;
  const word = pol === 'POS' ? 'positive' : 'negative';
  if (rng.chance(0.5)) {
    b.add('A case was observed: the ');
    evSpan = b.marked(evName);
    b.add(` came back ${word}.`);
  } else {
    b.add('On this case the ');
    evSpan = b.marked(evName);
    b.add(` read ${word}.`);
  }
  return { span: { s, e: b.mark() }, evSpan };
}

function emitQuestion(b, rng, targetName, negated, targetIsEvent = false) {
  const s = b.mark();
  const v = rng.int(0, 2);
  if (targetIsEvent) {
    const phr = [
      `What is the probability the ${targetName} is positive?`,
      `Given the observation, how likely is the ${targetName} to fire?`,
      `Estimate the probability that the ${targetName} is positive.`,
    ][v];
    b.add(phr);
  } else if (negated) {
    b.add(`What is the probability the case is not ${targetName}?`);
  } else if (v === 0) {
    b.add(`What is the probability the case is ${targetName}?`);
  } else if (v === 1) {
    b.add(`Given the observation, how likely is ${targetName}?`);
  } else {
    b.add(`Estimate the probability of ${targetName} given the reading.`);
  }
  return { s, e: b.mark() };
}

// ---------- formal spec sampler ----------
function sampleSpec(rng, idNum) {
  const nH = rng.weighted([[2, 0.45], [3, 0.35], [4, 0.2]]);
  const names = rng.shuffle(HYP_NAMES).slice(0, nH);
  // priors: composition of denominator D into nH positive integer parts
  const D = rng.weighted([[10, 0.2], [20, 0.5], [100, 0.3]]);
  const cuts = new Set();
  while (cuts.size < nH - 1) cuts.add(rng.int(1, D - 1));
  const sorted = [0, ...[...cuts].sort((a, b) => a - b), D];
  const priors = {};
  names.forEach((nm, i) => {
    priors[`h${i}`] = R.make(BigInt(sorted[i + 1] - sorted[i]), BigInt(D));
  });
  const nEv = rng.weighted([[1, 0.3], [2, 0.45], [3, 0.25]]);
  const evNames = rng.shuffle(EV_NAMES).slice(0, nEv);
  const liks = {};
  for (let e = 0; e < nEv; e++) {
    liks[`e${e}`] = {};
    for (let h = 0; h < nH; h++) {
      const d = rng.pick([2, 4, 5, 8, 10, 12, 16, 20, 3, 6]);
      const n = rng.int(1, d - 1);
      liks[`e${e}`][`h${h}`] = R.make(BigInt(n), BigInt(d));
    }
  }
  // observations: 1-2 of the events (bias toward 2 for multi-hop)
  const nObs = Math.min(nEv, rng.weighted([[1, 0.4], [2, 0.6]]));
  const obsEvs = rng.shuffle([...Array(nEv).keys()]).slice(0, nObs);
  const obs = obsEvs.map((e) => ({
    ev: `e${e}`,
    pol: rng.chance(0.85) ? 'POS' : 'NEG',
  }));
  // target: mostly a hypothesis; sometimes an event (marginalization hop)
  const target =
    rng.chance(0.85)
      ? { kind: 'hyp', id: `h${rng.int(0, nH - 1)}` }
      : { kind: 'ev', id: `e${rng.int(0, nEv - 1)}` };
  const negated = target.kind === 'hyp' && rng.chance(0.12);
  return {
    nH, names, priors, nEv, evNames, liks, obs, target, negated,
    register: rng.pick(REGISTERS),
    // deterministic nudge so fraction-word coverage includes the spec'd forms
    forceWords: idNum % 7 === 0,
  };
}

// independent float recompute (cross-check, not used as ground truth)
function floatSolve(spec) {
  const w = spec.names.map((_, h) => {
    let p = R.toNumber(spec.priors[`h${h}`]);
    for (const g of spec.obs) {
      let l = R.toNumber(spec.liks[g.ev][`h${h}`]);
      if (g.pol === 'NEG') l = 1 - l;
      p *= l;
    }
    return p;
  });
  const z = w.reduce((a, b) => a + b, 0);
  if (spec.target.kind === 'hyp') {
    const h = Number(spec.target.id.slice(1));
    const v = w[h] / z;
    return spec.negated ? 1 - v : v;
  }
  // event target: a query conditioned on the queried event itself is a
  // tautology (exactly 1 on POS evidence, 0 on NEG), not a marginalization
  const selfGiven = spec.obs.find((g) => g.ev === spec.target.id);
  if (selfGiven) {
    const v = selfGiven.pol === 'POS' ? 1 : 0;
    return spec.negated ? 1 - v : v;
  }
  // event target: marginalize
  let acc = 0;
  for (let h = 0; h < spec.names.length; h++) {
    acc += (w[h] / z) * R.toNumber(spec.liks[spec.target.id][`h${h}`]);
  }
  return acc;
}

// ---------- renderer: spec -> {X, spans, program} ----------
function renderSolve(spec, rng, opts = {}) {
  const b = new XBuilder();
  const decls = [];
  const spans = { hyps: [], priors: [], evs: [], liks: [], obs: [] };
  const litKind = { count: {} };
  const pickLit = (r) => {
    const f = renderLiteral(rng, r, opts.forceLiteralKind ?? null);
    litKind.count[f.kind] = (litKind.count[f.kind] ?? 0) + 1;
    return f.text;
  };

  b.add(INTROS[spec.register]).add(' ');

  // ---- prior sentence ----
  const priorStart = b.mark();
  spec.names.forEach((nm, h) => {
    const lit = pickLit(spec.priors[`h${h}`]);
    const { nameSpan, litSpan } = emitPriorClause(b, rng, nm, lit);
    spans.hyps[h] = spans.hyps[h] ?? nameSpan;
    spans.priors[h] = litSpan;
    if (h < spec.names.length - 1) b.add(h === spec.names.length - 2 ? ', and ' : '; ');
    else b.add('. ');
  });
  const priorEnd = b.mark();

  // ---- likelihood sentences ----
  for (let e = 0; e < spec.nEv; e++) {
    const evName = spec.evNames[e];
    spec.names.forEach((nm, h) => {
      const lit = pickLit(spec.liks[`e${e}`][`h${h}`]);
      const { litSpan, evSpan } = emitLikClause(b, rng, evName, nm, lit);
      spans.liks[`e${e}:h${h}`] = litSpan;
      spans.evs[e] = spans.evs[e] ?? evSpan;
      if (h < spec.names.length - 1) b.add('; ');
      else b.add('. ');
    });
  }
  // ---- independence license (prose must assert what the program declares) ----
  // The kernel requires COND_INDEP for >=2 givens AND for a distinct event
  // target marginalized over any given — the synth truth model marginalizes,
  // which presumes that license, so the prose must assert it.
  const tgtIsEv = spec.target.kind === 'ev';
  const tgtDistinctEv = tgtIsEv && !spec.obs.some((g) => g.ev === spec.target.id);
  const licenseEvs = [
    ...spec.obs.map((g) => g.ev),
    ...(tgtDistinctEv ? [spec.target.id] : []),
  ];
  if (licenseEvs.length >= 2) {
    const evList = licenseEvs.map((e) => spec.evNames[Number(e.slice(1))]);
    const cs = b.mark();
    b.add(`The ${evList.join(' and ')} are recorded independently for each cohort. `);
    spans.condIndep = { s: cs, e: b.mark() };
  }

  // ---- distractor ----
  if (rng.chance(0.55)) b.add(rng.pick(DISTRACTORS)).add(' ');

  // ---- observations ----
  for (const g of spec.obs) {
    const { span, evSpan } = emitObsClause(b, rng, spec.evNames[Number(g.ev.slice(1))], g.pol);
    spans.obs.push({ ev: g.ev, pol: g.pol, span });
    spans.evs[Number(g.ev.slice(1))] = spans.evs[Number(g.ev.slice(1))] ?? evSpan;
  }

  // ---- question ----
  const tgtName = tgtIsEv
    ? spec.evNames[Number(spec.target.id.slice(1))]
    : spec.names[Number(spec.target.id.slice(1))];
  emitQuestion(b, rng, tgtName, spec.negated, tgtIsEv);

  const X = b.text();

  // ---- program emission ----
  spec.names.forEach((nm, h) => {
    decls.push(`HYP h${h} [${spans.hyps[h].s},${spans.hyps[h].e}]`);
  });
  decls.push(`PART_EXHAUSTIVE (${spec.names.map((_, h) => `h${h}`).join(' ')}) [${priorStart},${priorEnd}]`);
  for (let e = 0; e < spec.nEv; e++) {
    decls.push(`EVENT e${e} [${spans.evs[e].s},${spans.evs[e].e}]`);
  }
  spec.names.forEach((_, h) => {
    const v = R.toString(spec.priors[`h${h}`]);
    decls.push(`PRIOR h${h} ${v} ctx0 [${spans.priors[h].s},${spans.priors[h].e}]`);
  });
  for (let e = 0; e < spec.nEv; e++) {
    for (let h = 0; h < spec.names.length; h++) {
      const v = R.toString(spec.liks[`e${e}`][`h${h}`]);
      const sp = spans.liks[`e${e}:h${h}`];
      decls.push(`LIK e${e} POS GIVEN h${h} ${v} ctx0 [${sp.s},${sp.e}]`);
    }
  }
  for (const g of spans.obs) {
    decls.push(`OBS ${g.ev} ${g.pol} ctx0 [${g.span.s},${g.span.e}]`);
  }
  if (spans.condIndep) {
    const evs = licenseEvs.join(' ');
    const cs = spans.condIndep;
    decls.push(`COND_INDEP (${evs}) GIVEN * ctx0 [${cs.s},${cs.e}]`);
  }
  const given = spec.obs.map((g) => `${g.ev}:${g.pol}`).join(' ');
  decls.push(
    `ASK_PROB ${spec.negated ? 'NOT ' : ''}${spec.target.id} GIVEN (${given}) ctx0`
  );
  decls.push('END');

  return { X, program: withLedger(X, decls.join('\n')), spans, litKind };
}

// ---------- trap builders (each emits X + program + expected code) ----------
function trapMissingBaseRate(rng, spec) {
  // prose states likelihoods only; program carries no PRIORs
  const b = new XBuilder();
  const decls = [];
  const spans = { hyps: [], evs: [] };
  b.add(INTROS[spec.register]).add(' ');
  for (let e = 0; e < spec.nEv; e++) {
    spec.names.forEach((nm, h) => {
      const lit = renderLiteral(rng, spec.liks[`e${e}`][`h${h}`]).text;
      const { litSpan, evSpan, hypSpan } = emitLikClause(b, rng, spec.evNames[e], nm, lit);
      spans.hyps[h] = spans.hyps[h] ?? hypSpan;
      spans.evs[e] = spans.evs[e] ?? evSpan;
      decls.push(`LIK e${e} POS GIVEN h${h} ${R.toString(spec.liks[`e${e}`][`h${h}`])} ctx0 [${litSpan.s},${litSpan.e}]`);
      if (h < spec.names.length - 1) b.add('; ');
      else b.add('. ');
    });
  }
  const g = spec.obs[0];
  const { span } = emitObsClause(b, rng, spec.evNames[Number(g.ev.slice(1))], g.pol);
  const tgt = spec.names[Number(spec.target.id.slice(1))];
  emitQuestion(b, rng, tgt, false);
  spec.names.forEach((nm, h) => decls.unshift(`HYP h${h} [${spans.hyps[h].s},${spans.hyps[h].e}]`));
  for (let e = 0; e < spec.nEv; e++) {
    decls.unshift(`EVENT e${e} [${spans.evs[e].s},${spans.evs[e].e}]`);
  }
  decls.push(`OBS ${g.ev} ${g.pol} ctx0 [${span.s},${span.e}]`);
  decls.push(`ASK_PROB ${spec.target.id} GIVEN (${g.ev}:${g.pol}) ctx0`);
  decls.push('END');
  return { X: b.text(), program: withLedger(b.text(), decls.join('\n')), expected: 'MISSING_BASE_RATE' };
}

function trapPriorsNotSum1(rng, spec) {
  spec = { ...spec, obs: [{ ev: 'e0', pol: 'POS' }], target: { kind: 'hyp', id: 'h1' }, negated: false };
  const b = new XBuilder();
  const decls = [];
  b.add(INTROS[spec.register]).add(' ');
  // render priors that don't sum to 1: force first two to 3/5, 3/5 or 1/2+1/4
  const priors =
    spec.nH === 2
      ? [R.make(3, 5), R.make(3, 5)]
      : spec.names.map((_, h) => (h === 0 ? R.make(1, 2) : R.make(1, 2 * spec.names.length)));
  const spans = { hyps: [], priors: [] };
  spec.names.forEach((nm, h) => {
    const lit = renderLiteral(rng, priors[h]).text;
    const { nameSpan, litSpan } = emitPriorClause(b, rng, nm, lit);
    spans.hyps[h] = nameSpan;
    spans.priors[h] = litSpan;
    if (h < spec.names.length - 1) b.add('; ');
    else b.add('. ');
  });
  const e = 0;
  spec.names.forEach((nm, h) => {
    const lit = renderLiteral(rng, spec.liks[`e${e}`][`h${h}`]).text;
    const { litSpan, evSpan } = emitLikClause(b, rng, spec.evNames[e], nm, lit);
    spans[`lik${h}`] = litSpan;
    spans.ev = evSpan;
    if (h < spec.names.length - 1) b.add('; ');
    else b.add('. ');
  });
  const g = spec.obs[0];
  const { span } = emitObsClause(b, rng, spec.evNames[Number(g.ev.slice(1))], g.pol);
  emitQuestion(b, rng, spec.names[Number(spec.target.id.slice(1))], false);
  spec.names.forEach((_, h) => {
    decls.push(`HYP h${h} [${spans.hyps[h].s},${spans.hyps[h].e}]`);
    decls.push(`PRIOR h${h} ${R.toString(priors[h])} ctx0 [${spans.priors[h].s},${spans.priors[h].e}]`);
    const v = R.toString(spec.liks[`e${e}`][`h${h}`]);
    decls.push(`LIK e${e} POS GIVEN h${h} ${v} ctx0 [${spans[`lik${h}`].s},${spans[`lik${h}`].e}]`);
  });
  decls.unshift(`PART_EXHAUSTIVE (${spec.names.map((_, h) => `h${h}`).join(' ')}) [0,${b.mark()}]`);
  decls.unshift(`EVENT e${e} [${spans.ev.s},${spans.ev.e}]`);
  decls.push(`OBS ${g.ev} ${g.pol} ctx0 [${span.s},${span.e}]`);
  decls.push(`ASK_PROB ${spec.target.id} GIVEN (${g.ev}:${g.pol}) ctx0`);
  decls.push('END');
  return { X: b.text(), program: withLedger(b.text(), decls.join('\n')), expected: 'MISSING_BASE_RATE' };
}

function trapReversedConditional(rng, spec, variant = 'stated') {
  // text states P(H|E); query needs the factorization the other direction
  const b = new XBuilder();
  const decls = [];
  const spans = { hyps: [], priors: [] };
  b.add(INTROS[spec.register]).add(' ');
  spec.names.forEach((nm, h) => {
    const lit = renderLiteral(rng, spec.priors[`h${h}`]).text;
    const { nameSpan, litSpan } = emitPriorClause(b, rng, nm, lit);
    spans.hyps[h] = nameSpan;
    spans.priors[h] = litSpan;
    if (h < spec.names.length - 1) b.add('; ');
    else b.add('. ');
  });
  // stated posterior sentence: "Among flagged cases, four fifths are Alpha."
  const evName = spec.evNames[0];
  const postVals = spec.names.map((_, h) =>
    h === 0 ? R.make(4, 5) : R.make(1, 5 * (spec.names.length - 1))
  );
  const st = b.mark();
  b.add(`Among cases where the `);
  const evSpan = b.marked(evName);
  b.add(' is positive, ');
  const postSpans = spec.names.map((nm, h) => {
    const lit = renderLiteral(rng, postVals[h]).text;
    const ls = b.marked(lit);
    b.add(` are ${nm}`);
    if (h < spec.names.length - 1) b.add(' and ');
    return ls;
  });
  b.add('. ');
  const se = b.mark();
  // second event with proper liks, then two observations
  let ev1Span = null;
  spec.names.forEach((nm, h) => {
    const lit = renderLiteral(rng, spec.liks.e1[`h${h}`]).text;
    const { litSpan, evSpan: es } = emitLikClause(b, rng, spec.evNames[1], nm, lit);
    ev1Span = ev1Span ?? es;
    spans[`lik1${h}`] = litSpan;
    if (h < spec.names.length - 1) b.add('; ');
    else b.add('. ');
  });
  const o0 = emitObsClause(b, rng, evName, 'POS');
  const o1 = emitObsClause(b, rng, spec.evNames[1], 'POS');
  emitQuestion(b, rng, spec.names[0], false);
  spec.names.forEach((_, h) => {
    decls.push(`HYP h${h} [${spans.hyps[h].s},${spans.hyps[h].e}]`);
    decls.push(`PRIOR h${h} ${R.toString(spec.priors[`h${h}`])} ctx0 [${spans.priors[h].s},${spans.priors[h].e}]`);
    decls.push(`POSTERIOR_STATED h${h} GIVEN e0 POS ${R.toString(postVals[h])} ctx0 [${postSpans[h].s},${postSpans[h].e}]`);
    const v = R.toString(spec.liks.e1[`h${h}`]);
    decls.push(`LIK e1 POS GIVEN h${h} ${v} ctx0 [${spans[`lik1${h}`].s},${spans[`lik1${h}`].e}]`);
  });
  decls.unshift(`PART_EXHAUSTIVE (${spec.names.map((_, h) => `h${h}`).join(' ')}) [${st},${se}]`);
  decls.unshift(`EVENT e0 [${evSpan.s},${evSpan.e}]`);
  decls.unshift(`EVENT e1 [${ev1Span.s},${ev1Span.e}]`);
  decls.push(`OBS e0 POS ctx0 [${o0.span.s},${o0.span.e}]`);
  decls.push(`OBS e1 POS ctx0 [${o1.span.s},${o1.span.e}]`);
  decls.push(`COND_INDEP (e0 e1) GIVEN * ctx0 [${st},${se}]`);
  decls.push(`ASK_PROB h0 GIVEN (e0:POS e1:POS) ctx0`);
  decls.push('END');
  return { X: b.text(), program: withLedger(b.text(), decls.join('\n')), expected: 'REVERSED_CONDITIONAL' };
}

function trapUnlicensedIndependence(rng, spec) {
  // full spec with 2 obs, minus COND_INDEP
  const mod = { ...spec, obs: spec.obs.slice(0, 2) };
  if (mod.obs.length < 2) {
    mod.obs = [
      { ev: 'e0', pol: 'POS' },
      { ev: 'e1', pol: 'POS' },
    ];
  }
  const { X, program } = renderSolve(mod, rng);
  // strip the COND_INDEP line
  const stripped = program
    .split('\n')
    .filter((l) => !l.startsWith('COND_INDEP'))
    .join('\n');
  return { X, program: stripped, expected: 'UNLICENSED_INDEPENDENCE' };
}

function trapDuplicateEvidence(rng, spec) {
  const mod = { ...spec, obs: [spec.obs[0]] };
  const { X, program } = renderSolve(mod, rng);
  // shared-sensor double count: same observation ledgered twice
  const lines = program.split('\n');
  const obsLine = lines.find((l) => l.startsWith('OBS'));
  const obsIdx = lines.indexOf(obsLine);
  lines.splice(obsIdx + 1, 0, obsLine);
  return { X, program: lines.join('\n'), expected: 'DUPLICATE_EVIDENCE' };
}

function trapConflictingFigures(rng, spec) {
  const { X, program } = renderSolve(spec, rng);
  const lines = program.split('\n');
  // duplicate a PRIOR with a different rational (same key, incompatible value)
  const priorIdx = lines.findIndex((l) => l.startsWith('PRIOR h0 '));
  const p = lines[priorIdx];
  const m = p.match(/PRIOR h0 (\d+)\/(\d+) ctx0 \[(\d+),(\d+)\]/);
  const alt = R.make(BigInt(m[1]) + 1n > BigInt(m[2]) ? BigInt(m[1]) - 1n : BigInt(m[1]) + 1n, BigInt(m[2]));
  lines.splice(priorIdx + 1, 0, `PRIOR h0 ${R.toString(alt)} ctx0 [${m[3]},${m[4]}]`);
  return { X, program: lines.join('\n'), expected: 'CONFLICTING_FIGURES' };
}

function trapContextMismatch(rng, spec) {
  const { X, program } = renderSolve(spec, rng);
  const lines = program.split('\n');
  // move all LIK decls to ctx_pre while the query asks ctx0/final
  const out = lines.map((l) => (l.startsWith('LIK ') ? l.replace(' ctx0 ', ' ctx_pre ') : l));
  return { X, program: out.join('\n'), expected: 'CONTEXT_MISMATCH' };
}

function trapNonAsserted(rng, spec, reason, prefix) {
  const mod = { ...spec, obs: spec.obs.slice(0, 1) };
  const b = new XBuilder();
  const decls = [];
  const spans = { hyps: [], priors: [] };
  b.add(prefix).add(' ');
  spec.names.forEach((nm, h) => {
    const lit = renderLiteral(rng, spec.priors[`h${h}`]).text;
    const { nameSpan, litSpan } = emitPriorClause(b, rng, nm, lit);
    spans.hyps[h] = nameSpan;
    spans.priors[h] = litSpan;
    if (h < spec.names.length - 1) b.add('; ');
    else b.add('. ');
  });
  const scopedStart = spans.priors[0].s - Math.min(spans.priors[0].s, 60);
  const scopedEnd = spans.priors[spans.priors.length - 1].e;
  for (let e = 0; e < spec.nEv; e++) {
    spec.names.forEach((nm, h) => {
      const lit = renderLiteral(rng, spec.liks[`e${e}`][`h${h}`]).text;
      const { litSpan, evSpan } = emitLikClause(b, rng, spec.evNames[e], nm, lit);
      spans[`lik${e}:${h}`] = litSpan;
      spans.evs = spans.evs ?? [];
      spans.evs[e] = evSpan;
      if (h < spec.names.length - 1) b.add('; ');
      else b.add('. ');
    });
  }
  const g = spec.obs[0];
  const { span } = emitObsClause(b, rng, spec.evNames[Number(g.ev.slice(1))], g.pol);
  emitQuestion(b, rng, spec.names[Number(spec.target.id.slice(1))] ?? spec.names[0], false);
  spec.names.forEach((_, h) => {
    decls.push(`HYP h${h} [${spans.hyps[h].s},${spans.hyps[h].e}]`);
    decls.push(`PRIOR h${h} ${R.toString(spec.priors[`h${h}`])} ctx0 [${spans.priors[h].s},${spans.priors[h].e}]`);
  });
  decls.unshift(`PART_EXHAUSTIVE (${spec.names.map((_, h) => `h${h}`).join(' ')}) [${scopedStart},${scopedEnd}]`);
  for (let e = 0; e < spec.nEv; e++) {
    decls.unshift(`EVENT e${e} [${spans.evs[e].s},${spans.evs[e].e}]`);
    spec.names.forEach((_, h) => {
      const v = R.toString(spec.liks[`e${e}`][`h${h}`]);
      const sp = spans[`lik${e}:${h}`];
      decls.push(`LIK e${e} POS GIVEN h${h} ${v} ctx0 [${sp.s},${sp.e}]`);
    });
  }
  decls.push(`NON_ASSERTED [${scopedStart},${scopedEnd}] ${reason}`);
  decls.push(`OBS ${g.ev} ${g.pol} ctx0 [${span.s},${span.e}]`);
  decls.push(`ASK_PROB ${spec.target.id} GIVEN (${g.ev}:${g.pol}) ctx0`);
  decls.push('END');
  return { X: b.text(), program: withLedger(b.text(), decls.join('\n')), expected: 'SCOPED_OR_RETRACTED_PREMISE' };
}

function trapStatedVsComputed(rng, spec) {
  const mod = {
    ...spec,
    obs: spec.obs.slice(0, 1),
    target: { kind: 'hyp', id: 'h0' },
    negated: false,
  };
  const { X, program, spans } = renderSolve(mod, rng);
  const lines = program.split('\n');
  // stated posterior that contradicts the computed one: pick a different value
  const g = spec.obs[0];
  const h0span = spans.priors[0];
  const wrong = R.make(1, 9);
  lines.splice(
    lines.findIndex((l) => l.startsWith('OBS')),
    0,
    `POSTERIOR_STATED ${spec.target.id} GIVEN ${g.ev} ${g.pol} ${R.toString(wrong)} ctx0 [${h0span.s},${h0span.e}]`
  );
  return { X, program: lines.join('\n'), expected: 'CONFLICTING_FIGURES' };
}

function trapBulmaSwap(rng, spec) {
  // gate-level trap: parenthetical owner-swap phrasing; the solvable program
  // is correct — the GATE verdict differs by extractor quality.
  const b = new XBuilder();
  const decls = [];
  const spans = { hyps: [], priors: [] };
  b.add('In the last audit, ');
  const a = spec.names[0];
  const bt = spec.names[1];
  const aSpan = b.marked(a);
  b.add(', compared to the ');
  const litB = renderLiteral(rng, spec.priors.h1).text;
  const litBSpan = b.marked(litB);
  b.add(' share for ');
  const bSpan = b.marked(bt);
  b.add(', accounted for ');
  const litA = renderLiteral(rng, spec.priors.h0).text;
  const litASpan = b.marked(litA);
  b.add(' of flagged cases. ');
  // likelihood sentence
  spec.names.slice(0, 2).forEach((nm, h) => {
    const lit = renderLiteral(rng, spec.liks.e0[`h${h}`]).text;
    const { litSpan } = emitLikClause(b, rng, spec.evNames[0], nm, lit);
    spans[`lik${h}`] = litSpan;
    b.add('. ');
  });
  const g = { ev: 'e0', pol: 'POS' };
  const { span: obsSpan, evSpan } = emitObsClause(b, rng, spec.evNames[0], 'POS');
  emitQuestion(b, rng, a, false);
  const priorStart = aSpan.s;
  const priorEnd = litASpan.e;
  decls.push(`HYP h0 [${aSpan.s},${aSpan.e}]`);
  decls.push(`HYP h1 [${bSpan.s},${bSpan.e}]`);
  decls.push(`PART_EXHAUSTIVE (h0 h1) [${priorStart},${priorEnd}]`);
  decls.push(`EVENT e0 [${evSpan.s},${evSpan.e}]`);
  decls.push(`PRIOR h0 ${R.toString(spec.priors.h0)} ctx0 [${litASpan.s},${litASpan.e}]`);
  decls.push(`PRIOR h1 ${R.toString(spec.priors.h1)} ctx0 [${litBSpan.s},${litBSpan.e}]`);
  [0, 1].forEach((h) => {
    const v = R.toString(spec.liks.e0[`h${h}`]);
    decls.push(`LIK e0 POS GIVEN h${h} ${v} ctx0 [${spans[`lik${h}`].s},${spans[`lik${h}`].e}]`);
  });
  decls.push(`OBS e0 POS ctx0 [${obsSpan.s},${obsSpan.e}]`);
  decls.push(`ASK_PROB h0 GIVEN (e0:POS) ctx0`);
  decls.push('END');
  return { X: b.text(), program: withLedger(b.text(), decls.join('\n')), expected: 'UNSTABLE_BINDING', gateLevel: true };
}

const TRAP_BUILDERS = [
  ['missing_base_rate', trapMissingBaseRate],
  ['priors_not_sum_1', trapPriorsNotSum1],
  ['reversed_conditional', (r, s) => trapReversedConditional(r, s, 'stated')],
  ['only_if_direction', (r, s) => trapReversedConditional(r, s, 'only_if')],
  ['unlicensed_independence', trapUnlicensedIndependence],
  ['duplicate_sensor', trapDuplicateEvidence],
  ['conflicting_figures', trapConflictingFigures],
  ['context_mismatch', trapContextMismatch],
  ['retracted_premise', (r, s) => trapNonAsserted(r, s, 'RETRACTED', 'The obsolete 1998 manual, since retracted, states the following base rates.')],
  ['quoted_premise', (r, s) => trapNonAsserted(r, s, 'QUOTED', "A competitor's filing, quoted verbatim and unverified, lists these figures.")],
  ['hypothetical_premise', (r, s) => trapNonAsserted(r, s, 'HYPOTHETICAL', 'Suppose, purely as a hypothetical, that the base rates were as follows.')],
  ['hedged_approx', (r, s) => trapNonAsserted(r, s, 'HEDGED_APPROX', 'The memo gives only rough approximate figures.')],
  ['stated_vs_computed', trapStatedVsComputed],
  ['bulma_owner_swap', trapBulmaSwap],
];

// ---------- top-level ----------
export function generateBench({ seed = 20261122, count = 200 } = {}) {
  const rng = makeRng(seed);
  const items = [];
  const stats = { literalKinds: {}, registers: {}, multiHop: 0, traps: 0 };
  const seen = new Set();

  for (let i = 0; i < count; i++) {
    let spec = sampleSpec(rng, i);
    const opts = {};
    // force the two spec'd word-forms into the corpus deterministically
    if (i === 3) {
      spec.liks.e0.h0 = R.make(5, 8);
      opts.forceLiteralKind = 'words';
    }
    if (i === 11) {
      spec.liks.e0.h0 = R.make(7, 12);
      opts.forceLiteralKind = 'words';
    }
    if (spec.forceWords) opts.forceLiteralKind = 'words';
    const { X, program, litKind } = renderSolve(spec, rng, opts);
    const decision = solve(program, X);
    if (decision.status !== 'ANSWER') {
      // resample rather than emit a broken item
      i--;
      continue;
    }
    // a query conditioned on the queried event itself is a degenerate
    // tautology item — carries no discrimination signal; resample
    if (decision.via === 'TAUTOLOGY') {
      i--;
      continue;
    }
    const float = floatSolve(spec);
    const exact = R.toNumber(decision.value);
    if (Math.abs(float - exact) > 1e-9) {
      throw new Error(`kernel/float mismatch on item ${i}: ${R.toString(decision.value)} vs ${float}`);
    }
    if (spec.obs.length >= 2 || spec.target.kind === 'ev') stats.multiHop++;
    const sig = X.slice(0, 80);
    if (seen.has(sig)) {
      i--;
      continue;
    }
    seen.add(sig);
    items.push({
      id: `SYN-${String(i).padStart(4, '0')}`,
      kind: 'solve',
      X,
      program,
      expected: { status: 'ANSWER', rational: R.toString(decision.value) },
      floatCheck: float,
      meta: {
        register: spec.register,
        nH: spec.nH,
        nEv: spec.nEv,
        nObs: spec.obs.length,
        negated: spec.negated,
        targetKind: spec.target.kind,
        literalKinds: litKind.count,
      },
    });
    stats.registers[spec.register] = (stats.registers[spec.register] ?? 0) + 1;
    for (const [k, v] of Object.entries(litKind.count)) {
      stats.literalKinds[k] = (stats.literalKinds[k] ?? 0) + v;
    }
  }

  // traps: ≥14, one per family
  TRAP_BUILDERS.forEach(([family, builder], t) => {
    const spec = sampleSpec(rng, 1000 + t);
    if (spec.nEv < 2 && ['reversed_conditional', 'only_if_direction', 'unlicensed_independence'].includes(family)) {
      spec.nEv = 2;
      spec.evNames = rng.shuffle(EV_NAMES).slice(0, 2);
      spec.liks.e1 = {};
      spec.names.forEach((_, h) => {
        spec.liks.e1[`h${h}`] = R.make(BigInt(rng.int(1, 7)), 10n);
      });
      spec.obs = [
        { ev: 'e0', pol: 'POS' },
        { ev: 'e1', pol: 'POS' },
      ];
    }
    const out = builder(rng, spec);
    let decision = null;
    if (!out.gateLevel) {
      try {
        decision = solve(out.program, out.X);
      } catch (e) {
        throw new Error(`trap ${family} crashed: ${e.message}\n${out.program}\n---\n${out.X}`);
      }
    }
    if (decision && decision.code !== out.expected) {
      throw new Error(`trap ${family}: expected ${out.expected}, kernel gave ${decision.status}:${decision.code}`);
    }
    items.push({
      id: `TRAP-${String(t).padStart(2, '0')}`,
      kind: out.gateLevel ? 'gate' : 'trap',
      trapFamily: family,
      X: out.X,
      program: out.program,
      expected: { status: 'REFUSED', code: out.expected },
      meta: { family },
    });
    stats.traps++;
  });

  // fraction-word coverage check: the spec'd forms must literally appear
  stats.hasFiveEighths = items.some((it) => it.X.includes('five eighths'));
  stats.hasSevenTwelfths = items.some((it) => it.X.includes('seven twelfths'));
  return { items, stats };
}
