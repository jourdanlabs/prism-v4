// Round-3 hardening regressions (Toph-authored, greenhouse domain).
// Covers: fail-closed name-free materiality + structural premise doubt (F2),
// the unhookable sealed-witness brand (W1), and prototype-pollution-immune
// witness options (W2). All passages below are fresh wording written for
// this suite.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags, premiseDoubt, sentenceMateriality } from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import {
  checkSemanticEntailment,
  solveWithWitness,
  verifyEntailmentWitness,
  WITNESS_MODEL,
  passageSha256,
  programCanonicalSha256,
} from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

// Self-contained chassis builder (also serialized into subprocess helpers).
function buildGreen({ link, op = 'EQUIV', pre = [], post = [] }, deps) {
  const { toProgram: parse, autoSentTags: tags } = deps;
  const S1 = 'Every terrarium is either Coral or Slate.';
  const S2 = 'Coral makes up 25% of terrariums and Slate makes up 75%.';
  const S3 = 'The mister sprays for 50% of Coral terrariums and 20% of Slate terrariums.';
  const S4 = 'The vent opens for 50% of Coral terrariums and 20% of Slate terrariums.';
  const OBS = 'This terrarium was checked with its mister spraying.';
  const Q = 'What is the probability its vent opened?';
  const X = [S1, S2, S3, S4, ...pre, link, ...post, OBS, Q].join(' ');
  const at = (s, from = 0) => { const i = X.indexOf(s, from); if (i < 0) throw new Error('needle ' + s); return `[${i},${i + s.length}]`; };
  const s4 = X.indexOf(S4);
  const bc = `HYP h0 ${at('Coral')}
HYP h1 ${at('Slate')}
PART_EXHAUSTIVE (h0 h1) ${at('Coral or Slate')}
EVENT e0 ${at('mister')}
EVENT e1 ${at('vent')}
PRIOR h0 1/4 ctx0 ${at('25% of terrariums')}
PRIOR h1 3/4 ctx0 ${at('75%.')}
LIK e0 POS GIVEN h0 1/2 ctx0 ${at('50% of Coral')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Slate')}
LIK e1 POS GIVEN h0 1/2 ctx0 ${at('50% of Coral', s4)}
LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Slate', s4)}
DET_LINK e0 ${op} e1 ctx0 ${at(link)}
OBS e0 POS ctx0 ${at(OBS)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = parse(bc);
  const tagLines = tags(X, prog).map((t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`);
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tagLines.join('\n')}\n$2`) };
}

const DEPS = { toProgram, autoSentTags };
const build = (spec) => buildGreen(spec, DEPS);
const LINK = 'The mister and vent share a manifold.';
const NAMES = ['mister', 'vent', 'Coral', 'Slate'];
const mat = (text) => sentenceMateriality({ text, s: 0, e: text.length }, NAMES);

// ---------------------------------------------------------------- F2 units
test('F2: fresh name-free retractions are material and carry premise doubt', () => {
  const traps = [
    'However, that arrangement was scrapped during the rebuild.',
    'Growers have since unplugged them and fitted separate timers.',
    'The shared wiring was swapped out weeks ago.',
    'The nursery bulletin about the linkage proved unreliable.',
    'Inspectors dispute that.',
    'Nobody on the crew believes the old hookup note.',
    'Neither bench follows the old hookup any longer.',
  ];
  for (const t of traps) {
    assert.equal(mat(t).kind, 'material', t);
    assert.ok(premiseDoubt(t), `premiseDoubt should fire: ${t}`);
  }
  // layer-(b)-only back-reference: material even without a doubt predicate
  assert.equal(mat('In fact, this never held outside the trial bench.').kind, 'material');
  assert.equal(mat('They stopped being paired years ago.').kind, 'material');
});

test('F2: concrete scene sentences stay inert fluff', () => {
  const inert = [
    'The glass panels were washed on Tuesday.',
    'There are 12 spare nozzles in the potting shed.',
    'The gravel paths were raked flat before opening.',
    'This terrarium sits beside the fern bench.',
    'Four volunteers rotated through the morning shift roster.',
    'The seed trays were stacked beside the sink.',
  ];
  for (const t of inert) {
    assert.equal(mat(t).kind, 'fluff', t);
    assert.equal(premiseDoubt(t), null, t);
  }
});

test('F2: expletive outer frame is not premise doubt', () => {
  assert.equal(premiseDoubt('It is rumored that the mister and vent always run together.'), null);
  assert.equal(premiseDoubt('It is likely that the mister and vent share a manifold.'), null);
});

// ------------------------------------------------------------ F2 end-to-end
const REFUSALS = [
  { id: 'scrapped-arrangement', post: ['However, that arrangement was scrapped during the rebuild.'] },
  { id: 'unplugged-them', post: ['Growers have since unplugged them and fitted separate timers.'] },
  { id: 'wiring-swapped', post: ['The shared wiring was swapped out weeks ago.'] },
  { id: 'bulletin-unreliable', post: ['The nursery bulletin about the linkage proved unreliable.'] },
  { id: 'dispute-that', post: ['Inspectors dispute that.'] },
  { id: 'never-held', post: ['In fact, this never held outside the trial bench.'] },
  { id: 'doubt-before', pre: ['Nobody on the crew believes the old hookup note.'] },
];

for (const c of REFUSALS) {
  test(`F2 e2e ${c.id}: kernel and offline gate refuse`, async () => {
    const { X, bc } = build({ link: LINK, pre: c.pre ?? [], post: c.post ?? [] });
    const prog = toProgram(bc, X.length);
    assert.equal(solve(prog, X).status, 'REFUSED');
    const gate = await verifyInvariance({ X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true });
    assert.equal(gate.verdict, 'REFUSED');
  });
}

test('F2 e2e: a retraction hidden under a FLUFF tag still refuses', () => {
  const post = 'Growers have since unplugged them and fitted separate timers.';
  const { X, bc } = build({ link: LINK, post: [post] });
  const prog = toProgram(bc, X.length);
  const start = X.indexOf(post);
  const segIdx = prog.decls.find((d) => d.op === 'SENT_TAG' && d.span && d.span.s <= start && start < d.span.e).idx;
  prog.decls.find((d) => d.op === 'SENT_TAG' && d.idx === segIdx).tag = 'FLUFF';
  assert.equal(solve(prog, X).status, 'REFUSED');
});

test('F2 e2e controls: clean link, anaphoric link, inert post-sentence all answer', async () => {
  const controls = [
    { link: LINK },
    { link: 'These two always operate together.' },
    { link: LINK, post: ['The glass panels were washed on Tuesday.'] },
    { link: LINK, post: ['There are 12 spare nozzles in the potting shed.'] },
  ];
  for (const spec of controls) {
    const { X, bc } = build({ pre: [], post: [], ...spec });
    const prog = toProgram(bc, X.length);
    const d = solve(prog, X);
    assert.equal(d.status, 'ANSWER', spec.link);
    assert.equal(`${d.value.n}/${d.value.d}`, '1/1');
    const gate = await verifyInvariance({ X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true });
    assert.equal(gate.verdict, 'PASS', spec.link);
  }
});

test('F2 e2e: rumored outer frame refuses as unlicensed, not retracted', () => {
  const link = 'It is rumored that the mister and vent always run together.';
  const { X, bc } = build({ link });
  const d = solve(toProgram(bc, X.length), X);
  assert.equal(d.status, 'REFUSED');
  assert.equal(d.code, 'UNLICENSED_INDEPENDENCE');
});

// --------------------------------------------------------------------- W1
test('W1: forged, cloned and proxied receipts are unauthenticated', async (t) => {
  const { X, bc } = build({ link: LINK });
  const prog = toProgram(bc, X.length);
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({ message: { content: JSON.stringify({ entailed: true, checks: ids.map((id) => ({ id, entailed: true })) }) } });
  });
  const genuine = await checkSemanticEntailment(prog, X);
  assert.equal(verifyEntailmentWitness(genuine, prog, X).ok, true, 'genuine receipt verifies');

  const forged = {
    witnessModel: WITNESS_MODEL, witnessHost: 'http://127.0.0.1:11434',
    passageSha256: passageSha256(X), programCanonicalSha256: programCanonicalSha256(prog),
    entailed: true, code: null, reason: '', checks: [],
  };
  assert.equal(verifyEntailmentWitness(forged, prog, X).code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED');
  assert.equal(verifyEntailmentWitness({ ...genuine }, prog, X).code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED', 'spread clone');
  assert.equal(verifyEntailmentWitness(JSON.parse(JSON.stringify(genuine)), prog, X).code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED', 'json clone');
  assert.equal(verifyEntailmentWitness(new Proxy(genuine, {}), prog, X).code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED', 'proxy wrap');
  assert.equal(verifyEntailmentWitness(Object.create(genuine), prog, X).code, 'ENTAILMENT_WITNESS_UNAUTHENTICATED', 'prototype child');
});

test('W1: pre-load intrinsic poisoning cannot forge the brand (subprocess)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'prism-w1-'));
  try {
    const repo = (p) => pathToFileURL(join(process.cwd(), p)).href;
    const helper = `
import { toProgram } from '${repo('src/kernel/parser.js')}';
import { autoSentTags } from '${repo('src/kernel/sentences.js')}';
export const buildGreen = ${buildGreen.toString()};
export const DEPS = { toProgram, autoSentTags };
`;
    writeFileSync(join(dir, 'helper.mjs'), helper);
    const main = `
const mode = process.argv[2];
const origHas = WeakSet.prototype.has;
if (mode === 'weakset') WeakSet.prototype.has = function () { return true; };
if (mode === 'call') Function.prototype.call = function (thisArg, ...a) { if (this === origHas) return true; return Reflect.apply(this, thisArg, a); };
if (mode === 'bind') Function.prototype.bind = function (thisArg, ...pre) { const f = this; if (thisArg === origHas) return () => true; return function (...a) { return Reflect.apply(f, thisArg, [...pre, ...a]); }; };
if (mode === 'reflect') Reflect.has = () => true;
const E = await import('${repo('src/invariance/entailment.js')}');
const { toProgram } = await import('${repo('src/kernel/parser.js')}');
const { buildGreen, DEPS } = await import(new URL('./helper.mjs', import.meta.url).href);
const { X, bc } = buildGreen({ link: 'The mister and vent share a manifold.' }, DEPS);
const prog = toProgram(bc, X.length);
const forged = { witnessModel: E.WITNESS_MODEL, witnessHost: 'http://127.0.0.1:11434', passageSha256: E.passageSha256(X), programCanonicalSha256: E.programCanonicalSha256(prog), entailed: true, code: null, reason: '', checks: [] };
console.log(E.verifyEntailmentWitness(forged, prog, X).code);
const sw = await E.solveWithWitness(prog, X, { entailmentWitness: forged });
console.log(sw.status + ' ' + sw.code);
`;
    writeFileSync(join(dir, 'main.mjs'), main);
    for (const mode of ['weakset', 'call', 'bind', 'reflect']) {
      const out = execFileSync(process.execPath, [join(dir, 'main.mjs'), mode], { encoding: 'utf8', cwd: dir }).trim().split('\n');
      assert.equal(out[0], 'ENTAILMENT_WITNESS_UNAUTHENTICATED', mode);
      assert.equal(out[1], 'REFUSED ENTAILMENT_WITNESS_UNAUTHENTICATED', mode);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// --------------------------------------------------------------------- W2
test('W2: prototype and __proto__ pollution cannot set witness flags', async (t) => {
  const { X, bc } = build({ link: LINK });
  const prog = toProgram(bc, X.length);
  const STUB = 'http://127.0.0.1:65531';
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    const prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({ message: { content: JSON.stringify({ entailed: true, checks: ids.map((id) => ({ id, entailed: true })) }) } });
  });
  try {
    // custom endpoint via polluted prototype: must stay refused
    Object.prototype.allowCustomWitnessEndpoint = true;
    assert.equal((await solveWithWitness(prog, X, { host: STUB })).status, 'REFUSED', 'proto allowCustom (solve)');
    const g1 = await verifyInvariance({ X, extract: makeSemanticExtractor(bc), ollamaHost: STUB });
    assert.equal(g1.verdict, 'REFUSED', 'proto allowCustom (gate)');
    delete Object.prototype.allowCustomWitnessEndpoint;

    // __proto__ payloads: direct, assigned and spread forms
    const parsed = JSON.parse(`{"host":"${STUB}","__proto__":{"allowCustomWitnessEndpoint":true}}`);
    assert.equal((await solveWithWitness(prog, X, parsed)).status, 'REFUSED', 'parsed __proto__');
    assert.equal((await solveWithWitness(prog, X, Object.assign({}, parsed))).status, 'REFUSED', 'assign __proto__');
    assert.equal((await solveWithWitness(prog, X, { ...parsed })).status, 'REFUSED', 'spread __proto__');

    // bench escape via polluted prototype: witness must still be demanded
    Object.prototype.offlineSyntheticBench = true;
    Object.prototype.requireEntailmentWitness = false;
    const refused = await solveWithWitness(prog, X, {});
    assert.equal(refused.status, 'ANSWER'); // stub witness certifies the CLEAN passage — the point is it was consulted
    assert.ok(refused.entailmentWitness, 'witness was consulted despite polluted bench flags');
    const g2 = await verifyInvariance({ X, extract: makeSemanticExtractor(bc) });
    assert.equal(g2.verdict, 'PASS');
    assert.ok(g2.slots.entailment, 'gate consulted the witness despite polluted bench flag');
    delete Object.prototype.offlineSyntheticBench;
    delete Object.prototype.requireEntailmentWitness;

    // documented own-property paths still work
    assert.equal((await solveWithWitness(prog, X, { host: STUB, allowCustomWitnessEndpoint: true })).status, 'ANSWER', 'own opt-in');
    const bench = await solveWithWitness(prog, X, { offlineSyntheticBench: true });
    assert.equal(bench.status, 'ANSWER', 'own bench flag');
    assert.equal(bench.entailmentWitness, null);
  } finally {
    delete Object.prototype.allowCustomWitnessEndpoint;
    delete Object.prototype.offlineSyntheticBench;
    delete Object.prototype.requireEntailmentWitness;
  }
});

test('W2: polluted bench flag cannot release a doubted passage', async (t) => {
  const { X, bc } = build({ link: LINK, post: ['Growers have since unplugged them and fitted separate timers.'] });
  const prog = toProgram(bc, X.length);
  t.mock.method(globalThis, 'fetch', async () => { throw new Error('witness down'); });
  try {
    Object.prototype.offlineSyntheticBench = true;
    assert.equal((await solveWithWitness(prog, X, {})).status, 'REFUSED');
    const gate = await verifyInvariance({ X, extract: makeSemanticExtractor(bc) });
    assert.equal(gate.verdict, 'REFUSED');
  } finally {
    delete Object.prototype.offlineSyntheticBench;
  }
});
