// Exact passage and bytecode chassis from the GrokBot-Pan 1e15cb5 gate probes.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags, sentenceMateriality, sentences, validateLicenseDecls } from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import { checkSemanticEntailment, solveWithWitness } from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { nameMaps, ownershipCheck } from '../src/invariance/ownership.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

const OBS_PAST = 'This cabinet was inspected with its beacon flashing.';
const OBS_NOW = 'This cabinet is currently inspected with its beacon flashing.';
const Q = 'What is the probability its chime rang?';
const S1 = 'Every cabinet is either Oak or Elm.';
const S2 = 'Oak makes up 30% of cabinets and Elm makes up 70%.';
const S3 = 'The beacon flashes for 50% of Oak cabinets and 10% of Elm cabinets.';
const S4 = 'The chime rings for 50% of Oak cabinets and 10% of Elm cabinets.';

function build({ link, op = 'EQUIV', obs = OBS_PAST, pre = [], post = [] }) {
  const X = [S1, S2, S3, S4, ...pre, link, ...post, obs, Q].join(' ');
  const at = (s, from = 0) => { const i = X.indexOf(s, from); if (i < 0) throw new Error('needle ' + s); return `[${i},${i + s.length}]`; };
  const s4 = X.indexOf(S4);
  const linkSpan = at(link);
  const lic = op === 'COND_INDEP' ? `COND_INDEP (e0 e1) GIVEN * ctx0 ${linkSpan}` : `DET_LINK e0 ${op} e1 ctx0 ${linkSpan}`;
  const bc = `HYP h0 ${at('Oak')}
HYP h1 ${at('Elm')}
PART_EXHAUSTIVE (h0 h1) ${at('Oak or Elm')}
EVENT e0 ${at('beacon')}
EVENT e1 ${at('chime')}
PRIOR h0 3/10 ctx0 ${at('30% of cabinets')}
PRIOR h1 7/10 ctx0 ${at('70%.')}
LIK e0 POS GIVEN h0 1/2 ctx0 ${at('50% of Oak')}
LIK e0 POS GIVEN h1 1/10 ctx0 ${at('10% of Elm')}
LIK e1 POS GIVEN h0 1/2 ctx0 ${at('50% of Oak', s4)}
LIK e1 POS GIVEN h1 1/10 ctx0 ${at('10% of Elm', s4)}
${lic}
OBS e0 POS ctx0 ${at(obs)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc);
  const tags = autoSentTags(X, prog).map((t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`);
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tags.join('\n')}\n$2`) };
}

// expect: 'REFUSE' (should refuse) or an exact answer string (control)
const CASES = [
  // --- scope qualifiers in new positions ---
  { id: 'S01-pilot-lead', cls: 'scope', link: 'In the pilot, the beacon and chime always flash together.', expect: 'REFUSE' },
  { id: 'S02-pilot-tail', cls: 'scope', link: 'The beacon and chime always flash together in the pilot.', expect: 'REFUSE' },
  { id: 'S03-among-oak', cls: 'scope', link: 'Among Oak cabinets, the beacon and chime always flash together.', expect: 'REFUSE' },
  { id: 'S04-except-maint', cls: 'scope', link: 'The beacon and chime always flash together, except during maintenance.', expect: 'REFUSE' },
  { id: 'S05-for-most', cls: 'scope', link: 'For most cabinets, the beacon and chime are wired to the same relay.', expect: 'REFUSE' },
  { id: 'S06-trial-phase-lead', cls: 'scope', link: 'During the trial phase, the beacon and chime share a relay.', expect: 'REFUSE' },
  { id: 'S07-test-batch-tail', cls: 'scope', link: 'The beacon and chime are hard-wired together in the test batch.', expect: 'REFUSE' },
  { id: 'S08-old-firmware', cls: 'scope', link: 'Under the old firmware, the beacon and chime share a relay.', expect: 'REFUSE' },
  // --- hedges / fluff ---
  { id: 'H01-generally', cls: 'hedge', link: 'The beacon and chime generally always flash together.', expect: 'REFUSE' },
  { id: 'H02-tend-to', cls: 'hedge', link: 'The beacon and chime tend to flash together.', expect: 'REFUSE' },
  { id: 'H03-reportedly', cls: 'hedge', link: 'The beacon and chime reportedly share a relay.', expect: 'REFUSE' },
  { id: 'H04-might-never-mutex', cls: 'hedge/modal', op: 'MUTEX', link: 'The beacon and chime might never both sound.', expect: 'REFUSE' },
  { id: 'H05-may-never-mutex', cls: 'hedge/modal', op: 'MUTEX', link: 'The beacon and chime may never both sound.', expect: 'REFUSE' },
  { id: 'H06-fluff-retraction-after', cls: 'hedge/fluff', link: 'The beacon and chime share a relay.', post: ['That claim was later withdrawn.'], expect: 'REFUSE' },
  { id: 'H07-fluff-unverified-after', cls: 'hedge/fluff', link: 'The beacon and chime share a relay.', post: ['Nobody has checked this on site.'], expect: 'REFUSE' },
  // --- tense shifts (present observation) ---
  { id: 'T01-always-flashed', cls: 'tense', obs: OBS_NOW, link: 'The beacon and chime always flashed together.', expect: 'REFUSE' },
  { id: 'T02-used-to-share', cls: 'tense', obs: OBS_NOW, link: 'The beacon and chime used to share a relay.', expect: 'REFUSE' },
  { id: 'T03-will-be-wired', cls: 'tense', obs: OBS_NOW, link: 'The beacon and chime will be wired to the same relay.', expect: 'REFUSE' },
  { id: 'T04-had-been-wired', cls: 'tense', obs: OBS_NOW, link: 'The beacon and chime had been wired to the same relay.', expect: 'REFUSE' },
  { id: 'T05-were-independent', cls: 'tense', op: 'COND_INDEP', obs: OBS_NOW, link: 'The beacon and chime were independent.', expect: 'REFUSE' },
  { id: 'T06-had-been-tested-indep', cls: 'tense', op: 'COND_INDEP', obs: OBS_NOW, link: 'The beacon and chime had been tested independently.', expect: 'REFUSE' },
  { id: 'T07-did-not-both', cls: 'tense', op: 'MUTEX', obs: OBS_NOW, link: 'The beacon and chime did not both sound.', expect: 'REFUSE' },
  { id: 'T08-whenever-flashed', cls: 'tense', op: 'IMPLIES', obs: OBS_NOW, link: 'Whenever the beacon flashed, the chime rang.', expect: 'REFUSE' },
  // --- negated context ---
  { id: 'N01-not-true-that', cls: 'negated', link: 'It is not true that the beacon and chime always flash together.', expect: 'REFUSE' },
  { id: 'N02-no-one-claims', cls: 'negated', link: 'No one claims the beacon and chime share a relay.', expect: 'REFUSE' },
  { id: 'N03-not-a-test-batch', cls: 'negated', obs: 'This cabinet is not a test-batch cabinet and was inspected with its beacon flashing.', link: 'In the test batch, the beacon and chime share a relay.', expect: 'REFUSE' },
  // --- anaphora: two/three-event ---
  { id: 'A01-these-two-third-pair', cls: 'anaphora', pre: ['Each cabinet also holds a fan and a heater.'], link: 'These two always activate together.', expect: 'REFUSE' },
  { id: 'A02-both-of-them-third-pair', cls: 'anaphora', pre: ['The fan and the heater are serviced yearly.'], link: 'Both of them share a relay.', expect: 'REFUSE' },
  { id: 'A03-the-former-latter', cls: 'anaphora', op: 'IMPLIES', link: 'The former always triggers the latter.', expect: 'REFUSE' },
  { id: 'A04-these-three', cls: 'anaphora', pre: ['A siren is also fitted.'], link: 'These three always activate together.', expect: 'REFUSE' },
  { id: 'A05-they-third-pair', cls: 'anaphora', pre: ['Its fan and heater are on one fuse.'], link: 'They are wired to the same relay.', expect: 'REFUSE' },
  { id: 'A06-these-two-signals-third-pair', cls: 'anaphora', pre: ['Each cabinet also holds a fan and a heater.'], link: 'These two signals always activate together.', expect: 'REFUSE' },
  { id: 'A07-the-pair-third-pair', cls: 'anaphora', pre: ['A fan and a heater sit beside them.'], link: 'The pair are wired to the same relay.', expect: 'REFUSE' },
  // --- conditionals / quoted / modal ---
  { id: 'C01-if-relay-installed', cls: 'conditional', link: 'If the relay is installed, the beacon and chime always flash together.', expect: 'REFUSE' },
  { id: 'C02-quoted-manual', cls: 'quoted', link: 'The manual says, "the beacon and chime always flash together."', expect: 'REFUSE' },
  { id: 'C03-could-share', cls: 'modal', link: 'The beacon and chime could share a relay.', expect: 'REFUSE' },
  { id: 'C04-when-door-shut', cls: 'conditional', link: 'The beacon and chime always flash together when the door is shut.', expect: 'REFUSE' },
  { id: 'C05-if-may', cls: 'modal', op: 'IMPLIES', link: 'If the beacon flashes, the chime may ring.', expect: 'REFUSE' },
  { id: 'C06-would-never-both', cls: 'modal', op: 'MUTEX', link: 'The beacon and chime would never both sound.', expect: 'REFUSE' },
  // --- unconditional controls ---
  { id: 'K01-equiv', cls: 'control', link: 'The beacon and chime always flash together.', expect: '1/1' },
  { id: 'K02-wired', cls: 'control', link: 'The beacon and chime are wired to the same relay.', expect: '1/1' },
  { id: 'K03-mutex', cls: 'control', op: 'MUTEX', link: 'The beacon and chime never both sound.', expect: '0/1' },
  { id: 'K04-indep', cls: 'control', op: 'COND_INDEP', link: 'The beacon and chime are conditionally independent.', expect: '41/110' },
  { id: 'K05-implies', cls: 'control', op: 'IMPLIES', link: 'Whenever the beacon flashes, the chime rings.', expect: '1/1' },
  { id: 'K06-these-two', cls: 'control', link: 'These two always activate together.', expect: '1/1' },
  { id: 'K09-these-two-signals', cls: 'control', link: 'These two signals always activate together.', expect: '1/1' },
  { id: 'K07-every-cabinet', cls: 'control', link: 'On every cabinet, the beacon and chime share a relay.', expect: '1/1' },
  { id: 'K08-present-obs-present-link', cls: 'control', obs: OBS_NOW, link: 'The beacon and chime are wired to the same relay.', expect: '1/1' },
];

const fmt = (value) => `${value.n}/${value.d}`;

test('GrokBot-Pan gate fixture contains all 39 refusals and all 9 controls', () => {
  assert.equal(CASES.filter((c) => c.expect === 'REFUSE').length, 39);
  assert.equal(CASES.filter((c) => c.expect !== 'REFUSE').length, 9);
  assert.equal(new Set(CASES.map((c) => c.id)).size, 48);
});

for (const c of CASES) {
  test(`GrokBot-Pan ${c.id}: ${c.expect}`, async (t) => {
    const { X, bc } = build(c);
    const prog = toProgram(bc, X.length);
    const decision = solve(prog, X);
    const gateOptions = { X, extract: makeSemanticExtractor(bc) };
    if (c.expect === 'REFUSE') {
      const fetch = t.mock.method(globalThis, 'fetch', async () => {
        throw new Error('deterministic refusal must skip the witness');
      });
      assert.equal(decision.status, 'REFUSED');
      assert.equal((await solveWithWitness(prog, X)).status, 'REFUSED');
      assert.equal((await verifyInvariance(gateOptions)).verdict, 'REFUSED');
      assert.equal(fetch.mock.callCount(), 0);
    } else {
      assert.equal(decision.status, 'ANSWER');
      assert.equal(fmt(decision.value), c.expect);
      const gate = await verifyInvariance({ ...gateOptions, offlineSyntheticBench: true });
      assert.equal(gate.verdict, 'PASS');
      assert.equal(fmt(gate.decision.value), c.expect);
    }
  });
}

test('witness rejects modal exclusion and post-license doubt without endpoint cooperation', async (t) => {
  const fetch = t.mock.method(globalThis, 'fetch', async () =>
    Response.json({ message: { content: JSON.stringify({
      entailed: true,
      checks: [{ id: 'L1', entailed: true }, { id: 'M1', entailed: true }],
    }) } })
  );
  for (const c of CASES.filter(({ id }) => /^H0[4-7]/.test(id))) {
    const { X, bc } = build(c);
    const witness = await checkSemanticEntailment(bc, X);
    assert.equal(witness.entailed, false, c.id);
    assert.equal(witness.code, c.op === 'MUTEX' ? 'MODAL_WRAPPER' : 'SCOPED_OR_RETRACTED_PREMISE');
  }
  assert.equal(fetch.mock.callCount(), 0);
});

test('doubt about a license stays material even when incorrectly ledgered as FLUFF', async () => {
  for (const post of [
    'No one has verified the wiring.',
    'The finding remains unconfirmed.',
    'The relay connection is still unchecked.',
    'That report remains disputed.',
    'Unverified on site.',
    'Verification is still pending.',
  ]) {
    const { X, bc } = build({ link: 'The beacon and chime share a relay.', post: [post] });
    const prog = toProgram(bc);
    const seg = sentences(X).find(({ text }) => text.includes(post));
    assert.equal(sentenceMateriality(seg, []).kind, 'material', post);
    prog.decls.find((d) => d.op === 'SENT_TAG' && d.idx === seg.idx).tag = 'FLUFF';
    assert.equal(validateLicenseDecls(X, prog).length, 1, post);
    assert.equal(ownershipCheck({ X, decls: prog.decls, ...nameMaps(X, prog) })[0].code,
      'SCOPED_OR_RETRACTED_PREMISE', post);
    assert.equal(solve(prog, X).code, 'SCOPED_OR_RETRACTED_PREMISE', post);
  }
});

test('only categorical never modals license MUTEX', () => {
  for (const modal of ['cannot', 'can never', 'will never', 'must never', 'never', 'could never']) {
    const { X, bc } = build({ link: `The beacon and chime ${modal} both sound.`, op: 'MUTEX' });
    const decision = solve(toProgram(bc), X);
    if (modal === 'could never') {
      assert.equal(decision.status, 'REFUSED');
    } else {
      assert.equal(decision.status, 'ANSWER', modal);
      assert.equal(fmt(decision.value), '0/1', modal);
    }
  }
});

test('historical contractions and independence forms cannot license current observations', () => {
  for (const [op, link] of [
    ['MUTEX', 'The beacon and chime didn’t both sound.'],
    ['IMPLIES', 'Whenever the beacon flashes, the chime rang.'],
    ['COND_INDEP', 'The beacon and chime was independent.'],
    ['COND_INDEP', 'The beacon and chime were tested independently.'],
    ['COND_INDEP', 'The beacon and chime operated independently.'],
  ]) {
    const { X, bc } = build({ link, op, obs: OBS_NOW });
    assert.equal(solve(toProgram(bc), X).status, 'REFUSED', link);
  }
  const { X, bc } = build({
    link: 'The beacon and chime are tested independently.', op: 'COND_INDEP', obs: OBS_NOW,
  });
  const decision = solve(toProgram(bc), X);
  assert.equal(decision.status, 'ANSWER');
  assert.equal(fmt(decision.value), '41/110');
});

test('a coordinated or plural noun phrase in the antecedent sentence blocks anaphora', () => {
  for (const ending of [
    '10% of Elm cabinets, beside a fan and a heater.',
    '10% of Elm cabinets; fans spin on every cabinet.',
    '10% of Elm cabinets, beside the fans.',
    '10% of Elm cabinets, beside fans.',
    '10% of Elm cabinets; a fan, a heater, a pump are nearby.',
  ]) {
    const { X, bc } = build({ link: 'These two always activate together.' });
    // Keep the original bytecode spans up to the affected declaration, then
    // bind the license directly at its new offset for the resolver check.
    const prog = toProgram(bc);
    const revised = X.replace('10% of Elm cabinets. These two', `${ending} These two`);
    const decl = prog.decls.find(({ op }) => op === 'DET_LINK');
    const start = revised.indexOf('These two');
    decl.span = { s: start, e: start + 'These two always activate together.'.length };
    assert.match(validateLicenseDecls(revised, prog)[0].reason, /does not reference/, ending);
  }
});
