// Bulma R5 final-gate regressions (Toph-authored, aquarium domain).
// Round 5 released 1/1 on four fresh residual-clause classes inside the
// grounded observation/premise sentence: a clause-final finite verb
// before a bare coordinator, a parenthetical counterexample, a verbless
// comma-absolute stating the other event's state, and a concessive
// gerund clause. Every sentence below is written fresh for this suite;
// no gate wording is reused. Controls (locative parenthetical, inert
// scene sister clause, verbless adjunct) must keep answering 1/1.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { toProgram } from '../src/kernel/parser.js';
import { autoSentTags } from '../src/kernel/sentences.js';
import { solve } from '../src/kernel/solve.js';
import {
  checkSemanticEntailment,
  solveWithWitness,
  verifyEntailmentWitness,
} from '../src/invariance/entailment.js';
import { verifyInvariance } from '../src/invariance/gate.js';
import { makeSemanticExtractor } from '../src/invariance/reference-extractors.js';

const OBS = 'this tank was checked with its beacon glowing';
const LINKS = {
  EQUIV: 'The beacon and pump share a sump.',
  IMPLIES: 'The beacon always triggers the pump.',
};

function build({ obsSentence, obsSpanText, rel = 'EQUIV', likSentence, wholeObsSpan = false }) {
  const S1 = 'Every tank is either Reef or Kelp.';
  const S2 = 'Reef makes up 25% of tanks and Kelp makes up 75%.';
  const S3 = likSentence ?? 'The beacon glows for 50% of Reef tanks and 20% of Kelp tanks.';
  const S4 = 'The pump runs for 50% of Reef tanks and 20% of Kelp tanks.';
  const Q = 'What is the probability its pump ran?';
  const X = [S1, S2, S3, S4, LINKS[rel], obsSentence, Q].join(' ');
  const at = (s, from = 0) => {
    const i = X.indexOf(s, from);
    if (i < 0) throw new Error('needle ' + s);
    return `[${i},${i + s.length}]`;
  };
  const s4 = X.indexOf(S4);
  const bc = `HYP h0 ${at('Reef')}
HYP h1 ${at('Kelp')}
PART_EXHAUSTIVE (h0 h1) ${at('Reef or Kelp')}
EVENT e0 ${at('beacon')}
EVENT e1 ${at('pump')}
PRIOR h0 1/4 ctx0 ${at('25% of tanks')}
PRIOR h1 3/4 ctx0 ${at('75%.')}
LIK e0 POS GIVEN h0 1/2 ctx0 ${at('50% of Reef')}
LIK e0 POS GIVEN h1 1/5 ctx0 ${at('20% of Kelp')}
LIK e1 POS GIVEN h0 1/2 ctx0 ${at('50% of Reef', s4)}
LIK e1 POS GIVEN h1 1/5 ctx0 ${at('20% of Kelp', s4)}
DET_LINK e0 ${rel} e1 ctx0 ${at(LINKS[rel])}
OBS e0 POS ctx0 ${at(wholeObsSpan ? obsSentence.replace(/\.$/, '') : obsSpanText)}
ASK_PROB e1 GIVEN (e0:POS) ctx0
END`;
  const prog = toProgram(bc);
  const tagLines = autoSentTags(X, prog).map(
    (t) => `SENT_TAG ${t.idx} ${t.tag}${t.ctx ? ` ${t.ctx}` : ''} [${t.span.s},${t.span.e}]`
  );
  return { X, bc: bc.replace(/(^|\n)(ASK_PROB)/, `$1${tagLines.join('\n')}\n$2`) };
}

const cla = `During the evening round ${OBS}`;
const TRAPS = [
  // clause-final finite verb before a bare coordinator: past '-ed' form
  { obsSentence: `Before sunrise only the beacon glowed and during the evening round ${OBS}.`,
    obsSpanText: `during the evening round ${OBS}` },
  // clause-final finite verb before a bare coordinator: irregular past
  { obsSentence: `Overnight the pump ran on its own and during the evening round ${OBS}.`,
    obsSpanText: `during the evening round ${OBS}` },
  { obsSentence: `Overnight only the pump ran and during the evening round ${OBS}.`,
    obsSpanText: `during the evening round ${OBS}` },
  // parenthetical and bracketed counterexamples after the observation
  { obsSentence: `${cla} (at dawn the beacon glowed by itself).`, obsSpanText: cla },
  { obsSentence: `${cla} [at dawn the beacon glowed by itself].`, obsSpanText: cla },
  // verbless comma-absolute stating the queried event's contrary state
  { obsSentence: `${cla}, its pump idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
  { obsSentence: `${cla}, the pump remaining idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
  { obsSentence: `Its pump idle for the whole round, ${cla.charAt(0).toLowerCase() + cla.slice(1)}.`,
    obsSpanText: cla.charAt(0).toLowerCase() + cla.slice(1), rel: 'IMPLIES' },
  // concessive / gerund clause, trailing and leading
  { obsSentence: `${cla} despite the pump staying idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
  { obsSentence: `${cla} although the pump stayed idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
  { obsSentence: `Despite the pump staying idle for the whole round, ${cla.charAt(0).toLowerCase() + cla.slice(1)}.`,
    obsSpanText: cla.charAt(0).toLowerCase() + cla.slice(1), rel: 'IMPLIES' },
  { obsSentence: `${cla} with its pump idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
  // parenthetical residual inside a likelihood sentence, OBS untouched
  { obsSentence: `${cla}.`, obsSpanText: cla,
    likSentence: 'The beacon glows for 50% of Reef tanks and 20% of Kelp tanks (at dawn it glowed by itself).' },
];
for (const [i, spec] of TRAPS.entries()) {
  test(`R5 residual clause inside the grounded sentence refuses (${i})`, async () => {
    const { X, bc } = build(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'REFUSED', spec.obsSentence);
    assert.equal(d.code, 'UNACCOUNTED_MATERIAL_SENTENCE', spec.obsSentence);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'REFUSED', spec.obsSentence);
  });
}

test('R5 harmless adjacent clauses inside the observation sentence still answer 1/1', async () => {
  const controls = [
    { obsSentence: `${cla} (near the side door).`, obsSpanText: cla },
    { obsSentence: `${cla} (the hatch is painted green).`, obsSpanText: cla },
    { obsSentence: `${cla}, and the hatch is painted green.`, obsSpanText: cla },
    { obsSentence: `In the evening round; ${OBS}.`, obsSpanText: OBS },
    { obsSentence: `${cla}.`, obsSpanText: cla },
  ];
  for (const spec of controls) {
    const { X, bc } = build(spec);
    const d = solve(toProgram(bc, X.length), X);
    assert.equal(d.status, 'ANSWER', spec.obsSentence);
    assert.equal(`${d.value.n}/${d.value.d}`, '1/1', spec.obsSentence);
    const gate = await verifyInvariance({
      X, extract: makeSemanticExtractor(bc), offlineSyntheticBench: true,
    });
    assert.equal(gate.verdict, 'PASS', spec.obsSentence);
  }
});

// ------------------------------------- witness: explicit R-checks -----
// With the OBS span stretched over the WHOLE compound sentence the
// kernel's overlap accounting is satisfied, so only the entailment
// witness stands between the smuggled clause and a release. The
// witness must name that clause in its batch and refuse even when the
// endpoint approves every check it was given.
test('R5 witness explicitly checks residual clauses and cannot pardon one', async (t) => {
  const variants = [
    { obsSentence: `${cla}, its pump idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
    { obsSentence: `${cla} (at dawn the beacon glowed by itself).`, obsSpanText: cla },
    { obsSentence: `${cla} despite the pump staying idle for the whole round.`, obsSpanText: cla, rel: 'IMPLIES' },
    { obsSentence: `Before sunrise only the beacon glowed and during the evening round ${OBS}.`,
      obsSpanText: `during the evening round ${OBS}` },
  ];
  for (const spec of variants) {
    const name = spec.obsSentence;
    await t.test(name, async (st) => {
      const { X, bc } = build({ ...spec, wholeObsSpan: true });
      const prog = toProgram(bc, X.length);
      // R9: the deterministic kernel itself refuses the smuggled clause
      assert.equal(solve(prog, X).code, 'UNACCOUNTED_MATERIAL_SENTENCE', `kernel refusal: ${name}`);
      let prompt = '';
      const fetch = st.mock.method(globalThis, 'fetch', async (url, init) => {
        prompt = JSON.parse(init.body).messages[0].content;
        const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
        return Response.json({
          message: { content: JSON.stringify({ entailed: true, checks: ids.map((id) => ({ id, entailed: true })) }) },
        });
      });
      const witness = await checkSemanticEntailment(prog, X);
      assert.match(prompt, /^R1 \[RESIDUAL CLAUSE\]/m, `batch names the residual clause: ${name}`);
      assert.equal(fetch.mock.callCount(), 1);
      assert.equal(witness.entailed, false, name);
      assert.equal(witness.code, 'RESIDUAL_CLAUSE_NOT_ENTAILED', name);
      assert.ok(witness.checks.some((c) => c.id === 'R1' && c.entailed === false), name);
      assert.equal(verifyEntailmentWitness(witness, prog, X).code, 'SEMANTIC_NOT_ENTAILED', name);
      assert.equal(
        (await solveWithWitness(prog, X, { entailmentWitness: witness })).code,
        'UNACCOUNTED_MATERIAL_SENTENCE', name
      );
      const gate = await verifyInvariance({
        X, extract: makeSemanticExtractor(bc), entailmentWitness: witness,
      });
      assert.equal(gate.verdict, 'REFUSED', name);
      assert.equal(gate.code, 'UNACCOUNTED_MATERIAL_SENTENCE', name);
    });
  }
});

test('R5 witness emits no residual check for a clean compound-free passage', async (t) => {
  const { X, bc } = build({ obsSentence: `${cla}.`, obsSpanText: cla });
  const prog = toProgram(bc, X.length);
  let prompt = '';
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    prompt = JSON.parse(init.body).messages[0].content;
    const ids = [...new Set([...prompt.matchAll(/^([A-Z]+\d+) \[/gm)].map((m) => m[1]))];
    return Response.json({
      message: { content: JSON.stringify({ entailed: true, checks: ids.map((id) => ({ id, entailed: true })) }) },
    });
  });
  const witness = await checkSemanticEntailment(prog, X);
  assert.doesNotMatch(prompt, /\[RESIDUAL CLAUSE\]/);
  assert.equal(witness.entailed, true);
  assert.deepEqual(verifyEntailmentWitness(witness, prog, X), { ok: true });
  assert.equal((await solveWithWitness(prog, X, { entailmentWitness: witness })).status, 'ANSWER');
});
