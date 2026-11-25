// prism-v4.1 :: credal decision layer + MCP server — offline tests.
// No live Ollama: mock extractor + stubbed witness fetch, injected deps.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import * as R from '../src/kernel/rational.js';
import {
  compileV4, decideFromBelief, decideFromProse, parseExactPayoff,
} from '../src/v4/index.js';
import { handleMcpMessage, TOOLS } from '../src/mcp/server.js';

const sha = (s) => createHash('sha256').update(s, 'utf8').digest('hex');
const SHIP = [
  { id: 'halt', label: 'Halt batch', payoffs: { defective: '-5000', ok: '-5000' } },
  { id: 'ship', label: 'Ship batch', payoffs: { defective: '-100000', ok: '10000' } },
];
// A: 8p, B: 3(1-p) -> flip at 8p = 3-3p -> p* = 3/11
const AB = [
  { id: 'A', payoffs: { T: '8', N: '0' } },
  { id: 'B', payoffs: { T: '0', N: '3' } },
];

// ---------- parseExactPayoff ----------
test('parseExactPayoff: exact integers, fractions, decimals, rationals', () => {
  assert.equal(R.toString(parseExactPayoff('-5000')), '-5000/1');
  assert.equal(R.toString(parseExactPayoff('-5000/1')), '-5000/1');
  assert.equal(R.toString(parseExactPayoff('125.50')), '251/2');
  assert.equal(R.toString(parseExactPayoff('0.1')), '1/10');
  assert.equal(R.toString(parseExactPayoff(42)), '42/1');
  assert.equal(R.toString(parseExactPayoff(R.make(3n, 4n))), '3/4');
  assert.equal(R.toString(parseExactPayoff(' 2 / 4 ')), '1/2');
});
test('parseExactPayoff: NaN/Infinity/garbage fail closed', () => {
  for (const bad of ['NaN', 'Infinity', '', 'abc', '1/0', '1e3x', Number.NaN, Infinity, null]) {
    assert.throws(() => parseExactPayoff(bad), /INVALID_PAYOFF/, String(bad));
  }
});

// ---------- decideFromBelief ----------
test('decideFromBelief: point posterior proves optimal action with exact EU + margin', () => {
  const r = decideFromBelief({
    belief: { status: 'POINT', lo: '1/10', hi: '1/10' },
    actions: SHIP, targetHyp: 'defective', altHyp: 'ok',
  });
  assert.equal(r.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(r.optimalAction, 'ship');
  assert.equal(r.pointOptimal, true);
  // ship EU = -100000/10 + 10000*9/10 = -1000 ; halt = -5000 ; margin 4000
  assert.deepEqual(r.euInterval, ['-1000/1', '-1000/1']);
  assert.equal(r.dominanceMargins.halt, '4000/1');
  assert.equal(r.actionsSummary.find((a) => a.id === 'ship').euAtLo, '-1000/1');
  assert.match(r.receiptSha256, /^[0-9a-f]{64}$/);
});
test('decideFromBelief: credal dominance on [1/5, 2/5] without a point posterior', () => {
  const r = decideFromBelief({
    belief: { status: 'INTERVAL', lo: '1/5', hi: '2/5' },
    actions: SHIP, targetHyp: 'defective', altHyp: 'ok',
  });
  assert.equal(r.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(r.optimalAction, 'halt');
  assert.equal(r.pointOptimal, false);
  // halt-ship: at 1/5 -> 7000, at 2/5 -> 29000 ; worst case 7000
  assert.equal(r.dominanceMargins.ship, '7000/1');
  assert.deepEqual(r.undominatedActions, ['halt']);
  assert.deepEqual(r.flipThresholds, []);
});
test('decideFromBelief: flip at 3/11 inside [1/5, 1/2] is indeterminate', () => {
  const r = decideFromBelief({
    belief: { status: 'INTERVAL', lo: '1/5', hi: '1/2' },
    actions: AB, targetHyp: 'T', altHyp: 'N',
  });
  assert.equal(r.verdict, 'DECISION_INDETERMINATE');
  assert.equal(r.optimalAction, null);
  assert.equal(r.flipThresholds.length, 1);
  assert.equal(r.flipThresholds[0].flipThreshold, '3/11');
  assert.equal(r.flipThresholds[0].pStar, '3/11');
  assert.ok(Math.abs(r.flipThresholds[0].decimalPreview - 3 / 11) < 1e-12);
  assert.deepEqual([...r.undominatedActions].sort(), ['A', 'B']);
  assert.equal(r.missingInformation, 'UNDERDETERMINED_INTERVAL');
  // EUs at endpoints are exact: A: 8/5, 4/1 ; B: 12/5, 3/2
  const sA = r.actionsSummary.find((a) => a.id === 'A');
  assert.deepEqual([sA.euAtLo, sA.euAtHi], ['8/5', '4/1']);
});
test('decideFromBelief: exact tie at p* and tie at an endpoint never break ties', () => {
  const atTie = decideFromBelief({
    belief: { status: 'POINT', lo: '3/11', hi: '3/11' },
    actions: AB, targetHyp: 'T', altHyp: 'N',
  });
  assert.equal(atTie.verdict, 'DECISION_INDETERMINATE');
  assert.equal(atTie.optimalAction, null);
  assert.deepEqual([...atTie.undominatedActions].sort(), ['A', 'B']);
  const endpointTie = decideFromBelief({
    belief: { status: 'INTERVAL', lo: '3/11', hi: '1/2' },
    actions: AB, targetHyp: 'T', altHyp: 'N',
  });
  assert.equal(endpointTie.verdict, 'DECISION_INDETERMINATE');
  assert.deepEqual([...endpointTie.undominatedActions].sort(), ['A', 'B']);
});
test('decideFromBelief: poisoned REFUSED beliefs always refuse, even with a universal dominator', () => {
  const everywhere = [
    { id: 'good', payoffs: { T: '10', N: '10' } },
    { id: 'bad', payoffs: { T: '0', N: '0' } },
  ];
  for (const code of ['SEMANTIC_NOT_ENTAILED', 'UNSTABLE_BINDING', 'SCOPED_OR_RETRACTED_PREMISE', 'CONFLICTING_FIGURES', 'MISSING_BASE_RATE']) {
    const r = decideFromBelief({
      belief: { status: 'REFUSED', code, lo: '0', hi: '1' },
      actions: everywhere, targetHyp: 'T', altHyp: 'N',
    });
    assert.equal(r.verdict, 'REFUSED', code);
    assert.equal(r.code, code);
    assert.equal(r.optimalAction, null);
  }
});
test('decideFromBelief: verified UNDERDETERMINED_INTERVAL interval may still prove dominance', () => {
  const r = decideFromBelief({
    belief: { status: 'REFUSED', code: 'UNDERDETERMINED_INTERVAL', lo: '1/5', hi: '2/5' },
    actions: SHIP, targetHyp: 'defective', altHyp: 'ok',
  });
  assert.equal(r.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(r.optimalAction, 'halt');
});
test('decideFromBelief: 3 actions — dominator wins, dominated excluded, no spurious flips', () => {
  const three = [...AB, { id: 'C', payoffs: { T: '-10', N: '-10' } }];
  const ind = decideFromBelief({
    belief: { status: 'INTERVAL', lo: '1/5', hi: '1/2' },
    actions: three, targetHyp: 'T', altHyp: 'N',
  });
  assert.equal(ind.verdict, 'DECISION_INDETERMINATE');
  assert.deepEqual([...ind.undominatedActions].sort(), ['A', 'B']);
  assert.deepEqual(ind.dominatedActions, ['C']);
  assert.equal(ind.flipThresholds.length, 1);
  assert.equal(ind.flipThresholds[0].flipThreshold, '3/11');
  // advisory minimax is never released as optimal
  assert.equal(ind.optimalAction, null);
  assert.ok(['A', 'B'].includes(ind.minimaxAction));
  const pt = decideFromBelief({
    belief: { status: 'POINT', lo: '1/2', hi: '1/2' },
    actions: three, targetHyp: 'T', altHyp: 'N',
  });
  assert.equal(pt.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(pt.optimalAction, 'A');
  assert.equal(pt.dominanceMargins.B, '5/2');
  assert.equal(pt.dominanceMargins.C, '14/1');
});
test('decideFromBelief: decimal payoffs stay exact (1/3 of 0.1 + 2/3 of 0.2 = 1/6)', () => {
  const r = decideFromBelief({
    belief: { status: 'POINT', lo: '1/3', hi: '1/3' },
    actions: [
      { id: 'x', payoffs: { T: '0.1', N: '0.2' } },
      { id: 'y', payoffs: { T: '0', N: '0' } },
    ],
    targetHyp: 'T', altHyp: 'N',
  });
  assert.equal(r.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(r.actionsSummary.find((a) => a.id === 'x').euAtLo, '1/6');
});
test('decideFromBelief: invalid inputs fail closed with REFUSED', () => {
  const good = { belief: { status: 'POINT', lo: '1/2', hi: '1/2' }, actions: AB, targetHyp: 'T', altHyp: 'N' };
  const cases = [
    { ...good, belief: { status: 'INTERVAL', lo: '3/4', hi: '1/4' } },
    { ...good, belief: { status: 'INTERVAL', lo: '-1/4', hi: '1/2' } },
    { ...good, belief: { status: 'INTERVAL', lo: '0', hi: '3/2' } },
    { ...good, belief: { status: 'POINT', lo: 'NaN', hi: 'NaN' } },
    { ...good, actions: [AB[0]] },
    { ...good, actions: [AB[0], { id: 'A', payoffs: { T: '1', N: '1' } }] },
    { ...good, actions: [AB[0], { id: 'Z', payoffs: { T: '1' } }] },
    { ...good, actions: [AB[0], { id: 'Z', payoffs: { T: 'oops', N: '1' } }] },
    { ...good, actions: [AB[0], { id: '', payoffs: { T: '1', N: '1' } }] },
    { ...good, targetHyp: 'T', altHyp: 'T' },
  ];
  for (const c of cases) {
    const r = decideFromBelief(c);
    assert.equal(r.verdict, 'REFUSED', JSON.stringify(c));
    assert.equal(r.optimalAction, null);
    assert.match(r.receiptSha256, /^[0-9a-f]{64}$/);
  }
});
test('decideFromBelief: receipts are deterministic and input-sensitive', () => {
  const inp = { belief: { status: 'INTERVAL', lo: '1/5', hi: '2/5' }, actions: SHIP, targetHyp: 'defective', altHyp: 'ok' };
  const a = decideFromBelief(inp); const b = decideFromBelief(JSON.parse(JSON.stringify(inp)));
  assert.equal(a.receiptSha256, b.receiptSha256);
  const c = decideFromBelief({ ...inp, belief: { status: 'INTERVAL', lo: '1/5', hi: '1/2' } });
  assert.notEqual(a.receiptSha256, c.receiptSha256);
});

// ---------- decideFromProse (mock extractor + stubbed witness fetch) ----------
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
  posteriors: [], obs: [{ ev: 'e0', pol: 'POS', span: 'observed with an amber notch', ctx: 'base' }],
  condIndep: [], detLinks: [], nonAsserted: [], partition: ['h0', 'h1'],
  query: { target: 'h0', given: [{ ev: 'e0', pol: 'POS' }], ctx: 'base' },
};
// P(repaired | notch) = (3/10*23/50)/(3/10*23/50 + 7/10*1/10) = 69/104
const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);
function cloneProg(prog) {
  return JSON.parse(
    JSON.stringify(prog, (k, v) => (v && typeof v === 'object' && 'n' in v && 'd' in v ? { n: String(v.n), d: String(v.d) } : v)),
    (k, v) => (v && typeof v === 'object' && 'n' in v && 'd' in v && typeof v.n === 'string' ? { n: BigInt(v.n), d: BigInt(v.d) } : v));
}
function semanticExtractor(json, X0) {
  const c = compileV4(JSON.parse(JSON.stringify(json)), X0);
  assert.deepEqual(c.errors, [], 'template must compile');
  const template = c.program;
  return async (X, info = {}) => {
    const prog = cloneProg(template);
    if (info.slot === 'swap' && info.remapSpan) {
      for (const d of prog.decls) if (d.span) d.span = { ...info.remapSpan(d.span) };
      prog.decls.filter((d) => NUMERIC_OPS.has(d.op)).forEach((d, i) => {
        const rs = info.remappedSites?.[info.declSiteIdx?.[i]];
        if (rs) d.value = rs.value;
      });
    }
    prog.hypNames = { ...c.hypNames }; prog.evNames = { ...c.evNames };
    return prog;
  };
}
function mockWitnessFetch(t, { failIds = new Set() } = {}) {
  return t.mock.method(globalThis, 'fetch', async (url, init) => {
    const body = JSON.parse(init.body);
    const ids = /CHECK_IDS: (.+)/.exec(body.messages[0].content)[1].split(',');
    const content = JSON.stringify({
      checks: ids.map((id) => ({ id, entailed: !failIds.has(id), reason: 'mock' })),
      entailed: failIds.size === 0, code: failIds.size === 0 ? null : 'NOT_ENTAILED', reason: 'mock',
    });
    return Response.json({ message: { content } });
  });
}
const BET_BY_NAME = [
  { id: 'betRepaired', payoffs: { 'repaired unit': '10', 'new replacement': '0' } },
  { id: 'betNew', payoffs: { 'repaired unit': '0', 'new replacement': '10' } },
];
test('decideFromProse: end-to-end PROVED_OPTIMAL_ACTION at verified 69/104 (name-mapped payoffs)', async (t) => {
  mockWitnessFetch(t);
  const r = await decideFromProse({ X: STD_X, actions: BET_BY_NAME, extract: semanticExtractor(STD_J, STD_X) });
  assert.equal(r.verdict, 'PROVED_OPTIMAL_ACTION', `got ${r.verdict} / ${r.code}`);
  assert.equal(r.optimalAction, 'betRepaired');
  assert.equal(r.rational, '69/104');
  assert.equal(r.belief.lo, '69/104');
  // EU diff = 10*(2*69/104 - 1) = 85/26
  assert.equal(r.dominanceMargins.betNew, '85/26');
  assert.equal(r.passageSha256, sha(STD_X));
  assert.equal(r.decisionReceiptSha256, r.receiptSha256);
  assert.equal(r.witnessSummary.entailed, true);
});
test('decideFromProse: h0/h1 payoff keys and declared targetState keys both map', async (t) => {
  mockWitnessFetch(t);
  const byId = await decideFromProse({
    X: STD_X, extract: semanticExtractor(STD_J, STD_X),
    actions: [
      { id: 'a', payoffs: { h0: '10', h1: '0' } },
      { id: 'b', payoffs: { h0: '0', h1: '10' } },
    ],
  });
  assert.equal(byId.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(byId.optimalAction, 'a');
  const byDeclared = await decideFromProse({
    X: STD_X, extract: semanticExtractor(STD_J, STD_X),
    targetState: 'foo', altState: 'bar',
    actions: [
      { id: 'a', payoffs: { foo: '10', bar: '0' } },
      { id: 'b', payoffs: { foo: '0', bar: '10' } },
    ],
  });
  assert.equal(byDeclared.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(byDeclared.optimalAction, 'a');
});
test('decideFromProse: unmappable payoff keys fail closed INVALID_ACTION_PAYOFFS', async (t) => {
  mockWitnessFetch(t);
  const r = await decideFromProse({
    X: STD_X, extract: semanticExtractor(STD_J, STD_X),
    actions: [
      { id: 'a', payoffs: { foo: '10', bar: '0' } },
      { id: 'b', payoffs: { foo: '0', bar: '10' } },
    ],
  });
  assert.equal(r.verdict, 'REFUSED');
  assert.equal(r.code, 'INVALID_ACTION_PAYOFFS');
  assert.equal(r.optimalAction, null);
});
test('decideFromProse: witness trap REFUSED keeps optimalAction null with advisory flips only', async (t) => {
  mockWitnessFetch(t, { failIds: new Set(['G1']) });
  const r = await decideFromProse({
    X: STD_X, actions: BET_BY_NAME,
    targetState: 'repaired unit', altState: 'new replacement',
    extract: semanticExtractor(STD_J, STD_X),
  });
  assert.equal(r.verdict, 'REFUSED');
  assert.equal(r.code, 'SEMANTIC_NOT_ENTAILED');
  assert.equal(r.optimalAction, null);
  assert.ok(r.whatWouldBeNeeded, 'advisory block present');
  assert.match(r.whatWouldBeNeeded.note, /advisory/);
  // Over [0,1] the two bets flip at 1/2 — advisory only, not a release.
  assert.equal(r.whatWouldBeNeeded.flipThresholds[0].flipThreshold, '1/2');
});

// ---------- MCP: handleMcpMessage ----------
const rpc = (id, method, params) => ({ jsonrpc: '2.0', id, method, ...(params !== undefined ? { params } : {}) });
const toolPayload = (resp) => JSON.parse(resp.result.content[0].text);

test('mcp: initialize, ping, notifications/initialized, tools/list shapes', async () => {
  const init = await handleMcpMessage(rpc(1, 'initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 't', version: '0' } }));
  assert.equal(init.jsonrpc, '2.0'); assert.equal(init.id, 1);
  assert.equal(init.result.protocolVersion, '2025-03-26');
  assert.ok(init.result.serverInfo.name);
  assert.deepEqual(await handleMcpMessage({ jsonrpc: '2.0', method: 'notifications/initialized' }), null);
  const ping = await handleMcpMessage(rpc(2, 'ping'));
  assert.deepEqual(ping.result, {});
  const list = await handleMcpMessage(rpc(3, 'tools/list'));
  assert.equal(list.result.tools.length, 3);
  assert.deepEqual(list.result.tools.map((x) => x.name),
    ['prism_verify_probability', 'prism_decide_from_belief', 'prism_decide_from_prose']);
  assert.deepEqual(TOOLS.map((x) => x.name), list.result.tools.map((x) => x.name));
  for (const tool of list.result.tools) {
    assert.equal(tool.inputSchema.type, 'object', tool.name);
    assert.ok(tool.inputSchema.properties && Array.isArray(tool.inputSchema.required), tool.name);
    assert.ok(tool.description.length > 10, tool.name);
  }
});
test('mcp: prism_decide_from_belief proves dominance and reports the 3/11 flip', async () => {
  const dom = await handleMcpMessage(rpc(1, 'tools/call', {
    name: 'prism_decide_from_belief',
    arguments: { lo: '1/5', hi: '2/5', targetState: 'defective', altState: 'ok', actions: SHIP },
  }));
  assert.equal(dom.result.isError, false);
  const p = toolPayload(dom);
  assert.equal(p.verdict, 'PROVED_OPTIMAL_ACTION');
  assert.equal(p.optimalAction, 'halt');
  assert.deepEqual(dom.result.structuredContent, p);
  const ind = await handleMcpMessage(rpc(2, 'tools/call', {
    name: 'prism_decide_from_belief',
    arguments: { lo: '1/5', hi: '1/2', targetState: 'T', altState: 'N', actions: AB },
  }));
  const q = toolPayload(ind);
  assert.equal(q.verdict, 'DECISION_INDETERMINATE');
  assert.equal(q.flipThresholds[0].flipThreshold, '3/11');
  const bad = await handleMcpMessage(rpc(3, 'tools/call', {
    name: 'prism_decide_from_belief',
    arguments: { lo: 'oops', hi: '1/2', targetState: 'T', altState: 'N', actions: AB },
  }));
  assert.equal(bad.result.isError, true);
  assert.equal(toolPayload(bad).verdict, 'REFUSED');
});
function fakeVerifyPass() {
  const c = compileV4(JSON.parse(JSON.stringify(STD_J)), STD_X);
  const value = R.parse('69/104');
  return async () => ({
    verdict: 'PASS', code: null, value, rational: '69/104',
    decision: { status: 'ANSWER', value, target: 'h0' },
    interval: { min: value, max: value }, swaps: [{ assign: [1, 0], ok: true, detail: null }],
    witness: {
      entailed: true, code: null, reason: 'mock', witnessModel: 'mock-model', witnessHost: 'http://127.0.0.1:11434',
      passageSha256: sha(STD_X), programCanonicalSha256: 'ab'.repeat(32), checks: [{ id: 'PART', entailed: true }],
    },
    compiled: c, elapsedMs: 1,
  });
}
test('mcp: prism_verify_probability + prism_decide_from_prose with injected mock verifyV4', async () => {
  const deps = { verifyV4: fakeVerifyPass() };
  const v = await handleMcpMessage(rpc(1, 'tools/call', {
    name: 'prism_verify_probability', arguments: { passage: STD_X, maxSwaps: 2 },
  }), deps);
  const vp = toolPayload(v);
  assert.equal(vp.verdict, 'PASS');
  assert.equal(vp.rational, '69/104');
  assert.ok(Math.abs(vp.decimalPreview - 69 / 104) < 1e-12);
  assert.equal(vp.passageSha256, sha(STD_X));
  assert.equal(vp.programCanonicalSha256, 'ab'.repeat(32));
  assert.equal(vp.witnessSummary.entailed, true);
  const d = await handleMcpMessage(rpc(2, 'tools/call', {
    name: 'prism_decide_from_prose',
    arguments: { passage: STD_X, targetState: 'repaired unit', altState: 'new replacement', actions: BET_BY_NAME },
  }), deps);
  const dp = toolPayload(d);
  assert.equal(dp.verdict, 'PROVED_OPTIMAL_ACTION', JSON.stringify(dp).slice(0, 300));
  assert.equal(dp.optimalAction, 'betRepaired');
  assert.match(dp.decisionReceiptSha256, /^[0-9a-f]{64}$/);
  // injected extract/witnessCheck path through the real verifyV4 also works
});
test('mcp: prism_verify_probability via injected extract (real verifyV4, stubbed witness fetch)', async (t) => {
  mockWitnessFetch(t);
  const resp = await handleMcpMessage(rpc(1, 'tools/call', {
    name: 'prism_verify_probability', arguments: { passage: STD_X },
  }), { extract: semanticExtractor(STD_J, STD_X) });
  const p = toolPayload(resp);
  assert.equal(p.verdict, 'PASS');
  assert.equal(p.rational, '69/104');
});
test('mcp: unknown tool, unknown method, malformed requests fail closed', async () => {
  const tool = await handleMcpMessage(rpc(1, 'tools/call', { name: 'prism_nope', arguments: {} }));
  assert.equal(tool.error.code, -32602);
  const meth = await handleMcpMessage(rpc(2, 'resources/list'));
  assert.equal(meth.error.code, -32601);
  const malformed = await handleMcpMessage({ jsonrpc: '1.0', id: 3 });
  assert.equal(malformed.error.code, -32600);
  const notObj = await handleMcpMessage('hello');
  assert.equal(notObj.error.code, -32600);
  const missing = await handleMcpMessage(rpc(4, 'tools/call', { name: 'prism_decide_from_belief', arguments: { lo: '0' } }));
  assert.equal(missing.result.isError, true);
  assert.equal(toolPayload(missing).code, 'INVALID_ARGUMENTS');
  const badHost = await handleMcpMessage(rpc(5, 'tools/call', {
    name: 'prism_verify_probability',
    arguments: { passage: STD_X, ollamaHost: 'http://example.com:11434' },
  }), { verifyV4: fakeVerifyPass() });
  assert.equal(badHost.result.isError, true);
  assert.equal(toolPayload(badHost).code, 'ENTAILMENT_WITNESS_UNAVAILABLE');
});

// ---------- MCP: real child-process stdio ----------
const SERVER = fileURLToPath(new URL('../src/mcp/server.js', import.meta.url));
function spawnServer() {
  const child = spawn('node', [SERVER], { stdio: ['pipe', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', (d) => { stderr += d; });
  return { child, getStderr: () => stderr };
}
function collectLines(child, count, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    let buf = ''; const out = [];
    const timer = setTimeout(() => reject(new Error(`timed out waiting for ${count} lines, got ${out.length}: ${buf}`)), timeoutMs);
    child.stdout.on('data', (d) => {
      buf += d.toString();
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (line) out.push(line);
        if (out.length === count) { clearTimeout(timer); resolve(out); return; }
      }
    });
    child.once('exit', (code) => { if (out.length < count) { clearTimeout(timer); reject(new Error(`server exited ${code} after ${out.length} lines`)); } });
  });
}
test('mcp stdio child: initialize + tools/list + prism_decide_from_belief over NDJSON', async () => {
  const { child } = spawnServer();
  try {
    const linesP = collectLines(child, 3);
    child.stdin.write(`${JSON.stringify(rpc(1, 'initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 't', version: '0' } }))}\n`);
    child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`);
    child.stdin.write(`${JSON.stringify(rpc(2, 'tools/list'))}\n`);
    child.stdin.write(`${JSON.stringify(rpc(3, 'tools/call', {
      name: 'prism_decide_from_belief',
      arguments: { lo: '1/5', hi: '1/2', targetState: 'T', altState: 'N', actions: AB },
    }))}\n`);
    const lines = await linesP;
    const [init, list, call] = lines.map((l) => JSON.parse(l));
    assert.equal(init.id, 1);
    assert.equal(init.result.serverInfo.name, 'prism-v4-mcp');
    assert.equal(list.result.tools.length, 3);
    const payload = JSON.parse(call.result.content[0].text);
    assert.equal(payload.verdict, 'DECISION_INDETERMINATE');
    assert.equal(payload.flipThresholds[0].flipThreshold, '3/11');
  } finally { child.kill(); }
});
test('mcp stdio child: Content-Length framing + malformed JSON parse error', async () => {
  const { child } = spawnServer();
  try {
    const got = new Promise((resolve, reject) => {
      let buf = '';
      const timer = setTimeout(() => reject(new Error(`CL timeout: ${buf}`)), 8000);
      child.stdout.on('data', (d) => {
        buf += d.toString();
        const m = /content-length:\s*(\d+)\r\n\r\n/i.exec(buf);
        if (m && buf.length >= m.index + m[0].length + Number(m[1])) {
          clearTimeout(timer);
          resolve(buf.slice(m.index + m[0].length, m.index + m[0].length + Number(m[1])));
        }
      });
    });
    const body = JSON.stringify(rpc(9, 'tools/list'));
    child.stdin.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
    const parsed = JSON.parse(await got);
    assert.equal(parsed.id, 9);
    assert.equal(parsed.result.tools.length, 3);
  } finally { child.kill(); }
  const s2 = spawnServer();
  try {
    const linesP = collectLines(s2.child, 1);
    s2.child.stdin.write('this is not json\n');
    const [line] = await linesP;
    const parsed = JSON.parse(line);
    assert.equal(parsed.error.code, -32700);
    assert.equal(parsed.id, null);
  } finally { s2.child.kill(); }
});


