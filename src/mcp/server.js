#!/usr/bin/env node
// prism-v4.1 :: Zero-dependency local stdio MCP server (JSON-RPC 2.0).
// Supports newline-delimited JSON and Content-Length framing; stdout
// carries ONLY JSON-RPC messages (diagnostics go to stderr).
import { createHash } from 'node:crypto';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as R from '../kernel/rational.js';
import { verifyV4 } from '../v4/gate.js';
import { decideFromBelief, decideFromProse, parseExactPayoff } from '../v4/decision.js';

export const MCP_PROTOCOL_VERSION = '2024-11-05';
const SUPPORTED = new Set(['2024-11-05', '2025-03-26']);
const sha256hex = (s) => createHash('sha256').update(String(s), 'utf8').digest('hex');

const actionSchema = {
  type: 'array', minItems: 2,
  items: {
    type: 'object',
    properties: {
      id: { type: 'string' }, label: { type: 'string' },
      payoffs: { type: 'object', additionalProperties: { type: ['string', 'number'] } },
    },
    required: ['id', 'payoffs'],
  },
};
export const TOOLS = [
  {
    name: 'prism_verify_probability',
    description: 'Verify a Bayesian probability in prose with PRISM v4 (exact BigInt credal solve + counterfactual swaps + sealed cross-model witness). Returns PASS with an exact rational, or REFUSED with a typed code.',
    inputSchema: {
      type: 'object',
      properties: {
        passage: { type: 'string', description: 'Passage text to verify' },
        maxSwaps: { type: 'integer', minimum: 0, maximum: 4, default: 2 },
        model: { type: 'string' }, witnessModel: { type: 'string' },
        ollamaHost: { type: 'string' }, allowCustomWitnessEndpoint: { type: 'boolean' },
      },
      required: ['passage'], additionalProperties: false,
    },
  },
  {
    name: 'prism_decide_from_belief',
    description: 'Exact BigInt rational credal decision from a belief interval P(targetState) in [lo, hi]. Offline and deterministic; returns PROVED_OPTIMAL_ACTION only under strict dominance on the whole interval, else DECISION_INDETERMINATE with exact flip thresholds.',
    inputSchema: {
      type: 'object',
      properties: {
        lo: { type: 'string', description: 'Exact rational "p/q", decimal, or integer' },
        hi: { type: 'string' },
        targetState: { type: 'string' }, altState: { type: 'string' },
        actions: actionSchema,
      },
      required: ['lo', 'hi', 'targetState', 'altState', 'actions'], additionalProperties: false,
    },
  },
  {
    name: 'prism_decide_from_prose',
    description: 'End-to-end: verify the probability in a passage with PRISM v4, then decide between actions with exact rational expected utility. A REFUSED verification can never release an optimal action.',
    inputSchema: {
      type: 'object',
      properties: {
        passage: { type: 'string' },
        targetState: { type: 'string' }, altState: { type: 'string' },
        actions: actionSchema,
        maxSwaps: { type: 'integer', minimum: 0, maximum: 4, default: 2 },
        model: { type: 'string' }, witnessModel: { type: 'string' },
        ollamaHost: { type: 'string' }, allowCustomWitnessEndpoint: { type: 'boolean' },
      },
      required: ['passage', 'targetState', 'altState', 'actions'], additionalProperties: false,
    },
  },
];

const okResult = (id, result) => ({ jsonrpc: '2.0', id, result });
const errResult = (id, code, message, data) => ({ jsonrpc: '2.0', id: id ?? null, error: { code, message, ...(data !== undefined ? { data } : {}) } });
function toolResult(id, payload, isError = false) {
  const text = JSON.stringify(payload, (k, v) => {
    if (v && typeof v === 'object' && typeof v.n === 'bigint' && typeof v.d === 'bigint') return `${v.n}/${v.d}`;
    return typeof v === 'bigint' ? String(v) : v;
  });
  let structured = payload;
  try { structured = JSON.parse(text); } catch { /* keep raw payload */ }
  return okResult(id, {
    content: [{ type: 'text', text }],
    structuredContent: structured, isError,
  });
}
function toolError(id, code, reason) {
  return toolResult(id, { verdict: 'REFUSED', code, reason, optimalAction: null }, true);
}
function isLoopbackHost(u) {
  if (!u) return true;
  try {
    const h = new URL(u).hostname;
    return h === '127.0.0.1' || h === 'localhost' || h === '::1' || h === '[::1]';
  } catch { return false; }
}
function hostGuard(args, deps) {
  const host = args?.ollamaHost;
  if (!host) return null;
  if (args.allowCustomWitnessEndpoint === true || deps?.allowCustomWitnessEndpoint === true) return null;
  if (isLoopbackHost(host)) return null;
  return `non-local ollamaHost "${host}" requires allowCustomWitnessEndpoint: true`;
}
function validMaxSwaps(v) {
  return v === undefined || (Number.isInteger(v) && v >= 0 && v <= 4);
}
function decimalOfRationalString(s) {
  if (typeof s !== 'string') return null;
  try { const r = parseExactPayoff(s); const n = R.toNumber(r); return Number.isFinite(n) ? n : null; } catch { return null; }
}
function witnessSummaryOf(w) {
  if (!w || typeof w !== 'object') return null;
  return {
    entailed: w.entailed ?? null, code: w.code ?? null, reason: w.reason ?? null,
    witnessModel: w.witnessModel ?? null, witnessHost: w.witnessHost ?? null,
    passageSha256: w.passageSha256 ?? null, programCanonicalSha256: w.programCanonicalSha256 ?? null,
    checks: Array.isArray(w.checks) ? w.checks.map((c) => ({ id: c.id, entailed: c.entailed })) : [],
  };
}
function intervalStrings(res) {
  const iv = res?.interval;
  if (!iv) return null;
  const fmt = (v) => (v && typeof v === 'object' && 'n' in v ? R.toString(R.make(BigInt(v.n), BigInt(v.d))) : v ?? null);
  return { min: fmt(iv.min), max: fmt(iv.max) };
}

async function callVerify(args, deps) {
  const passage = args.passage ?? args.X;
  if (typeof passage !== 'string' || !passage.trim()) return { error: ['INVALID_ARGUMENTS', 'passage (non-empty string) is required'] };
  if (!validMaxSwaps(args.maxSwaps)) return { error: ['INVALID_ARGUMENTS', 'maxSwaps must be an integer in 0..4'] };
  const hg = hostGuard(args, deps); if (hg) return { error: ['ENTAILMENT_WITNESS_UNAVAILABLE', hg] };
  const verifyFn = deps.verifyV4 ?? deps.verify ?? verifyV4;
  const opts = { X: passage, passage };
  for (const k of ['maxSwaps', 'model', 'witnessModel', 'ollamaHost', 'allowCustomWitnessEndpoint']) {
    if (args[k] !== undefined) opts[k] = args[k];
  }
  if (!deps.verifyV4) {
    if (deps.extract !== undefined) opts.extract = deps.extract;
    if (deps.witnessCheck !== undefined) opts.witnessCheck = deps.witnessCheck;
  }
  let res;
  try { res = await verifyFn(opts); } catch (e) { return { error: ['EXTRACTION_FAILED', `verifyV4 threw: ${e.message}`] }; }
  const wSum = witnessSummaryOf(res?.witness);
  const rational = res?.rational ?? (res?.value && typeof res.value === 'object' && 'n' in res.value
    ? R.toString(R.make(BigInt(res.value.n), BigInt(res.value.d))) : res?.decision?.value
    ? R.toString(R.make(BigInt(res.decision.value.n), BigInt(res.decision.value.d))) : null);
  return { payload: {
    verdict: res?.verdict ?? null, code: res?.code ?? null, rational,
    decimalPreview: rational ? decimalOfRationalString(rational) : null,
    target: res?.decision?.target ?? res?.compiled?.program?.query?.target ?? null,
    interval: intervalStrings(res), swaps: res?.swaps ?? [],
    witnessSummary: wSum, witness: wSum,
    passageSha256: wSum?.passageSha256 ?? sha256hex(passage),
    programCanonicalSha256: wSum?.programCanonicalSha256 ?? null,
  } };
}
function callDecideFromBelief(args, deps) {
  const targetState = args.targetState ?? args.targetHyp;
  const altState = args.altState ?? args.altHyp;
  const lo = args.lo ?? args.belief?.lo;
  const hi = args.hi ?? args.belief?.hi;
  if (lo === undefined || hi === undefined || !targetState || !altState || !Array.isArray(args.actions)) {
    return { error: ['INVALID_ARGUMENTS', 'lo, hi, targetState, altState and actions are required'] };
  }
  const fn = deps.decideFromBelief ?? decideFromBelief;
  let payload;
  try {
    let status = args.belief?.status ?? 'INTERVAL';
    try { if (R.eq(parseExactPayoff(lo), parseExactPayoff(hi))) status = args.belief?.status === 'REFUSED' ? 'REFUSED' : 'POINT'; } catch { /* engine will refuse */ }
    payload = fn({
      belief: args.belief ? { ...args.belief, lo, hi } : { status, lo, hi },
      actions: args.actions, targetHyp: targetState, altHyp: altState,
      targetState, altState,
    });
  } catch (e) { return { error: ['INVALID_ACTION_PAYOFFS', e.message] }; }
  const isError = payload?.verdict === 'REFUSED' && /^INVALID_/.test(String(payload.code ?? ''));
  return { payload, isError };
}
async function callDecideFromProse(args, deps) {
  const passage = args.passage ?? args.X;
  if (typeof passage !== 'string' || !passage.trim()) return { error: ['INVALID_ARGUMENTS', 'passage (non-empty string) is required'] };
  if (!args.targetState && !args.targetHyp) return { error: ['INVALID_ARGUMENTS', 'targetState is required'] };
  if (!args.altState && !args.altHyp) return { error: ['INVALID_ARGUMENTS', 'altState is required'] };
  if (!Array.isArray(args.actions)) return { error: ['INVALID_ARGUMENTS', 'actions array is required'] };
  if (!validMaxSwaps(args.maxSwaps)) return { error: ['INVALID_ARGUMENTS', 'maxSwaps must be an integer in 0..4'] };
  const hg = hostGuard(args, deps); if (hg) return { error: ['ENTAILMENT_WITNESS_UNAVAILABLE', hg] };
  const fn = deps.decideFromProse ?? decideFromProse;
  const opts = {
    X: passage, passage, actions: args.actions,
    targetState: args.targetState ?? args.targetHyp, altState: args.altState ?? args.altHyp,
    targetHyp: args.targetState ?? args.targetHyp, altHyp: args.altState ?? args.altHyp,
  };
  for (const k of ['maxSwaps', 'model', 'witnessModel', 'ollamaHost', 'allowCustomWitnessEndpoint']) {
    if (args[k] !== undefined) opts[k] = args[k];
  }
  if (!deps.decideFromProse) {
    if (deps.verifyV4 ?? deps.verify) opts.verifyV4 = deps.verifyV4 ?? deps.verify;
    if (deps.extract !== undefined) opts.extract = deps.extract;
    if (deps.witnessCheck !== undefined) opts.witnessCheck = deps.witnessCheck;
  }
  let payload;
  try { payload = await fn(opts); } catch (e) { return { error: ['EXTRACTION_FAILED', `decideFromProse threw: ${e.message}`] }; }
  // Keep JSON-RPC payload JSON-safe: drop non-serializable raw internals.
  const clean = { ...payload };
  delete clean.compiled;
  if (clean.witness && typeof clean.witness === 'object') clean.witness = witnessSummaryOf(clean.witness) ?? clean.witness;
  const isError = clean?.verdict === 'REFUSED' && /^INVALID_/.test(String(clean.code ?? ''));
  return { payload: clean, isError };
}

/**
 * handleMcpMessage(msg, deps?) -> JSON-RPC response object, or null for
 * notifications. deps may inject verifyV4 / decideFromBelief /
 * decideFromProse, or extract / witnessCheck for the real pipeline.
 */
export async function handleMcpMessage(msg, deps = {}) {
  if (!msg || typeof msg !== 'object' || Array.isArray(msg)) {
    return errResult(null, -32600, 'Invalid Request: expected a JSON-RPC object');
  }
  const id = 'id' in msg ? msg.id : undefined;
  const isNotification = id === undefined;
  if (msg.jsonrpc !== '2.0' || typeof msg.method !== 'string') {
    return isNotification ? null : errResult(id ?? null, -32600, 'Invalid Request: jsonrpc "2.0" and method are required');
  }
  switch (msg.method) {
    case 'initialize': {
      const req = msg.params?.protocolVersion;
      return okResult(id, {
        protocolVersion: SUPPORTED.has(req) ? req : MCP_PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: { name: 'prism-v4-mcp', version: '4.1.0' },
      });
    }
    case 'notifications/initialized':
    case 'notifications/cancelled':
      return null;
    case 'ping':
      return isNotification ? null : okResult(id, {});
    case 'tools/list':
      return okResult(id, { tools: TOOLS });
    case 'tools/call': {
      const name = msg.params?.name;
      const args = msg.params?.arguments ?? {};
      if (typeof name !== 'string' || !name) return errResult(id, -32602, 'Invalid params: tool name is required');
      if (args === null || typeof args !== 'object' || Array.isArray(args)) return errResult(id, -32602, 'Invalid params: arguments must be an object');
      let out;
      if (name === 'prism_verify_probability') out = await callVerify(args, deps);
      else if (name === 'prism_decide_from_belief') out = callDecideFromBelief(args, deps);
      else if (name === 'prism_decide_from_prose') out = await callDecideFromProse(args, deps);
      else return errResult(id, -32602, `Unknown tool: ${name}`);
      if (out.error) return toolError(id, out.error[0], out.error[1]);
      return toolResult(id, out.payload, out.isError === true);
    }
    default:
      return isNotification ? null : errResult(id, -32601, `Method not found: ${msg.method}`);
  }
}

// ---------------- stdio transport ----------------
// Framing is echoed per-message: Content-Length in -> Content-Length
// out, otherwise newline-delimited JSON. Parse errors answer -32700.
export function frameMessage(obj, mode) {
  const body = JSON.stringify(obj);
  return mode === 'content-length'
    ? `Content-Length: ${Buffer.byteLength(body, 'utf8')}\r\n\r\n${body}`
    : `${body}\n`;
}
export function startMcpStdio({ input = process.stdin, output = process.stdout, deps = {} } = {}) {
  let buf = Buffer.alloc(0);
  let queue = Promise.resolve();
  const dispatch = (text, mode) => {
    queue = queue.then(async () => {
      let msg;
      try { msg = JSON.parse(text); } catch {
        output.write(frameMessage(errResult(null, -32700, 'Parse error'), mode));
        return;
      }
      let resp = null;
      try { resp = await handleMcpMessage(msg, deps); }
      catch (e) { resp = errResult(msg?.id ?? null, -32603, `Internal error: ${e.message}`); }
      if (resp !== null && resp !== undefined) output.write(frameMessage(resp, mode));
    }).catch((e) => { console.error(`prism-mcp: ${e.message}`); });
  };
  const pump = () => {
    for (;;) {
      // skip leading blank lines between NDJSON messages
      while (buf.length && (buf[0] === 0x0a || buf[0] === 0x0d)) buf = buf.subarray(1);
      if (!buf.length) return;
      if (/^content-length\s*:/i.test(buf.toString('utf8', 0, Math.min(buf.length, 32)))) {
        const head = buf.toString('utf8', 0, Math.min(buf.length, 8192));
        const m = /\r\n\r\n|\n\n/.exec(head);
        if (!m) { if (buf.length > 8192) { buf = Buffer.alloc(0); } return; }
        const headerText = head.slice(0, m.index);
        const lm = /content-length\s*:\s*(\d+)/i.exec(headerText);
        if (!lm) { buf = buf.subarray(m.index + m[0].length); continue; }
        const len = Number(lm[1]);
        const start = m.index + m[0].length;
        if (buf.length < start + len) return;
        const body = buf.subarray(start, start + len).toString('utf8');
        buf = buf.subarray(start + len);
        dispatch(body, 'content-length');
        continue;
      }
      const nl = buf.indexOf(0x0a);
      if (nl < 0) return;
      const line = buf.subarray(0, nl).toString('utf8').trim();
      buf = buf.subarray(nl + 1);
      if (line) dispatch(line, 'ndjson');
    }
  };
  input.on('data', (chunk) => { buf = Buffer.concat([buf, Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)]); pump(); });
  input.on('end', () => {
    const rest = buf.toString('utf8').trim();
    buf = Buffer.alloc(0);
    if (rest && !/^content-length\s*:/i.test(rest)) dispatch(rest, 'ndjson');
  });
  return { get idle() { return queue; } };
}

const isMain = (() => {
  try { return process.argv[1] && realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url)); }
  catch { return false; }
})();
if (isMain) {
  process.stderr.write('prism-mcp: PRISM v4.1 MCP server on stdio\n');
  startMcpStdio();
}
