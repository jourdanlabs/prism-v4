// prism-v4 :: Substrate 3 — full cross-model semantic entailment witness.
//
// Runs ONLY on a swap-verified exact rational candidate. Every semantic
// claim in the compiled program — partition, every prior / likelihood /
// stated posterior (exact value, owner, direction, population), every
// independence / deterministic license, observation & query alignment,
// and a global retraction / counterexample / missing-premise audit — is
// put to an independent model family (default Qwen3.8-27B, temperature
// 0, seed 0, think:false, format json) as one itemized checklist.
// The receipt is sealed with the same lexical private-brand +
// null-prototype + freeze + SHA-256 passage/program binding design as
// src/invariance/entailment.js, under a STRICT AND: top-level entailed
// AND every emitted check id present and entailed, else refusal.

import { chat } from '../live/ollama.js';
import { toProgram } from '../kernel/parser.js';
import { solve } from '../kernel/solve.js';
import * as R from '../kernel/rational.js';
import {
  passageSha256,
  programCanonicalSha256,
  WITNESS_MODEL,
} from '../invariance/entailment.js';

export const V4_WITNESS_MODEL = WITNESS_MODEL;
const DEFAULT_HOST = 'http://127.0.0.1:11434';
const APPROVED_HOSTS = new Set(['http://127.0.0.1:11434', 'http://localhost:11434']);

// Sealed receipt: capability-gated mint, lexical #brand, frozen.
const MINT_CAP = Symbol('V4SealedWitnessReceipt.mint');
class V4SealedWitnessReceipt {
  #brand = true;
  constructor(fields, cap) {
    if (cap !== MINT_CAP) {
      throw new TypeError('V4SealedWitnessReceipt cannot be constructed directly');
    }
    Object.assign(this, fields);
    Object.setPrototypeOf(this, null);
    Object.freeze(this);
  }
  static isIssued(v) {
    try {
      return Boolean(v && typeof v === 'object' && #brand in v);
    } catch {
      return false;
    }
  }
}
Object.freeze(V4SealedWitnessReceipt.prototype);
Object.freeze(V4SealedWitnessReceipt);

function issueWitness(receipt) {
  for (const check of receipt.checks) Object.freeze(check);
  Object.freeze(receipt.checks);
  return new V4SealedWitnessReceipt(receipt, MINT_CAP);
}

const hasOwn = Object.hasOwn;
function ownOpt(opts, key, fallback) {
  if (!opts || (typeof opts !== 'object' && typeof opts !== 'function')) return fallback;
  if (!hasOwn(opts, key)) return fallback;
  const v = opts[key];
  return v === undefined ? fallback : v;
}

function unapprovedEndpoint(model, host, optIn) {
  return optIn !== true && (model !== V4_WITNESS_MODEL || !APPROVED_HOSTS.has(host));
}
const ENDPOINT_OPT_IN =
  'custom witness host or model requires allowCustomWitnessEndpoint: true';

// Name resolution: caller-supplied maps (object or Map) win; otherwise
// the verbatim HYP/EVENT span text is the name (compileV4 narrows spans
// to the declared name whenever it occurs uniquely inside the quote).
function nameResolver(prog, X, opts) {
  const fromOpt = (optVal, id) => {
    if (optVal instanceof Map) return optVal.get(id) ?? null;
    if (optVal && typeof optVal === 'object' && typeof optVal[id] === 'string') {
      return optVal[id];
    }
    return null;
  };
  const hypOpt = ownOpt(opts, 'hypNames', null) ?? prog.hypNames ?? null;
  const evOpt = ownOpt(opts, 'evNames', null) ?? prog.evNames ?? null;
  const spanName = (op, id) => {
    const d = prog.decls.find((x) => x.op === op && x.id === id);
    return d?.span ? X.slice(d.span.s, d.span.e).trim() : id;
  };
  return {
    hyp: (id) => fromOpt(hypOpt, id) ?? spanName('HYP', id),
    ev: (id) => fromOpt(evOpt, id) ?? spanName('EVENT', id),
  };
}

const ctxDesc = (ctx) =>
  !ctx || ctx === 'ctx0' || ctx === '*'
    ? "the passage's population"
    : `population "${ctx.replace(/^ctx_/, '').replace(/_/g, ' ')}"`;

const valDesc = (v) => `${R.toString(v)} (exactly ${R.toString(v)})`;

// Build the complete itemized checklist for one compiled program.
export function buildV4Checks(prog, X, opts = {}) {
  const names = nameResolver(prog, X, opts);
  const q = prog.query;
  const checks = [];
  const excerpt = (d) =>
    d?.span ? ` (passage excerpt: ${JSON.stringify(X.slice(d.span.s, d.span.e))})` : '';

  // PART — partition exhaustiveness & mutual exclusivity
  const partDecl = prog.decls.find((d) => d.op === 'PART_EXHAUSTIVE');
  const partHyps = partDecl
    ? partDecl.hyps
    : prog.decls.filter((d) => d.op === 'HYP').map((d) => d.id);
  checks.push({
    id: 'PART',
    text:
      `PART [PARTITION]: Does the passage establish that ` +
      `{${partHyps.map((h) => names.hyp(h)).join(', ')}} partition the queried ` +
      `population (${ctxDesc(q.ctx)}): every queried item is exactly one of them, ` +
      `they are mutually exclusive, and there is no third category?`,
  });

  // P1..Pk — every prior (unconditional marginal share)
  let pi = 0;
  for (const d of prog.decls) {
    if (d.op !== 'PRIOR') continue;
    pi++;
    checks.push({
      id: `P${pi}`,
      text:
        `P${pi} [PRIOR]: Does the passage unconditionally assert that the ` +
        `unconditional base rate / marginal share of "${names.hyp(d.hyp)}" in ` +
        `${ctxDesc(d.ctx)} equals ${valDesc(d.value)} — exact value, correct ` +
        `owner, NOT a conditional subgroup rate, NOT from a different ` +
        `population, NOT retracted, disputed, or unverified?${excerpt(d)}`,
    });
  }

  // L1..Lm — every likelihood (forward conditional P(E|H))
  let li = 0;
  for (const d of prog.decls) {
    if (d.op !== 'LIK') continue;
    li++;
    checks.push({
      id: `L${li}`,
      text:
        `L${li} [LIKELIHOOD]: Does the passage unconditionally assert that, ` +
        `given "${names.hyp(d.hyp)}" (forward conditional P(E|H), NOT the ` +
        `inverse P(H|E)), the rate/probability of "${names.ev(d.ev)}" ` +
        `${d.pol === 'NEG' ? 'not occurring' : 'occurring'} in ${ctxDesc(d.ctx)} ` +
        `equals ${valDesc(d.value)} — exact value, correct owner and event, ` +
        `NOT retracted, disputed, or unverified?${excerpt(d)}`,
    });
  }

  // D1..Dn — every stated posterior (direct reverse conditional P(H|E))
  let di = 0;
  for (const d of prog.decls) {
    if (d.op !== 'POSTERIOR_STATED') continue;
    di++;
    checks.push({
      id: `D${di}`,
      text:
        `D${di} [STATED POSTERIOR]: Does the passage unconditionally assert ` +
        `that, given "${names.ev(d.ev)}" ${d.pol === 'NEG' ? 'does not occur' : 'occurs'} ` +
        `(direct posterior direction P(H|E), NOT P(E|H)), the probability/share ` +
        `of "${names.hyp(d.hyp)}" in ${ctxDesc(d.ctx)} equals ${valDesc(d.value)}` +
        ` — exact value, correct owner and event, NOT retracted, disputed, ` +
        `or unverified?${excerpt(d)}`,
    });
  }

  // I1..Ip — every independence / deterministic license
  let ii = 0;
  for (const d of prog.decls) {
    if (d.op === 'COND_INDEP') {
      ii++;
      checks.push({
        id: `I${ii}`,
        text:
          `I${ii} [CONDITIONAL INDEPENDENCE]: Does the passage affirmatively ` +
          `and unconditionally assert that ${d.evs.map((e) => `"${names.ev(e)}"`).join(' and ')} ` +
          `are probabilistically independent within EVERY hypothesis in scope ` +
          `(present tense, NOT restricted to only one hypothesis, NOT negated, ` +
          `NOT separate unlinked tallies, NOT hedged or approximate)?${excerpt(d)}`,
      });
    } else if (d.op === 'DET_LINK') {
      ii++;
      checks.push({
        id: `I${ii}`,
        text:
          `I${ii} [DETERMINISTIC LINK ${d.rel}]: Does the passage affirmatively ` +
          `and unconditionally assert ${d.rel} between "${names.ev(d.a)}" and ` +
          `"${names.ev(d.b)}" over the ENTIRE queried population, with ZERO ` +
          `counterexample occurrence anywhere in the passage, ZERO hedge ` +
          `(usually/mostly/might/rarely), and ZERO subset exception?${excerpt(d)}`,
      });
    }
  }

  // O1 — observation & query alignment
  const givensList = (q.given ?? [])
    .map((g) => `"${names.ev(g.ev)}" ${g.pol === 'NEG' ? 'absent' : 'present'}`)
    .join(', ') || 'no givens';
  const targetName = prog.decls.some((d) => d.op === 'HYP' && d.id === q.target)
    ? names.hyp(q.target)
    : names.ev(q.target);
  checks.push({
    id: 'O1',
    text:
      `O1 [OBSERVATION & QUERY]: Does the passage ask for the probability of ` +
      `"${targetName}"${q.negated ? ' NOT holding' : ''} given ${givensList}, for ` +
      `an item affirmatively belonging to ${ctxDesc(q.ctx)} (NOT a different ` +
      `shift, room, season, wing, or stratum, NOT negated membership), and are ` +
      `ALL relevant observed events for the queried item included in that ` +
      `given list?`,
  });

  // G1 — global soundness & completeness audit
  const usesOnlyDirectPosteriors =
    prog.decls.some((d) => d.op === 'POSTERIOR_STATED') &&
    !prog.decls.some((d) => d.op === 'PRIOR');
  checks.push({
    id: 'G1',
    text:
      `G1 [GLOBAL AUDIT]: Read EVERY sentence of the passage. Is it TRUE that ` +
      `NO sentence anywhere (1) retracts, withdraws, disputes, questions, or ` +
      `marks unverified/preliminary/draft any figure or premise used above; ` +
      `(2) reports a concrete exception or counterexample contradicting any ` +
      `universal rule used above; (3) states that a premise, rate, or base ` +
      `rate needed by the calculation was lost, missing, or unrecorded` +
      (usesOnlyDirectPosteriors
        ? ` (note: this program uses ONLY direct posteriors and ZERO priors, ` +
          `so a remark that overall base rates were not recorded does NOT ` +
          `contradict any premise)`
        : '') +
      `; or (4) states a population/stratum mismatch or extra condition that ` +
      `invalidates applying these premises to the queried item?`,
  });

  return checks;
}

function buildPrompt(checks, X) {
  return `You are an independent semantic entailment verifier for a Bayesian extraction pipeline. Verify, from the PASSAGE TEXT ONLY, that every check below is unconditionally entailed. Be strict: a single wrong value, wrong owner, reversed conditional direction, subset, exception, negation, tense shift, hedge, or foreign-population reference fails that check. Do not give the program the benefit of the doubt.
Read the entire passage when judging each check. A statement anywhere in the passage that a figure was withdrawn, disputed, unverified, preliminary, or unconfirmed defeats the affected check.

PASSAGE:
${X}

CHECKS:
${checks.map((c) => c.text).join('\n')}

CHECK_IDS: ${checks.map((c) => c.id).join(',')}
Reply with ONLY a single-line compact JSON object, no markdown, no whitespace:
{"checks":[{"id":"<id>","entailed":true,"reason":"<short>"}],"entailed":true,"code":null,"reason":"<short>"}
- "checks" MUST contain one entry per check id listed above (${checks.map((c) => c.id).join(', ')}). Keep each "reason" under 12 words.
- A check's "entailed" is true only when the passage affirmatively and unconditionally supports it.
- Top-level "entailed" is true ONLY if every check is true.
- If any check fails, "code" is one of: NOT_ENTAILED, WRONG_VALUE, WRONG_OWNER, REVERSED_CONDITIONAL, SCOPE_EXCEPTION, POPULATION_MISMATCH, RETRACTED_OR_UNVERIFIED, COUNTEREXAMPLE; else null.`;
}

function extractJsonObject(text) {
  const start = text.indexOf('{');
  if (start < 0) return null;
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/**
 * checkV4SemanticEntailment(program, X, opts?) -> sealed witness receipt
 * Never throws: transport or malformed output returns entailed:false with
 * code ENTAILMENT_WITNESS_UNAVAILABLE (fail-closed).
 * opts: model, host, allowCustomWitnessEndpoint, timeoutMs, numPredict,
 *       hypNames, evNames
 */
export async function checkV4SemanticEntailment(program, X, opts = {}) {
  const model = ownOpt(opts, 'model', V4_WITNESS_MODEL);
  const host = ownOpt(opts, 'host', DEFAULT_HOST) ?? DEFAULT_HOST;
  const allowCustomWitnessEndpoint = ownOpt(opts, 'allowCustomWitnessEndpoint', false);
  const timeoutMs = ownOpt(opts, 'timeoutMs', 120000);
  const numPredict = ownOpt(opts, 'numPredict', 1600);
  const prog = toProgram(program, X.length);
  const base = {
    witnessModel: model,
    witnessHost: host,
    passageSha256: passageSha256(X),
    programCanonicalSha256: programCanonicalSha256(prog),
    checks: [],
  };
  const unavailable = (reason) =>
    issueWitness({ ...base, entailed: false, code: 'ENTAILMENT_WITNESS_UNAVAILABLE', reason });
  if (unapprovedEndpoint(model, host, allowCustomWitnessEndpoint)) {
    return unavailable(ENDPOINT_OPT_IN);
  }
  const claims = buildV4Checks(prog, X, opts);
  let content;
  try {
    ({ content } = await chat({
      model,
      host,
      timeoutMs,
      numPredict,
      think: false,
      seed: 0,
      messages: [{ role: 'user', content: buildPrompt(claims, X) }],
    }));
  } catch (e) {
    return unavailable(`entailment witness call failed: ${e.message}`);
  }
  let parsed = null;
  try {
    const obj = extractJsonObject(content);
    parsed = obj ? JSON.parse(obj) : null;
  } catch {
    parsed = null;
  }
  if (!parsed || typeof parsed !== 'object' || typeof parsed.entailed !== 'boolean') {
    return unavailable(`malformed witness output: ${String(content).slice(0, 160)}`);
  }
  const byId = new Map();
  const outChecks = [];
  if (Array.isArray(parsed.checks)) {
    for (const c of parsed.checks) {
      if (c && typeof c.id === 'string') {
        // a duplicated check id cannot be trusted — it fails closed
        byId.set(c.id, !byId.has(c.id) && c.entailed === true);
        outChecks.push({ id: c.id, entailed: c.entailed === true, reason: String(c.reason ?? '') });
      }
    }
  }
  // STRICT AND: top-level entailed AND every emitted check id present
  // with entailed === true. A dropped check id is not a pass.
  const perCheck = claims.every((c) => byId.get(c.id) === true);
  const entailed = parsed.entailed === true && perCheck;
  return issueWitness({
    ...base,
    entailed,
    code: entailed ? null : (typeof parsed.code === 'string' ? parsed.code : 'NOT_ENTAILED'),
    reason: String(parsed.reason ?? ''),
    checks: outChecks,
  });
}

/**
 * verifyV4Witness(witness, prog, X, opts?) -> {ok:true} | {ok:false, code, detail}
 * Only a receipt issued by checkV4SemanticEntailment in this process,
 * bound by SHA-256 to this exact passage and canonical program, passes.
 */
export function verifyV4Witness(witness, prog, X, opts = {}) {
  const model = ownOpt(opts, 'model', V4_WITNESS_MODEL);
  const host = ownOpt(opts, 'host', DEFAULT_HOST) ?? DEFAULT_HOST;
  const allowCustomWitnessEndpoint = ownOpt(opts, 'allowCustomWitnessEndpoint', false);
  if (unapprovedEndpoint(model, host, allowCustomWitnessEndpoint)) {
    return { ok: false, code: 'ENTAILMENT_WITNESS_UNAVAILABLE', detail: ENDPOINT_OPT_IN };
  }
  if (!witness || typeof witness !== 'object') {
    return { ok: false, code: 'ENTAILMENT_WITNESS_UNAVAILABLE', detail: 'no entailment witness supplied' };
  }
  if (!V4SealedWitnessReceipt.isIssued(witness)) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_UNAUTHENTICATED',
      detail: 'witness was not issued by checkV4SemanticEntailment in this process',
    };
  }
  if (witness.witnessModel !== model) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: `witness issued by "${witness.witnessModel}", expected "${model}"`,
    };
  }
  if (witness.witnessHost !== host && !APPROVED_HOSTS.has(witness.witnessHost)) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: 'witness host does not match the configured endpoint',
    };
  }
  if (witness.passageSha256 !== passageSha256(X)) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: 'witness passageSha256 does not bind this passage',
    };
  }
  if (witness.programCanonicalSha256 !== programCanonicalSha256(toProgram(prog, X.length))) {
    return {
      ok: false,
      code: 'ENTAILMENT_WITNESS_MISMATCH',
      detail: 'witness programCanonicalSha256 does not bind this program',
    };
  }
  if (witness.entailed !== true) {
    return {
      ok: false,
      code: witness.code === 'ENTAILMENT_WITNESS_UNAVAILABLE' ? witness.code : 'SEMANTIC_NOT_ENTAILED',
      detail:
        `independent entailment witness did not certify the program ` +
        `(${witness.code ?? 'NOT_ENTAILED'}${witness.reason ? `: ${witness.reason}` : ''})`,
    };
  }
  return { ok: true };
}

/**
 * solveV4(program, X, opts?) -> decision + witness
 * Pure exact solve (solve(prog, null) — no text accounting), then an
 * ANSWER may only release behind a verified v4 entailment witness.
 */
export async function solveV4(program, X, opts = {}) {
  const prog = toProgram(program, X.length);
  const decision = solve(prog, null);
  if (decision.status !== 'ANSWER') return { ...decision, witness: null };
  const config = {
    model: ownOpt(opts, 'model', V4_WITNESS_MODEL),
    host: ownOpt(opts, 'host', DEFAULT_HOST) ?? DEFAULT_HOST,
    allowCustomWitnessEndpoint: ownOpt(opts, 'allowCustomWitnessEndpoint', false) === true,
    timeoutMs: ownOpt(opts, 'timeoutMs', 120000),
    numPredict: ownOpt(opts, 'numPredict', 1600),
    hypNames: ownOpt(opts, 'hypNames', null) ?? prog.hypNames ?? null,
    evNames: ownOpt(opts, 'evNames', null) ?? prog.evNames ?? null,
  };
  const witness =
    ownOpt(opts, 'witness', null) ??
    (await checkV4SemanticEntailment(prog, X, config));
  const v = verifyV4Witness(witness, prog, X, config);
  if (!v.ok) {
    return {
      status: 'REFUSED',
      code: v.code,
      details: { detail: v.detail, witnessCode: witness?.code ?? null, reason: witness?.reason ?? null },
      witness,
    };
  }
  return { ...decision, witness };
}
