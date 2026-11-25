// prism-v4.1 :: Exact BigInt rational credal decision layer.
//
// Sits on top of verifyV4 / src/kernel/rational.js. Every payoff and
// probability is an exact BigInt rational; the only Number produced is a
// display-only decimalPreview. PROVED_OPTIMAL_ACTION is emitted only
// from a verified POINT/INTERVAL belief when one action strictly beats
// every rival at BOTH endpoints of [lo, hi] (linearity makes endpoint
// positivity necessary and sufficient on the whole interval).
import * as R from '../kernel/rational.js';
import { sha256hex, canonicalJSON } from '../kernel/receipt.js';
import { verifyV4, normSwapName } from './gate.js';

export const POISONED_BELIEF_CODES = [
  'SEMANTIC_NOT_ENTAILED',
  'UNSTABLE_BINDING',
  'SCOPED_OR_RETRACTED_PREMISE',
  'CONFLICTING_FIGURES',
  'CONTEXT_MISMATCH',
  'EXTRACTION_FAILED',
  'ENTAILMENT_WITNESS_UNAVAILABLE',
  'ENTAILMENT_WITNESS_MISMATCH',
  'ENTAILMENT_WITNESS_UNAUTHENTICATED',
  'PARSE_ERROR',
];
const POISONED = new Set(POISONED_BELIEF_CODES);

const rStr = (r) => R.toString(r);
function decimalPreview(r) {
  try {
    const v = R.toNumber(r);
    return Number.isFinite(v) ? v : null;
  } catch { return null; }
}

/** Parse an exact payoff/probability: string "p/q"|decimal|int, integer,
 *  finite number (via its decimal string), bigint, or {n,d} rational. */
export function parseExactPayoff(v) {
  if (v && typeof v === 'object' && 'n' in v && 'd' in v) {
    try { return R.make(BigInt(v.n), BigInt(v.d)); }
    catch { throw new Error('INVALID_PAYOFF: malformed rational object'); }
  }
  if (typeof v === 'bigint') return R.make(v, 1n);
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) throw new Error('INVALID_PAYOFF: non-finite number');
    return parseExactPayoff(String(v));
  }
  if (typeof v !== 'string') throw new Error('INVALID_PAYOFF: unsupported type');
  const s = v.trim();
  if (!s) throw new Error('INVALID_PAYOFF: empty string');
  if (/^[+-]?\d+\s*\/\s*[+-]?\d+$/.test(s)) {
    const [a, b] = s.split('/').map((x) => BigInt(x.trim()));
    if (b === 0n) throw new Error('INVALID_PAYOFF: zero denominator');
    return R.make(a, b);
  }
  if (/^[+-]?\d+$/.test(s)) return R.make(BigInt(s), 1n);
  if (/^[+-]?(?:\d+\.\d*|\.\d+)$/.test(s)) {
    const norm = s.startsWith('+') ? s.slice(1) : s;
    const fixed = norm.endsWith('.') ? `${norm}0` : norm.startsWith('.')
      ? `0${norm}` : norm.startsWith('-.') ? `-0${norm.slice(1)}` : norm;
    return R.parseDecimal(fixed);
  }
  throw new Error(`INVALID_PAYOFF: cannot parse "${v}"`);
}
const parseRat = parseExactPayoff;

function canonicalForReceipt(v) {
  try { return JSON.parse(JSON.stringify(v ?? null)); } catch { return String(v); }
}
function makeReceipt(parts) {
  return sha256hex(canonicalJSON({ schema: 'prism.credal_decision.v1', ...parts }));
}

function findPayoffKey(payoffs, state) {
  if (!payoffs || typeof payoffs !== 'object') return null;
  if (Object.hasOwn(payoffs, state)) return state;
  const want = String(state).trim().toLowerCase();
  const hits = Object.keys(payoffs).filter((k) => k.trim().toLowerCase() === want);
  return hits.length === 1 ? hits[0] : null;
}

// Returns {parsed:[{id,label,uT,uA}]} or {code, reason}
function parseActions(actions, targetHyp, altHyp) {
  if (!Array.isArray(actions) || actions.length < 2) {
    return { code: 'INVALID_ACTION_PAYOFFS', reason: 'actions must be an array of at least 2 actions' };
  }
  const seen = new Set();
  const parsed = [];
  for (const a of actions) {
    if (!a || typeof a !== 'object') return { code: 'INVALID_ACTION_PAYOFFS', reason: 'each action must be an object' };
    const id = typeof a.id === 'string' ? a.id.trim() : '';
    if (!id) return { code: 'INVALID_ACTION_PAYOFFS', reason: 'each action needs a non-empty string id' };
    if (seen.has(id)) return { code: 'INVALID_ACTION_PAYOFFS', reason: `duplicate action id "${id}"` };
    seen.add(id);
    if (!a.payoffs || typeof a.payoffs !== 'object') return { code: 'INVALID_ACTION_PAYOFFS', reason: `action "${id}" missing payoffs` };
    const tKey = findPayoffKey(a.payoffs, targetHyp);
    const aKey = findPayoffKey(a.payoffs, altHyp);
    if (!tKey || !aKey || tKey === aKey) {
      return { code: 'INVALID_ACTION_PAYOFFS', reason: `action "${id}" must have payoffs for both "${targetHyp}" and "${altHyp}"` };
    }
    try {
      parsed.push({
        id, label: typeof a.label === 'string' ? a.label : null,
        uT: parseExactPayoff(a.payoffs[tKey]),
        uA: parseExactPayoff(a.payoffs[aKey]),
      });
    } catch (e) {
      return { code: 'INVALID_ACTION_PAYOFFS', reason: `action "${id}": ${e.message}` };
    }
  }
  return { parsed };
}

const euAt = (p, act) => R.add(R.mul(p, act.uT), R.mul(R.sub(R.one(), p), act.uA));

function computeCore(lo, hi, parsed) {
  const enriched = parsed.map((act) => {
    const euLo = euAt(lo, act); const euHi = euAt(hi, act);
    return { ...act, euLo, euHi, euMin: R.min(euLo, euHi), euMax: R.max(euLo, euHi) };
  });
  const byId = new Map(enriched.map((a) => [a.id, a]));
  const diff = (a, b, p) => R.sub(euAt(p, byId.get(a)), euAt(p, byId.get(b)));
  const strictlyDominates = (a, b) => R.cmp(diff(a, b, lo), R.zero()) > 0 && R.cmp(diff(a, b, hi), R.zero()) > 0;

  const flipThresholds = [];
  for (let i = 0; i < enriched.length; i++) {
    for (let j = i + 1; j < enriched.length; j++) {
      const a = enriched[i]; const b = enriched[j];
      // p* = (u_b(alt)-u_a(alt)) / ((u_a(T)-u_b(T))-(u_a(alt)-u_b(alt)))
      const numer = R.sub(b.uA, a.uA);
      const denom = R.sub(R.sub(a.uT, b.uT), R.sub(a.uA, b.uA));
      if (R.isZero(denom)) continue;
      let pStar; try { pStar = R.div(numer, denom); } catch { continue; }
      if (R.cmp(pStar, lo) > 0 && R.cmp(pStar, hi) < 0) {
        flipThresholds.push({
          a: a.id, b: b.id, actions: [a.id, b.id], pair: [a.id, b.id],
          flipThreshold: rStr(pStar), pStar: rStr(pStar), threshold: rStr(pStar), rational: rStr(pStar),
          decimalPreview: decimalPreview(pStar),
          betterAtLo: R.cmp(a.euLo, b.euLo) > 0 ? a.id : R.cmp(a.euLo, b.euLo) < 0 ? b.id : null,
          betterAtHi: R.cmp(a.euHi, b.euHi) > 0 ? a.id : R.cmp(a.euHi, b.euHi) < 0 ? b.id : null,
          _p: pStar,
        });
      }
    }
  }
  flipThresholds.sort((x, y) => R.cmp(x._p, y._p) || String(x.a).localeCompare(String(y.a)) || String(x.b).localeCompare(String(y.b)));
  const allCrit = [];
  for (let i = 0; i < enriched.length; i++) for (let j = i + 1; j < enriched.length; j++) {
    const a = enriched[i]; const b = enriched[j];
    const denom = R.sub(R.sub(a.uT, b.uT), R.sub(a.uA, b.uA));
    if (R.isZero(denom)) continue;
    const pStar = R.div(R.sub(b.uA, a.uA), denom);
    if (R.cmp(pStar, lo) >= 0 && R.cmp(pStar, hi) <= 0) allCrit.push(pStar);
  }
  // E-admissible: optimal at lo, hi, any crossing, or any open segment mid.
  const crit = [lo, hi, ...allCrit].sort((x, y) => R.cmp(x, y));
  const uniq = crit.filter((v, i) => i === 0 || !R.eq(v, crit[i - 1]));
  const samples = [...uniq];
  for (let i = 0; i + 1 < uniq.length; i++) samples.push(R.div(R.add(uniq[i], uniq[i + 1]), R.make(2n, 1n)));
  const undominated = new Set();
  for (const p of samples) {
    let best = null;
    for (const a of enriched) { const e = euAt(p, a); if (best === null || R.cmp(e, best) > 0) best = e; }
    for (const a of enriched) if (R.eq(euAt(p, a), best)) undominated.add(a.id);
  }
  const undominatedActions = enriched.map((a) => a.id).filter((id) => undominated.has(id));
  let minimaxAction = null; let minimaxVal = null;
  for (const a of enriched) if (minimaxVal === null || R.cmp(a.euMin, minimaxVal) > 0) { minimaxVal = a.euMin; minimaxAction = a.id; }
  const minimaxActions = enriched.filter((a) => R.eq(a.euMin, minimaxVal)).map((a) => a.id);
  for (const f of flipThresholds) delete f._p;
  return { enriched, flipThresholds, undominatedActions, strictlyDominates, diff, minimaxAction, minimaxActions, minimaxVal };
}

function summaryOf(enriched, targetHyp, altHyp) {
  return enriched.map((a) => ({
    id: a.id, label: a.label,
    payoffs: { [targetHyp]: rStr(a.uT), [altHyp]: rStr(a.uA) },
    payoffTarget: rStr(a.uT), payoffAlt: rStr(a.uA),
    euAtLo: rStr(a.euLo), euAtHi: rStr(a.euHi),
    euMin: rStr(a.euMin), euMax: rStr(a.euMax),
    euInterval: [rStr(a.euMin), rStr(a.euMax)],
    euAtLoDecimal: decimalPreview(a.euLo), euAtHiDecimal: decimalPreview(a.euHi),
  }));
}
function parsedOnlySummary(parsed, targetHyp, altHyp) {
  return parsed.map((a) => ({
    id: a.id, label: a.label,
    payoffs: { [targetHyp]: rStr(a.uT), [altHyp]: rStr(a.uA) },
    payoffTarget: rStr(a.uT), payoffAlt: rStr(a.uA),
    euAtLo: null, euAtHi: null, euMin: null, euMax: null, euInterval: null,
  }));
}
function baseRefused({ code, reason, belief, targetHyp, altHyp, actions, actionsSummary = [], extra = {} }) {
  const receiptSha256 = makeReceipt({
    verdict: 'REFUSED', code, belief: canonicalForReceipt(belief),
    targetHyp: targetHyp ?? null, altHyp: altHyp ?? null, actions: canonicalForReceipt(actions),
  });
  return {
    verdict: 'REFUSED', code, reason: reason ?? null, optimalAction: null,
    targetHyp: targetHyp ?? null, altHyp: altHyp ?? null,
    targetState: targetHyp ?? null, altState: altHyp ?? null,
    belief: belief ?? null, euInterval: null, pointOptimal: false,
    dominanceMargins: {}, flipThresholds: [], undominatedActions: [],
    minimaxAction: null, maximinAction: null, actionsSummary,
    missingInformation: code, receiptSha256, ...extra,
  };
}

/**
 * decideFromBelief({belief, actions, targetHyp, altHyp})
 * Also accepts targetState/altState aliases. Pure exact math; never
 * throws for domain inputs — invalid inputs return REFUSED fail-closed.
 */
export function decideFromBelief(input = {}) {
  const belief = input?.belief ?? null;
  const actions = input?.actions;
  const targetHyp = input?.targetHyp ?? input?.targetState ?? null;
  const altHyp = input?.altHyp ?? input?.altState ?? null;
  const refused = (code, reason, actionsSummary = []) => baseRefused({
    code, reason, belief: belief ? {
      status: belief.status ?? null,
      lo: belief.lo !== undefined ? String(belief.lo?.n !== undefined ? `${belief.lo.n}/${belief.lo.d}` : belief.lo) : null,
      hi: belief.hi !== undefined ? String(belief.hi?.n !== undefined ? `${belief.hi.n}/${belief.hi.d}` : belief.hi) : null,
      code: belief.code ?? null, reason: belief.reason ?? null,
    } : null, targetHyp, altHyp, actions, actionsSummary,
  });

  if (typeof targetHyp !== 'string' || !targetHyp.trim() || typeof altHyp !== 'string' || !altHyp.trim()
      || targetHyp.trim().toLowerCase() === altHyp.trim().toLowerCase()) {
    return refused('INVALID_ACTION_PAYOFFS', 'targetHyp and altHyp must be distinct non-empty state names');
  }
  const tHyp = targetHyp.trim(); const aHyp = altHyp.trim();
  if (!belief || typeof belief !== 'object' || typeof belief.status !== 'string') {
    return refused('INVALID_BELIEF_INTERVAL', 'belief with status POINT|INTERVAL|REFUSED is required');
  }
  const status = belief.status;
  // Poisoned-premise guard: a REFUSED belief never licenses a decision,
  // except a verified UNDERDETERMINED_INTERVAL carrying a real [lo,hi].
  if (status === 'REFUSED') {
    const code = belief.code ?? 'REFUSED_BELIEF';
    const isVerifiedInterval = code === 'UNDERDETERMINED_INTERVAL' && belief.lo !== undefined && belief.hi !== undefined;
    if (!isVerifiedInterval) {
      const actParse = parseActions(actions, tHyp, aHyp);
      const out = refused(POISONED.has(code) || true ? code : code,
        belief.reason ?? `belief refused (${code}); no verified credal interval — decision refused`,
        actParse.parsed ? parsedOnlySummary(actParse.parsed, tHyp, aHyp) : []);
      out.targetHyp = tHyp; out.altHyp = aHyp; out.targetState = tHyp; out.altState = aHyp;
      return out;
    }
    // fall through and decide over the verified interval
  } else if (status !== 'POINT' && status !== 'INTERVAL') {
    return refused('INVALID_BELIEF_INTERVAL', `unknown belief status "${status}"`);
  }

  let lo; let hi;
  try {
    if (belief.lo === undefined && belief.hi === undefined) throw new Error('missing lo/hi');
    lo = parseRat(belief.lo !== undefined ? belief.lo : belief.hi);
    hi = parseRat(belief.hi !== undefined ? belief.hi : belief.lo);
  } catch (e) {
    return refused('INVALID_BELIEF_INTERVAL', `invalid belief interval: ${e.message}`);
  }
  if (status === 'POINT' && !R.eq(lo, hi)) {
    return refused('INVALID_BELIEF_INTERVAL', 'POINT belief requires lo === hi');
  }
  if (R.cmp(lo, R.zero()) < 0 || R.cmp(hi, R.one()) > 0 || R.cmp(lo, hi) > 0) {
    return refused('INVALID_BELIEF_INTERVAL', `belief interval [${rStr(lo)}, ${rStr(hi)}] violates 0 <= lo <= hi <= 1`);
  }
  const actParse = parseActions(actions, tHyp, aHyp);
  if (!actParse.parsed) return refused(actParse.code, actParse.reason);
  const parsed = actParse.parsed;
  const core = computeCore(lo, hi, parsed);
  const { enriched, flipThresholds, undominatedActions, minimaxAction, minimaxActions } = core;
  const actionsSummary = summaryOf(enriched, tHyp, aHyp);
  const beliefOut = {
    status, lo: rStr(lo), hi: rStr(hi),
    loDecimal: decimalPreview(lo), hiDecimal: decimalPreview(hi),
    code: belief.code ?? null, reason: belief.reason ?? null,
  };
  const canonicalActions = parsed.map((a) => ({ id: a.id, label: a.label, payoffs: { [tHyp]: rStr(a.uT), [aHyp]: rStr(a.uA) } }));

  let optimal = null;
  for (const cand of enriched) {
    if (enriched.every((o) => o.id === cand.id || core.strictlyDominates(cand.id, o.id))) { optimal = cand; break; }
  }
  if (optimal) {
    const dominanceMargins = {};
    const dominanceMarginsList = [];
    for (const o of enriched) {
      if (o.id === optimal.id) continue;
      const m = R.min(core.diff(optimal.id, o.id, lo), core.diff(optimal.id, o.id, hi));
      dominanceMargins[o.id] = rStr(m);
      dominanceMarginsList.push({ against: o.id, action: o.id, margin: rStr(m), worstCaseMargin: rStr(m), decimalPreview: decimalPreview(m) });
    }
    const receiptSha256 = makeReceipt({
      verdict: 'PROVED_OPTIMAL_ACTION', optimalAction: optimal.id, belief: beliefOut,
      targetHyp: tHyp, altHyp: aHyp, actions: canonicalActions,
      euInterval: [rStr(optimal.euMin), rStr(optimal.euMax)], dominanceMargins, flipThresholds,
    });
    return {
      verdict: 'PROVED_OPTIMAL_ACTION', code: null, reason: null,
      optimalAction: optimal.id, targetHyp: tHyp, altHyp: aHyp, targetState: tHyp, altState: aHyp,
      belief: beliefOut, lo: rStr(lo), hi: rStr(hi),
      euInterval: [rStr(optimal.euMin), rStr(optimal.euMax)],
      euMin: rStr(optimal.euMin), euMax: rStr(optimal.euMax),
      pointOptimal: R.eq(lo, hi), dominanceMargins, dominanceMarginsList,
      flipThresholds, undominatedActions, dominatedActions: enriched.map((a) => a.id).filter((id) => !undominatedActions.includes(id)),
      minimaxAction, maximinAction: minimaxAction, minimaxActions,
      actionsSummary, missingInformation: null, receiptSha256,
    };
  }
  const missingInformation = belief.code ?? 'UNDERDETERMINED_INTERVAL';
  const dominatedActions = enriched.map((a) => a.id).filter((id) => !undominatedActions.includes(id));
  const receiptSha256 = makeReceipt({
    verdict: 'DECISION_INDETERMINATE', belief: beliefOut, targetHyp: tHyp, altHyp: aHyp,
    actions: canonicalActions, undominatedActions, flipThresholds, missingInformation,
  });
  // Convenience singular alias when exactly one interior flip exists.
  const singleFlip = flipThresholds.length === 1 ? flipThresholds[0].flipThreshold : null;
  return {
    verdict: 'DECISION_INDETERMINATE', code: missingInformation, reason: belief.reason ?? null,
    optimalAction: null, targetHyp: tHyp, altHyp: aHyp, targetState: tHyp, altState: aHyp,
    belief: beliefOut, lo: rStr(lo), hi: rStr(hi), euInterval: null,
    pointOptimal: R.eq(lo, hi), dominanceMargins: {}, flipThresholds,
    ...(singleFlip ? { flipThreshold: singleFlip } : {}),
    undominatedActions, dominatedActions,
    minimaxAction, maximinAction: minimaxAction, minimaxActions,
    minimaxEuMin: core.minimaxVal ? rStr(core.minimaxVal) : null,
    actionsSummary, missingInformation, receiptSha256,
  };
}

// ---------------- decideFromProse ----------------
const normState = (s) => normSwapName(String(s ?? ''));
function matchesState(key, id, name) {
  if (key === null || key === undefined) return false;
  const k = String(key).trim();
  if (id && k.toLowerCase() === String(id).trim().toLowerCase()) return true;
  if (name && normState(k) && normState(k) === normState(name)) return true;
  return false;
}
function payoffKeysOf(actions) {
  const keys = [];
  for (const a of actions ?? []) {
    if (a?.payoffs && typeof a.payoffs === 'object') {
      for (const k of Object.keys(a.payoffs)) if (!keys.includes(k)) keys.push(k);
    }
  }
  return keys;
}
// Resolve {targetKey, altKey} payoff keys, or {code, reason}.
function resolveMapping({ actions, targetState, altState, targetId, altId, targetName, altName }) {
  const keys = payoffKeysOf(actions);
  if (keys.length === 0) return { code: 'INVALID_ACTION_PAYOFFS', reason: 'actions carry no payoff keys' };
  const tMatches = keys.filter((k) => matchesState(k, targetId, targetName));
  const aMatches = keys.filter((k) => matchesState(k, altId, altName));
  if (tMatches.length === 1 && aMatches.length === 1 && tMatches[0] !== aMatches[0]) {
    if (targetState && !matchesState(targetState, targetId, targetName) && matchesState(targetState, altId, altName)) {
      return { code: 'INVALID_ACTION_PAYOFFS', reason: 'targetState names the alternative hypothesis, not the verified target' };
    }
    if (altState && !matchesState(altState, altId, altName) && matchesState(altState, targetId, targetName)) {
      return { code: 'INVALID_ACTION_PAYOFFS', reason: 'altState names the verified target, not the alternative hypothesis' };
    }
    return { targetKey: tMatches[0], altKey: aMatches[0] };
  }
  if (tMatches.length > 1 || aMatches.length > 1 || (tMatches.length === 1 && aMatches.length === 1)) {
    return { code: 'INVALID_ACTION_PAYOFFS', reason: 'payoff keys ambiguously match both hypotheses' };
  }
  // No (or only partial) identity match: fall back to caller's declared
  // targetState/altState payoff keys, if both are present in payoffs.
  if (targetState && altState) {
    const tk = keys.find((k) => k === targetState) ?? (keys.filter((k) => k.trim().toLowerCase() === String(targetState).trim().toLowerCase())[0] ?? null);
    const ak = keys.find((k) => k === altState) ?? (keys.filter((k) => k.trim().toLowerCase() === String(altState).trim().toLowerCase())[0] ?? null);
    // A partial identity match must agree with the declared keys.
    if (tMatches.length === 1 && tk && tk !== tMatches[0]) return { code: 'INVALID_ACTION_PAYOFFS', reason: 'targetState contradicts the verified target payoff key' };
    if (aMatches.length === 1 && ak && ak !== aMatches[0]) return { code: 'INVALID_ACTION_PAYOFFS', reason: 'altState contradicts the alternative payoff key' };
    if (tk && ak && tk !== ak) return { targetKey: tk, altKey: ak };
  }
  if (tMatches.length === 1 || aMatches.length === 1) {
    return { code: 'INVALID_ACTION_PAYOFFS', reason: 'only one hypothesis matches a payoff key; mapping is incomplete' };
  }
  return { code: 'INVALID_ACTION_PAYOFFS', reason: `cannot map verified target "${targetName ?? targetId}" to payoff keys [${keys.join(', ')}] by id or name, and no unambiguous targetState/altState keys were given` };
}
function witnessSummaryOf(w) {
  if (!w || typeof w !== 'object') return null;
  return {
    entailed: w.entailed ?? null, code: w.code ?? null, reason: w.reason ?? null,
    witnessModel: w.witnessModel ?? null, witnessHost: w.witnessHost ?? null,
    passageSha256: w.passageSha256 ?? null, programCanonicalSha256: w.programCanonicalSha256 ?? null,
    checks: Array.isArray(w.checks) ? w.checks.map((c) => ({ id: c.id, entailed: c.entailed, reason: c.reason ?? null })) : [],
  };
}
function advisoryOverUnit(actions, targetKey, altKey) {
  if (!targetKey || !altKey) return { flipThresholds: [], undominatedActions: [], minimaxAction: null, actionsSummary: [] };
  const p = parseActions(actions, targetKey, altKey);
  if (!p.parsed) return { flipThresholds: [], undominatedActions: [], minimaxAction: null, actionsSummary: [] };
  const core = computeCore(R.zero(), R.one(), p.parsed);
  return {
    flipThresholds: core.flipThresholds, undominatedActions: core.undominatedActions,
    minimaxAction: core.minimaxAction, maximinAction: core.minimaxAction,
    actionsSummary: summaryOf(core.enriched, targetKey, altKey),
  };
}

/**
 * decideFromProse({X, actions, targetState?, altState?, extract?,
 *   witnessCheck?, maxSwaps?, model?, witnessModel?, ollamaHost?, verifyV4?})
 * verifyV4 is an internal injection point (MCP tests); normal callers
 * inject extract/witnessCheck instead, exactly as with verifyV4.
 */
export async function decideFromProse(input = {}) {
  const X = input?.X ?? input?.passage ?? null;
  const actions = input?.actions;
  const targetState = input?.targetState ?? input?.targetHyp ?? null;
  const altState = input?.altState ?? input?.altHyp ?? null;
  const verifyFn = input?.verifyV4 ?? input?.verifyFn ?? verifyV4;
  const passageSha256 = typeof X === 'string' ? sha256hex(X) : null;
  const proseRefused = (code, reason, extra = {}) => {
    const receiptSha256 = makeReceipt({ verdict: 'REFUSED', code, passageSha256, actions: canonicalForReceipt(actions), targetState, altState });
    return {
      verdict: 'REFUSED', code, reason: reason ?? null, optimalAction: null,
      targetState: targetState ?? null, altState: altState ?? null,
      targetHyp: targetState ?? null, altHyp: altState ?? null,
      belief: null, passageSha256, programCanonicalSha256: null,
      witness: null, witnessSummary: null, flipThresholds: [],
      whatWouldBeNeeded: null, actionsSummary: [], missingInformation: code,
      receiptSha256, decisionReceiptSha256: receiptSha256, ...extra,
    };
  };
  if (typeof X !== 'string' || !X.trim()) return proseRefused('INVALID_ACTION_PAYOFFS', 'X (passage) must be a non-empty string');
  if (!Array.isArray(actions) || actions.length < 2) return proseRefused('INVALID_ACTION_PAYOFFS', 'actions must be an array of at least 2 actions');

  let res;
  try {
    res = await verifyFn({
      X,
      ...(input.extract !== undefined ? { extract: input.extract } : {}),
      ...(input.witnessCheck !== undefined ? { witnessCheck: input.witnessCheck } : {}),
      ...(input.maxSwaps !== undefined ? { maxSwaps: input.maxSwaps } : {}),
      ...(input.model !== undefined ? { model: input.model } : {}),
      ...(input.witnessModel !== undefined ? { witnessModel: input.witnessModel } : {}),
      ...(input.ollamaHost !== undefined ? { ollamaHost: input.ollamaHost } : {}),
      ...(input.allowCustomWitnessEndpoint !== undefined ? { allowCustomWitnessEndpoint: input.allowCustomWitnessEndpoint } : {}),
    });
  } catch (e) {
    return proseRefused('EXTRACTION_FAILED', `verifyV4 threw: ${e.message}`);
  }
  const prog = res?.compiled?.program ?? null;
  const hypNames = res?.compiled?.hypNames ?? prog?.hypNames ?? {};
  const evNames = res?.compiled?.evNames ?? prog?.evNames ?? {};
  const targetId = res?.decision?.target ?? prog?.query?.target ?? null;
  const targetKind = prog?.query?.targetKind ?? (targetId && hypNames[targetId] !== undefined ? 'hyp' : targetId ? 'hyp' : null);
  let partition = prog?.decls?.find((d) => d.op === 'PART_EXHAUSTIVE')?.hyps ?? null;
  if (!partition && prog) partition = prog.decls.filter((d) => d.op === 'HYP').map((d) => d.id);
  if ((!partition || partition.length === 0) && Object.keys(hypNames).length) partition = Object.keys(hypNames);
  const altId = partition && targetId ? (partition.find((h) => h !== targetId) ?? null) : null;
  const targetName = targetKind === 'ev' ? (evNames[targetId] ?? targetId) : (hypNames[targetId] ?? targetId);
  const altName = altId ? (hypNames[altId] ?? altId) : null;
  const wSummary = witnessSummaryOf(res?.witness);
  const programCanonicalSha256 = wSummary?.programCanonicalSha256 ?? null;
  const verification = {
    verdict: res?.verdict ?? null, code: res?.code ?? null, rational: res?.rational ?? null,
    target: targetId, targetKind, targetName: targetName ?? null, altTarget: altId, altName: altName ?? null,
    swaps: res?.swaps ?? [], passageSha256, programCanonicalSha256,
  };

  if (res?.verdict === 'REFUSED') {
    const code = res.code ?? 'REFUSED';
    const reason = res.details?.detail ?? res.decision?.details?.detail ?? res.details?.message ?? null;
    // Advisory-only flips over [0,1]: map via declared keys when possible.
    let adv = { flipThresholds: [], undominatedActions: [], minimaxAction: null, actionsSummary: [] };
    let advTarget = targetState; let advAlt = altState;
    if (targetId && altId && partition?.length === 2) {
      const m = resolveMapping({ actions, targetState, altState, targetId, altId, targetName, altName });
      if (m.targetKey) { advTarget = m.targetKey; advAlt = m.altKey; }
    }
    if (advTarget && advAlt) adv = advisoryOverUnit(actions, advTarget, advAlt);
    const whatWouldBeNeeded = {
      note: 'advisory only — computed over [0,1] because no verified belief was released; not a decision',
      missingInformation: code, flipThresholds: adv.flipThresholds,
      undominatedActions: adv.undominatedActions, minimaxAction: adv.minimaxAction,
    };
    const receiptSha256 = makeReceipt({ verdict: 'REFUSED', code, passageSha256, programCanonicalSha256, actions: canonicalForReceipt(actions), flipThresholds: adv.flipThresholds });
    return {
      verdict: 'REFUSED', code, reason, optimalAction: null,
      targetState: targetState ?? advTarget ?? null, altState: altState ?? advAlt ?? null,
      targetHyp: targetId, altHyp: altId, belief: { status: 'REFUSED', code, lo: null, hi: null },
      verification, verifyV4: verification, passageSha256, programCanonicalSha256,
      witness: res.witness ?? null, witnessSummary: wSummary,
      flipThresholds: adv.flipThresholds, whatWouldBeNeeded,
      undominatedActions: adv.undominatedActions, actionsSummary: adv.actionsSummary,
      missingInformation: code, receiptSha256, decisionReceiptSha256: receiptSha256,
      swaps: res.swaps ?? [], compiled: res.compiled ?? null,
    };
  }
  if (res?.verdict !== 'PASS') {
    return proseRefused(res?.code ?? 'EXTRACTION_FAILED', 'verifyV4 did not PASS', { verification, witnessSummary: wSummary, passageSha256 });
  }
  if (!targetId || targetKind === 'hyp' && (!partition || partition.length !== 2 || !altId)) {
    return proseRefused('INVALID_ACTION_PAYOFFS', 'verified query is not over a binary hypothesis partition; cannot map to binary payoffs',
      { verification, verifyV4: verification, witness: res.witness ?? null, witnessSummary: wSummary, passageSha256, programCanonicalSha256 });
  }
  // Exact point belief, un-negating a NOT query (credal value is P(not T)).
  let value;
  try {
    const raw = res.value ?? res.decision?.value ?? (res.rational ? parseRat(res.rational) : null);
    if (!raw) throw new Error('PASS without a rational value');
    value = raw && typeof raw === 'object' && 'n' in raw ? R.make(BigInt(raw.n), BigInt(raw.d)) : parseRat(raw);
  } catch (e) {
    return proseRefused('INVALID_BELIEF_INTERVAL', e.message, { verification, witnessSummary: wSummary });
  }
  if (prog?.query?.negated) value = R.sub(R.one(), value);
  const mapping = targetKind === 'ev'
    ? (targetState && altState ? resolveMapping({ actions, targetState, altState, targetId, altId: null, targetName, altName: null }) : { code: 'INVALID_ACTION_PAYOFFS', reason: 'event-target query needs explicit targetState/altState payoff keys' })
    : resolveMapping({ actions, targetState, altState, targetId, altId, targetName, altName });
  if (!mapping.targetKey) {
    const out = proseRefused(mapping.code, mapping.reason, {
      verification, verifyV4: verification, witness: res.witness ?? null, witnessSummary: wSummary,
      passageSha256, programCanonicalSha256,
      belief: { status: 'POINT', lo: rStr(value), hi: rStr(value), rational: rStr(value), target: targetId, targetName },
    });
    return out;
  }
  const decided = decideFromBelief({
    belief: { status: 'POINT', lo: value, hi: value },
    actions, targetHyp: mapping.targetKey, altHyp: mapping.altKey,
  });
  const belief = {
    status: 'POINT', lo: rStr(value), hi: rStr(value), rational: rStr(value),
    decimalPreview: decimalPreview(value), target: targetId, targetName: targetName ?? null,
    altTarget: altId, altName: altName ?? null,
  };
  return {
    ...decided, belief,
    targetHyp: mapping.targetKey, altHyp: mapping.altKey,
    targetState: mapping.targetKey, altState: mapping.altKey,
    verifiedTarget: targetId, verifiedTargetName: targetName ?? null,
    verifiedAlt: altId, verifiedAltName: altName ?? null,
    verification, verifyV4: verification,
    value: rStr(value), rational: rStr(value),
    passageSha256, programCanonicalSha256,
    witness: res.witness ?? null, witnessSummary: wSummary,
    swaps: res.swaps ?? [], compiled: res.compiled ?? null,
    decisionReceiptSha256: decided.receiptSha256,
  };
}

