// prism-kernel :: Factor-Graph Bytecode parser / validator.
// Grammar (ARCH-SPEC §2.3):
//   Program   := Decl+ Query END
//   Decl      := "HYP" HypId Span
//              | "PART_EXHAUSTIVE" "(" HypId+ ")" Span
//              | "EVENT" EvId Span
//              | "PRIOR" HypId Rational CtxId Span
//              | "LIK" EvId Polarity "GIVEN" HypId Rational CtxId Span
//              | "POSTERIOR_STATED" HypId "GIVEN" EvId Polarity Rational CtxId Span
//              | "JOINT_LIK" "(" (EvId ":" Polarity)+ ")" "GIVEN" HypId Rational CtxId Span
//              | "DET_LINK" EvId LinkRel EvId CtxId Span
//              | "OBS" EvId Polarity CtxId Span
//              | "COND_INDEP" "(" EvId+ ")" "GIVEN" HypId CtxId Span
//              | "SENT_TAG" Int SentTag Span
//              | "CONFLICT" Span Span
//              | "NON_ASSERTED" Span ReasonCode
//   Query     := "ASK_PROB" TargetExpr "GIVEN" "(" ObsRef* ")" CtxId
//   Rational  := Int "/" PosInt | Int
//   Polarity  := "POS" | "NEG"
//   ReasonCode:= "QUOTED" | "RETRACTED" | "HYPOTHETICAL" | "HEDGED_APPROX"
//   LinkRel   := "EQUIV" | "IMPLIES" | "MUTEX"
//   SentTag   := "PREMISE" | "QUERY" | "CONTEXT_SCOPE" | "NON_ASSERTED" | "FLUFF"
//   Span      := "[" Int "," Int "]"
//   TargetExpr:= ("NOT")? (HypId | EvId)
//   ObsRef    := EvId (":" Polarity)?
// CtxId "*" is a wildcard accepted on premise decls (matches any query ctx).

import * as R from './rational.js';

export const REASON_CODES = new Set(['QUOTED', 'RETRACTED', 'HYPOTHETICAL', 'HEDGED_APPROX']);
export const LINK_RELS = new Set(['EQUIV', 'IMPLIES', 'MUTEX']);
export const SENT_TAGS = new Set(['PREMISE', 'QUERY', 'CONTEXT_SCOPE', 'NON_ASSERTED', 'FLUFF']);

export class ParseError extends Error {
  constructor(msg, pos) {
    super(`PARSE_ERROR: ${msg}${pos !== undefined ? ` (at token ${pos})` : ''}`);
    this.code = 'PARSE_ERROR';
    this.pos = pos;
  }
}

// ---------- tokenizer ----------
const TOKEN_RE = /\s*(\[|\]|\(|\)|:|,|[A-Za-z_][A-Za-z0-9_\-]*|-?\d+(?:\s*\/\s*\d+)?|\*)/y;

function tokenize(src) {
  const toks = [];
  let i = 0;
  while (i < src.length) {
    if (/\s/.test(src[i])) {
      i++;
      continue;
    }
    TOKEN_RE.lastIndex = i;
    const m = TOKEN_RE.exec(src);
    if (!m || m.index !== i) throw new ParseError(`unexpected character ${JSON.stringify(src[i])}`, toks.length);
    toks.push(m[1]);
    i = TOKEN_RE.lastIndex;
  }
  return toks;
}

// ---------- recursive-descent parser ----------
class P {
  constructor(toks) {
    this.toks = toks;
    this.i = 0;
  }
  peek() {
    return this.toks[this.i];
  }
  next() {
    const t = this.toks[this.i];
    if (t === undefined) throw new ParseError('unexpected end of program', this.i);
    this.i++;
    return t;
  }
  expect(t) {
    const got = this.next();
    if (got !== t) throw new ParseError(`expected "${t}", got "${got}"`, this.i - 1);
    return got;
  }
  id() {
    const t = this.next();
    if (!/^[A-Za-z_][A-Za-z0-9_\-]*$/.test(t) || KEYWORDS.has(t)) {
      throw new ParseError(`expected identifier, got "${t}"`, this.i - 1);
    }
    return t;
  }
  ctxId() {
    const t = this.next();
    if (t === '*') return '*';
    if (!/^[A-Za-z_][A-Za-z0-9_\-]*$/.test(t)) throw new ParseError(`expected CtxId, got "${t}"`, this.i - 1);
    return t;
  }
  rational() {
    const t = this.next();
    if (!/^-?\d+(\s*\/\s*\d+)?$/.test(t)) throw new ParseError(`expected Rational, got "${t}"`, this.i - 1);
    let r;
    try {
      r = R.parse(t.replace(/\s+/g, ''));
    } catch (e) {
      throw new ParseError(e.message, this.i - 1);
    }
    if (R.cmp(r, R.zero()) < 0 || R.cmp(r, R.one()) > 0) {
      throw new ParseError(`probability rational out of [0,1]: "${t}"`, this.i - 1);
    }
    return r;
  }
  polarity() {
    const t = this.next();
    if (t !== 'POS' && t !== 'NEG') throw new ParseError(`expected POS|NEG, got "${t}"`, this.i - 1);
    return t;
  }
}

const KEYWORDS = new Set([
  'HYP', 'PART_EXHAUSTIVE', 'EVENT', 'PRIOR', 'LIK', 'POSTERIOR_STATED', 'OBS',
  'COND_INDEP', 'CONFLICT', 'NON_ASSERTED', 'ASK_PROB', 'GIVEN', 'POS', 'NEG',
  'NOT', 'END', 'QUOTED', 'RETRACTED', 'HYPOTHETICAL', 'HEDGED_APPROX',
  'JOINT_LIK', 'DET_LINK', 'SENT_TAG', 'EQUIV', 'IMPLIES', 'MUTEX',
  'PREMISE', 'QUERY', 'CONTEXT_SCOPE', 'FLUFF',
]);

function parseSpan(p) {
  p.expect('[');
  const sTok = p.next();
  p.expect(',');
  const eTok = p.next();
  p.expect(']');
  if (!/^-?\d+$/.test(sTok) || !/^-?\d+$/.test(eTok)) {
    throw new ParseError('span bounds must be integers', p.i);
  }
  const s = Number(sTok);
  const e = Number(eTok);
  if (!Number.isInteger(s) || !Number.isInteger(e) || s < 0 || e <= s) {
    throw new ParseError(`malformed span [${sTok},${eTok}]`, p.i);
  }
  return { s, e };
}

function parseIdList(p) {
  p.expect('(');
  const ids = [];
  while (p.peek() !== ')') ids.push(p.id());
  p.expect(')');
  if (ids.length === 0) throw new ParseError('empty id list', p.i);
  return ids;
}

function parseDecl(p, op) {
  switch (op) {
    case 'HYP':
      return { op, id: p.id(), span: parseSpan(p) };
    case 'PART_EXHAUSTIVE':
      return { op, hyps: parseIdList(p), span: parseSpan(p) };
    case 'EVENT':
      return { op, id: p.id(), span: parseSpan(p) };
    case 'PRIOR':
      return { op, hyp: p.id(), value: p.rational(), ctx: p.ctxId(), span: parseSpan(p) };
    case 'LIK': {
      const ev = p.id();
      const pol = p.polarity();
      p.expect('GIVEN');
      const hyp = p.id();
      const value = p.rational();
      const ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, ev, pol, hyp, value, ctx, span };
    }
    case 'POSTERIOR_STATED': {
      const hyp = p.id();
      p.expect('GIVEN');
      const ev = p.id();
      const pol = p.polarity();
      const value = p.rational();
      const ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, hyp, ev, pol, value, ctx, span };
    }
    case 'JOINT_LIK': {
      p.expect('(');
      const spec = [];
      while (p.peek() !== ')') {
        const ev = p.id();
        p.expect(':');
        const pol = p.polarity();
        spec.push({ ev, pol });
      }
      p.expect(')');
      if (spec.length < 2) throw new ParseError('JOINT_LIK needs >=2 events', p.i);
      p.expect('GIVEN');
      const hyp = p.id();
      const value = p.rational();
      const ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, spec, hyp, value, ctx, span };
    }
    case 'DET_LINK': {
      const a = p.id();
      const rel = p.next();
      if (!LINK_RELS.has(rel)) throw new ParseError(`expected EQUIV|IMPLIES|MUTEX, got "${rel}"`, p.i - 1);
      const b = p.id();
      const ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, a, rel, b, ctx, span };
    }
    case 'SENT_TAG': {
      const t = p.next();
      if (!/^-?\d+$/.test(t)) throw new ParseError(`expected sentence index, got "${t}"`, p.i - 1);
      const idx = Number(t);
      if (!Number.isInteger(idx) || idx < 0) throw new ParseError('SENT_TAG index must be >= 0', p.i - 1);
      const tag = p.next();
      if (!SENT_TAGS.has(tag)) throw new ParseError(`bad SentTag "${tag}"`, p.i - 1);
      // optional CtxId binding — required semantically for CONTEXT_SCOPE tags
      // (checked by the sentence accountant), optional for the rest
      let ctx = null;
      if (p.peek() !== '[') ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, idx, tag, ctx, span };
    }
    case 'OBS': {
      const ev = p.id();
      const pol = p.polarity();
      const ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, ev, pol, ctx, span };
    }
    case 'COND_INDEP': {
      const evs = parseIdList(p);
      p.expect('GIVEN');
      const hyp = p.peek() === '*' ? (p.next(), '*') : p.id();
      const ctx = p.ctxId();
      const span = parseSpan(p);
      return { op, evs, hyp, ctx, span };
    }
    case 'CONFLICT':
      return { op, spanA: parseSpan(p), spanB: parseSpan(p) };
    case 'NON_ASSERTED': {
      const span = parseSpan(p);
      const reason = p.next();
      if (!REASON_CODES.has(reason)) throw new ParseError(`bad ReasonCode "${reason}"`, p.i - 1);
      return { op, span, reason };
    }
    default:
      throw new ParseError(`unknown decl op "${op}"`, p.i - 1);
  }
}

function parseQuery(p) {
  p.expect('ASK_PROB');
  let negated = false;
  if (p.peek() === 'NOT') {
    p.next();
    negated = true;
  }
  const target = p.id();
  p.expect('GIVEN');
  p.expect('(');
  const given = [];
  while (p.peek() !== ')') {
    const ev = p.id();
    let pol = 'POS';
    if (p.peek() === ':') {
      p.next();
      pol = p.polarity();
    }
    given.push({ ev, pol });
  }
  p.expect(')');
  const ctx = p.ctxId();
  if (ctx === '*') throw new ParseError('query ctx cannot be wildcard', p.i);
  return { op: 'ASK_PROB', target, negated, given, ctx };
}

const DECL_OPS = new Set([
  'HYP', 'PART_EXHAUSTIVE', 'EVENT', 'PRIOR', 'LIK', 'POSTERIOR_STATED', 'OBS',
  'COND_INDEP', 'CONFLICT', 'NON_ASSERTED', 'JOINT_LIK', 'DET_LINK', 'SENT_TAG',
]);

export function parseBytecode(src) {
  const p = new P(tokenize(src));
  const decls = [];
  let query = null;
  while (p.peek() !== undefined && p.peek() !== 'END') {
    const t = p.next();
    if (t === 'ASK_PROB') {
      if (query) throw new ParseError('multiple ASK_PROB queries', p.i - 1);
      p.i--; // re-parse through parseQuery for uniform errors
      query = parseQuery(p);
      continue;
    }
    if (!DECL_OPS.has(t)) throw new ParseError(`expected decl op or ASK_PROB, got "${t}"`, p.i - 1);
    if (query) throw new ParseError('decls must precede ASK_PROB', p.i - 1);
    decls.push(parseDecl(p, t));
  }
  p.expect('END');
  if (p.peek() !== undefined) throw new ParseError('trailing tokens after END', p.i);
  if (!query) throw new ParseError('program has no ASK_PROB query', p.i);
  if (decls.length === 0) throw new ParseError('program has no decls', p.i);
  const prog = { decls, query };
  validateProgram(prog);
  return prog;
}

// ---------- semantic validation (structure, id resolution, spans) ----------
export function validateProgram(prog, textLen = null) {
  const hypIds = new Set();
  const evIds = new Set();
  for (const d of prog.decls) {
    if (d.op === 'HYP') {
      if (hypIds.has(d.id)) throw new ParseError(`duplicate HYP "${d.id}"`);
      hypIds.add(d.id);
    }
    if (d.op === 'EVENT') {
      if (evIds.has(d.id)) throw new ParseError(`duplicate EVENT "${d.id}"`);
      evIds.add(d.id);
    }
  }
  const needHyp = (id) => {
    if (!hypIds.has(id)) throw new ParseError(`undeclared HYP "${id}"`);
  };
  const needEv = (id) => {
    if (!evIds.has(id)) throw new ParseError(`undeclared EVENT "${id}"`);
  };
  const checkSpan = (sp) => {
    if (textLen !== null && (sp.s < 0 || sp.e > textLen)) {
      throw new ParseError(`span [${sp.s},${sp.e}] out of bounds for input length ${textLen}`);
    }
  };
  for (const d of prog.decls) {
    if (d.span) checkSpan(d.span);
    switch (d.op) {
      case 'PART_EXHAUSTIVE':
        if (new Set(d.hyps).size !== d.hyps.length) throw new ParseError('PART_EXHAUSTIVE has duplicate hyps');
        d.hyps.forEach(needHyp);
        break;
      case 'PRIOR':
        needHyp(d.hyp);
        break;
      case 'LIK':
        needEv(d.ev);
        needHyp(d.hyp);
        break;
      case 'POSTERIOR_STATED':
        needHyp(d.hyp);
        needEv(d.ev);
        break;
      case 'OBS':
        needEv(d.ev);
        break;
      case 'COND_INDEP':
        d.evs.forEach(needEv);
        if (d.hyp !== '*') needHyp(d.hyp);
        break;
      case 'JOINT_LIK':
        d.spec.forEach((s) => needEv(s.ev));
        needHyp(d.hyp);
        break;
      case 'DET_LINK':
        needEv(d.a);
        needEv(d.b);
        break;
      case 'CONFLICT':
        checkSpan(d.spanA);
        checkSpan(d.spanB);
        break;
      default:
        break;
    }
  }
  const q = prog.query;
  const targetIsHyp = hypIds.has(q.target);
  const targetIsEv = evIds.has(q.target);
  if (!targetIsHyp && !targetIsEv) throw new ParseError(`ASK_PROB target "${q.target}" is neither HYP nor EVENT`);
  q.targetKind = targetIsHyp ? 'hyp' : 'ev';
  q.given.forEach((g) => needEv(g.ev));
  return prog;
}

// ---------- canonical JSON / bytecode round-trip ----------
function rat(v) {
  return typeof v === 'string' ? R.parse(v) : v;
}

export function toJSON(prog) {
  const ser = (d) => {
    const o = { ...d };
    if (o.value !== undefined) o.value = R.toString(o.value);
    return o;
  };
  return {
    decls: prog.decls.map(ser),
    query: {
      op: 'ASK_PROB',
      target: prog.query.target,
      negated: !!prog.query.negated,
      given: prog.query.given.map((g) => ({ ev: g.ev, pol: g.pol })),
      ctx: prog.query.ctx,
    },
  };
}

export function fromJSON(j) {
  const de = (d) => {
    const o = { ...d };
    if (o.value !== undefined) o.value = rat(o.value);
    return o;
  };
  const prog = {
    decls: j.decls.map(de),
    query: {
      op: 'ASK_PROB',
      target: j.query.target,
      negated: !!j.query.negated,
      given: j.query.given.map((g) => ({ ev: g.ev, pol: g.pol ?? 'POS' })),
      ctx: j.query.ctx,
    },
  };
  return validateProgram(prog);
}

const spanStr = (sp) => `[${sp.s},${sp.e}]`;

export function toBytecode(prog) {
  const lines = [];
  for (const d of prog.decls) {
    switch (d.op) {
      case 'HYP':
        lines.push(`HYP ${d.id} ${spanStr(d.span)}`);
        break;
      case 'PART_EXHAUSTIVE':
        lines.push(`PART_EXHAUSTIVE (${d.hyps.join(' ')}) ${spanStr(d.span)}`);
        break;
      case 'EVENT':
        lines.push(`EVENT ${d.id} ${spanStr(d.span)}`);
        break;
      case 'PRIOR':
        lines.push(`PRIOR ${d.hyp} ${R.toString(d.value)} ${d.ctx} ${spanStr(d.span)}`);
        break;
      case 'LIK':
        lines.push(`LIK ${d.ev} ${d.pol} GIVEN ${d.hyp} ${R.toString(d.value)} ${d.ctx} ${spanStr(d.span)}`);
        break;
      case 'POSTERIOR_STATED':
        lines.push(`POSTERIOR_STATED ${d.hyp} GIVEN ${d.ev} ${d.pol} ${R.toString(d.value)} ${d.ctx} ${spanStr(d.span)}`);
        break;
      case 'JOINT_LIK':
        lines.push(
          `JOINT_LIK (${d.spec.map((s) => `${s.ev}:${s.pol}`).join(' ')}) GIVEN ${d.hyp} ${R.toString(d.value)} ${d.ctx} ${spanStr(d.span)}`
        );
        break;
      case 'DET_LINK':
        lines.push(`DET_LINK ${d.a} ${d.rel} ${d.b} ${d.ctx} ${spanStr(d.span)}`);
        break;
      case 'SENT_TAG':
        lines.push(`SENT_TAG ${d.idx} ${d.tag}${d.ctx ? ` ${d.ctx}` : ''} ${spanStr(d.span)}`);
        break;
      case 'OBS':
        lines.push(`OBS ${d.ev} ${d.pol} ${d.ctx} ${spanStr(d.span)}`);
        break;
      case 'COND_INDEP':
        lines.push(`COND_INDEP (${d.evs.join(' ')}) GIVEN ${d.hyp} ${d.ctx} ${spanStr(d.span)}`);
        break;
      case 'CONFLICT':
        lines.push(`CONFLICT ${spanStr(d.spanA)} ${spanStr(d.spanB)}`);
        break;
      case 'NON_ASSERTED':
        lines.push(`NON_ASSERTED ${spanStr(d.span)} ${d.reason}`);
        break;
      default:
        throw new Error(`cannot serialize op ${d.op}`);
    }
  }
  const q = prog.query;
  const giv = q.given.map((g) => `${g.ev}:${g.pol}`).join(' ');
  lines.push(`ASK_PROB ${q.negated ? 'NOT ' : ''}${q.target} GIVEN (${giv}) ${q.ctx}`);
  lines.push('END');
  return lines.join('\n');
}

// Accept either bytecode text, an AST, or a plain JSON object.
export function toProgram(input, textLen = null) {
  let prog;
  if (typeof input === 'string') prog = parseBytecode(input);
  else if (input && Array.isArray(input.decls) && input.query) {
    prog = input.decls.some((d) => typeof d.value === 'string') ? fromJSON(input) : validateProgram({ ...input });
  } else {
    throw new ParseError('unrecognized program input');
  }
  if (textLen !== null) validateProgram(prog, textLen);
  return prog;
}
