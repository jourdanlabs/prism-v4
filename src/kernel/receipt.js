// prism-kernel :: LUNA-compatible SHA-256 span-bound decision receipts.
// Every kernel decision (ANSWER or REFUSED) seals to a deterministic receipt:
// the hash binds the input text, the exact program, every premise span used,
// and the decision itself. No wall-clock, no RNG — the same (X, program)
// always produces the same receipt_id, so receipts are chainable in a ledger.

import { createHash } from 'node:crypto';
import * as R from './rational.js';
import { toJSON, toBytecode } from './parser.js';

export function sha256hex(data) {
  return createHash('sha256').update(data, 'utf8').digest('hex');
}

// Stable JSON stringify: object keys sorted recursively so hashing is canonical.
export function canonicalJSON(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJSON).join(',')}]`;
  const keys = Object.keys(value).sort();
  const body = keys.map((k) => `${JSON.stringify(k)}:${canonicalJSON(value[k])}`).join(',');
  return `{${body}}`;
}

function serDecl(d) {
  const o = { ...d };
  if (o.value !== undefined) o.value = R.toString(o.value);
  return o;
}

/**
 * buildReceipt({X, program, decision}) -> receipt object.
 * decision: result of solve(). Spans are bound both by offset and by the
 * sha256 of the exact covered excerpt, so a receipt cannot be replayed
 * against a shifted or edited text.
 */
export function buildReceipt({ X, program, decision }) {
  const progJSON = toJSON(program);
  const bytecode = toBytecode(program);
  const witnessDecls = decision.witnesses ?? [];

  const bindings = witnessDecls.map((d, i) => {
    const b = {
      index: i,
      op: d.op,
      decl: serDecl(d),
    };
    if (d.span) {
      const excerpt = X !== null ? X.slice(d.span.s, d.span.e) : null;
      b.span = [d.span.s, d.span.e];
      b.span_sha256 = sha256hex(`${d.span.s}:${d.span.e}`);
      if (excerpt !== null) b.excerpt_sha256 = sha256hex(excerpt);
    }
    return b;
  });

  const dec =
    decision.status === 'ANSWER'
      ? {
          status: 'ANSWER',
          target: decision.target,
          rational: R.toString(decision.value),
          via: decision.via ?? 'BAYES',
        }
      : {
          status: 'REFUSED',
          code: decision.code,
          details: canonDetails(decision.details),
        };

  const receipt = {
    schema: 'prism.decision_receipt.v1',
    kernel: 'prism-kernel/0.1.0',
    x_sha256: X !== null ? sha256hex(X) : null,
    program_sha256: sha256hex(bytecode),
    program: progJSON,
    bindings,
    decision: dec,
  };
  receipt.receipt_sha256 = sha256hex(canonicalJSON(receipt));
  return receipt;
}

function canonDetails(details) {
  if (!details || typeof details !== 'object') return details ?? null;
  const out = {};
  for (const [k, v] of Object.entries(details)) {
    if (v && typeof v === 'object' && 'n' in v && 'd' in v) {
      out[k] = R.toString(v);
    } else if (v && typeof v === 'object' && 's' in v && 'e' in v) {
      out[k] = [v.s, v.e];
    } else if (Array.isArray(v)) {
      out[k] = v.map((x) =>
        x && typeof x === 'object' && 's' in x && 'e' in x
          ? [x.s, x.e]
          : x && typeof x === 'object' && 'n' in x && 'd' in x
            ? R.toString(x)
            : x && typeof x === 'object'
              ? canonDetails(x)
              : x
      );
    } else {
      out[k] = v;
    }
  }
  return out;
}

// Verify a receipt against a candidate (X, program, decision): recomputes the
// seal. Returns true iff everything matches — the receipt is self-verifying.
export function verifyReceipt(receipt, { X, program, decision }) {
  const fresh = buildReceipt({ X, program, decision });
  return fresh.receipt_sha256 === receipt.receipt_sha256;
}
