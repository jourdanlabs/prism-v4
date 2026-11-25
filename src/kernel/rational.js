// prism-kernel :: exact BigInt GCD-normalized rational arithmetic.
// Invariants: denominator > 0, gcd(|n|, d) === 1n. Zero floats anywhere.

export function gcd(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b !== 0n) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

export function make(n, d = 1n) {
  n = BigInt(n);
  d = BigInt(d);
  if (d === 0n) throw new Error('RATIONAL_ZERO_DENOMINATOR');
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  if (n === 0n) return { n: 0n, d: 1n };
  if (d === 1n) return { n, d }; // denominator 1: gcd is already 1 — skip it
  const g = gcd(n, d);
  return { n: n / g, d: d / g };
}

export const zero = () => make(0n, 1n);
export const one = () => make(1n, 1n);

export function fromInt(k) {
  return make(BigInt(k), 1n);
}

export function parse(str) {
  if (typeof str !== 'string') throw new Error('RATIONAL_PARSE: not a string');
  const s = str.trim();
  const m = /^(-?\d+)\s*\/\s*(\d+)$/.exec(s);
  if (m) {
    const d = BigInt(m[2]);
    if (d === 0n) throw new Error('RATIONAL_PARSE: zero denominator');
    return make(BigInt(m[1]), d);
  }
  if (/^-?\d+$/.test(s)) return make(BigInt(s), 1n);
  throw new Error(`RATIONAL_PARSE: cannot parse "${str}"`);
}

// Decimal string like "0.625" -> exact rational (no float involved).
export function parseDecimal(str) {
  const s = String(str).trim();
  const m = /^(-?)(\d*)\.(\d+)$/.exec(s);
  if (!m) throw new Error(`RATIONAL_PARSE_DECIMAL: "${str}"`);
  const sign = m[1] === '-' ? -1n : 1n;
  const intPart = m[2] === '' ? 0n : BigInt(m[2]);
  const frac = m[3];
  const den = 10n ** BigInt(frac.length);
  return make(sign * (intPart * den + BigInt(frac)), den);
}

export const neg = (a) => make(-a.n, a.d);
export const add = (a, b) => make(a.n * b.d + b.n * a.d, a.d * b.d);
export const sub = (a, b) => make(a.n * b.d - b.n * a.d, a.d * b.d);
export const mul = (a, b) => make(a.n * b.n, a.d * b.d);
export function div(a, b) {
  if (b.n === 0n) throw new Error('RATIONAL_DIV_ZERO');
  return make(a.n * b.d, a.d * b.n);
}
export const inv = (a) => div(one(), a);

export const cmp = (a, b) => {
  const lhs = a.n * b.d;
  const rhs = b.n * a.d;
  return lhs < rhs ? -1 : lhs > rhs ? 1 : 0;
};
export const eq = (a, b) => a.n === b.n && a.d === b.d;
export const isZero = (a) => a.n === 0n;
export const isOne = (a) => a.n === a.d;

export const min = (a, b) => (cmp(a, b) <= 0 ? a : b);
export const max = (a, b) => (cmp(a, b) >= 0 ? a : b);
export const abs = (a) => (a.n < 0n ? neg(a) : a);

export function sum(list) {
  return list.reduce((acc, r) => add(acc, r), zero());
}

export const toString = (a) => `${a.n}/${a.d}`;

// Lossy float view — ONLY for test cross-checks and reporting, never kernel math.
export const toNumber = (a) => Number(a.n) / Number(a.d);
