// synth-bench :: rational -> literal surface forms.
// Every emitted form must round-trip through kernel/literals.parseLiteral.

const NUM = {
  1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven',
  8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
  14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
  19: 'nineteen', 20: 'twenty',
};
const DEN = {
  2: ['half', 'halves'], 3: ['third', 'thirds'], 4: ['fourth', 'fourths'],
  5: ['fifth', 'fifths'], 6: ['sixth', 'sixths'], 7: ['seventh', 'sevenths'],
  8: ['eighth', 'eighths'], 9: ['ninth', 'ninths'], 10: ['tenth', 'tenths'],
  11: ['eleventh', 'elevenths'], 12: ['twelfth', 'twelfths'], 13: ['thirteenth', 'thirteenths'],
  14: ['fourteenth', 'fourteenths'], 15: ['fifteenth', 'fifteenths'],
  16: ['sixteenth', 'sixteenths'], 17: ['seventeenth', 'seventeenths'],
  18: ['eighteenth', 'eighteenths'], 19: ['nineteenth', 'nineteenths'],
  20: ['twentieth', 'twentieths'],
};

export function toWords(n, d) {
  if (!NUM[Number(n)] || !DEN[Number(d)]) return null;
  const forms = DEN[Number(d)];
  if (n === 1n) return `one ${forms[0]}`;
  return `${NUM[Number(n)]} ${forms[1]}`;
}

// exact decimal iff denominator's prime factors ⊆ {2,5}
function terminates(d) {
  let x = Number(d);
  while (x % 2 === 0) x /= 2;
  while (x % 5 === 0) x /= 5;
  return x === 1;
}

// Exact decimal expansion via long division (caller guarantees it terminates).
function decimalString(num0, den0) {
  let num = Number(num0);
  const den = Number(den0);
  const int = Math.floor(num / den);
  num %= den;
  if (num === 0) return `${int}`;
  let digits = '';
  while (num % den !== 0 && digits.length < 16) {
    num *= 10;
    digits += Math.floor(num / den);
    num %= den;
  }
  digits = digits.replace(/0+$/, '');
  return digits === '' ? `${int}` : `${int}.${digits}`;
}

/** All surface forms a rational can take, tagged by kind. */
export function literalForms(r) {
  const { n, d } = r;
  const forms = [{ kind: 'fraction', text: `${n}/${d}` }];
  if (n >= 1n && n <= 20n && DEN[Number(d)]) {
    forms.push({ kind: 'words', text: toWords(n, d) });
  }
  if (terminates(d)) {
    forms.push({ kind: 'decimal', text: decimalString(n, d) });
    // percent = n*100/d — exact because d's factors ⊆ {2,5}
    forms.push({ kind: 'percent', text: `${decimalString(n * 100n, d)}%` });
  }
  if (d <= 20n && n <= 20n) {
    forms.push({ kind: 'inN', text: `${n} in ${d}` });
  }
  return forms;
}
