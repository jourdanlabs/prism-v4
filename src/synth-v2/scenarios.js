import { R, parseLiteral, toProgram, toBytecode } from '../kernel/index.js';
import { autoSentTags } from '../kernel/sentences.js';
import { DOMAINS } from './domains.js';

const ZERO = R.zero();
const ONE = R.one();
const complement = (p) => R.sub(ONE, p);
const percent = (n) => R.make(BigInt(n), 100n);
const interval = (min, max = min) => [R.toString(min), R.toString(max)];
const answer = (p) => ({ status: 'ANSWER', interval: interval(p) });
const refuse = (code, bounds) => bounds
  ? { status: 'REFUSED', code, interval: bounds }
  : { status: 'REFUSED', code };

function posterior(prior, [left, right]) {
  const a = R.mul(prior, left);
  return R.div(a, R.add(a, R.mul(complement(prior), right)));
}

function frechet(prior, [a, b], [c, d]) {
  const lower = (x, y) => R.sub(R.add(x, y), ONE);
  const upper = (x, y) => R.cmp(x, y) < 0 ? x : y;
  return interval(posterior(prior, [lower(a, c), upper(b, d)]),
    posterior(prior, [upper(a, c), lower(b, d)]));
}

function literal(value, rng) {
  const { n, d } = value;
  const forms = [`${n}/${d}`, `${n} out of ${d}`];
  if (100n % d === 0n) forms.push(`${100n * n / d}%`);
  if (n < d) forms.push(`odds of ${n} to ${d - n} in favor`, `a ${n}:${d - n} ratio of successes to failures`);
  return rng.pick(forms);
}

export function parseDatasetLiteral(text) {
  const odds = /^odds of (\d+) to (\d+) in favor$/.exec(text);
  const ratio = /^a (\d+):(\d+) ratio of successes to failures$/.exec(text);
  const match = odds ?? ratio;
  if (!match) return parseLiteral(text);
  const n = BigInt(match[1]);
  const m = BigInt(match[2]);
  return R.make(n, n + m);
}

class Passage {
  text = '';
  decls = [];

  sentence(text) {
    if (this.text) this.text += ' ';
    const start = this.text.length;
    this.text += text;
    return { start, text };
  }

  declare(instruction, source, excerpt = source.text, suffix = '') {
    const offset = source.text.indexOf(excerpt);
    if (offset < 0) throw new Error(`Missing source excerpt: ${excerpt}`);
    const start = source.start + offset;
    this.decls.push(`${instruction} [${start},${start + excerpt.length}]${suffix}`);
  }

  numeric(instruction, value, surface, source, ctx) {
    this.declare(`${instruction} ${R.toString(value)} ${ctx}`, source, surface);
  }

  bytecode(query) {
    const program = toProgram([...this.decls, query, 'END'].join('\n'), this.text.length);
    program.decls.push(...autoSentTags(this.text, program));
    return toBytecode(program);
  }
}

// Variants of a reasoning template share a family. The scheduler holds out
// whole family/domain combinations, keeping counterfactual siblings together.
export const FAMILIES = {
  'single-marker': [
    'bayes-positive', 'bayes-negative', 'conflicting-figures', 'duplicate-evidence',
    'missing-base-rate', 'zero-support', 'quoted-premise', 'retracted-premise',
    'hypothetical-premise', 'hedged-premise',
  ],
  'joint-table': ['joint-likelihood', 'conflicting-joint'],
  'shared-wiring': ['equivalent-signals', 'tense-mismatch', 'embedded-qualifier'],
  'graded-evidence': ['conditional-independence', 'unlicensed-independence'],
  'population-transfer': ['matched-context', 'context-mismatch', 'negated-membership'],
  'published-posterior': ['stated-posterior', 'reversed-conditional', 'posterior-conflict'],
  'ordered-signals': ['two-hop-implication'],
  'weighted-marginal': ['event-marginal', 'incomplete-factorization'],
  'mutually-exclusive': ['exclusive-signals', 'zero-joint-support', 'possible-zero-support'],
};

const NON_ASSERTED = {
  'quoted-premise': ['QUOTED', 'An unverified quotation gives'],
  'retracted-premise': ['RETRACTED', 'A withdrawn report formerly gave'],
  'hypothetical-premise': ['HYPOTHETICAL', 'A hypothetical example uses'],
  'hedged-premise': ['HEDGED_APPROX', 'A rough guess uses'],
};

export function renderScenario({ domain, variant, rng }) {
  const [item, h0, h1, a, b, c, aside] = rng.pick(DOMAINS[domain]);
  const article = /^[aeiou]/.test(item) ? 'an' : 'a';
  const names = [h0, h1];
  const p = percent(rng.int(16, 84));
  let first = [percent(rng.int(60, 87)), percent(rng.int(54, 75))];
  const second = [percent(rng.int(60, 87)), percent(rng.int(54, 75))];
  const joint = [percent(rng.int(45, 53)), percent(rng.int(28, 39))];
  const P = new Passage();
  const ctx = 'sample';
  let queryCtx = ctx;
  let target = 'h0';
  let targetText = `the ${item} is classified as ${h0}`;
  const givens = [];
  let expected;
  const partition = P.sentence(rng.pick([
    `Every ${item} is classified as either ${h0} or ${h1}, exclusively.`,
    `The only categories for ${article} ${item} are ${h0} and ${h1}; every ${item} belongs to exactly one.`,
    `Each ${item} belongs to exactly one category, either ${h0} or ${h1}.`,
  ]));
  P.declare('HYP h0', partition, h0);
  P.declare('HYP h1', partition, h1);
  P.declare('PART_EXHAUSTIVE (h0 h1)', partition);

  function priors({ scope = '', at = ctx, nonAsserted = null } = {}) {
    const values = [p, complement(p)];
    for (const i of rng.shuffle([0, 1])) {
      const surface = literal(values[i], rng);
      const wording = nonAsserted
        ? `${nonAsserted[1]} ${surface} for the share of ${names[i]} among the ${h0} and ${h1} categories.`
        : rng.pick([
          `Across the ${h0} and ${h1} categories, the share for ${names[i]} is ${surface}.`,
          `The register gives ${names[i]} a share of ${surface} in the ${h0} and ${h1} population.`,
        ]);
      const sentence = P.sentence(scope ? `${scope}, ${wording[0].toLowerCase()}${wording.slice(1)}` : wording);
      P.numeric(`PRIOR h${i}`, values[i], surface, sentence, at);
      if (nonAsserted) P.declare('NON_ASSERTED', sentence, sentence.text, ` ${nonAsserted[0]}`);
    }
  }

  function likelihood(i, value, name = a, id = 'e0', scope = '', at = ctx) {
    const surface = literal(value, rng);
    const wording = rng.pick([
      `Given category ${names[i]}, the ${name} reads positive at a rate expressed as ${surface}.`,
      `For ${names[i]}, the ${name} is recorded as positive at a rate expressed as ${surface}.`,
      `The ${name} is positive at a rate expressed as ${surface} among ${names[i]} cases.`,
    ]);
    const sentence = P.sentence(scope ? `${scope}${wording[0].toLowerCase()}${wording.slice(1)}` : wording);
    P.numeric(`LIK ${id} POS GIVEN h${i}`, value, surface, sentence, at);
  }

  function likelihoods(values, name = a, id = 'e0', scope = '', at = ctx) {
    for (const i of rng.shuffle([0, 1])) likelihood(i, values[i], name, id, scope, at);
  }

  function events(...eventNames) {
    const last = eventNames.at(-1);
    const earlier = eventNames.slice(0, -1).map((name) => `the ${name}`);
    const list = earlier.length ? `${earlier.join(', ')} and the ${last}` : `the ${last}`;
    const sentence = P.sentence(`The record tracks ${list}.`);
    eventNames.forEach((name, i) => P.declare(`EVENT e${i}`, sentence, name));
  }

  function observe(name = a, id = 'e0', { pol = 'POS', scope = '', at = queryCtx, repeat = false } = {}) {
    const reading = pol === 'POS' ? 'positive' : 'negative';
    const sentence = P.sentence(scope || rng.pick([
      `The ${name} reads ${reading} on the selected ${item}.`,
      `A ${reading} reading was recorded for the ${name} on the selected ${item}.`,
      `On the selected ${item}, the ${name} is currently observed to be ${reading}.`,
    ]));
    P.declare(`OBS ${id} ${pol} ${at}`, sentence);
    if (!repeat) givens.push(`${id}:${pol}`);
  }

  function link(text, aId, rel, bId) {
    P.declare(`DET_LINK ${aId} ${rel} ${bId} ${ctx}`, P.sentence(text));
  }

  P.sentence(aside);
  const decoyCount = rng.int(10, 99);
  P.sentence(rng.pick([
    `The binder has ${decoyCount} blank pages.`,
    `There are ${decoyCount} unused labels in the drawer.`,
    `The cover sheet took ${decoyCount} seconds to print.`,
  ]));

  if (['bayes-positive', 'bayes-negative', 'conflicting-figures', 'duplicate-evidence',
    'missing-base-rate', 'zero-support'].includes(variant) || Object.hasOwn(NON_ASSERTED, variant)) {
    events(a);
    if (variant !== 'missing-base-rate') priors({ nonAsserted: NON_ASSERTED[variant] });
    if (variant === 'zero-support') first = [ZERO, ZERO];
    likelihoods(first);
    const pol = variant === 'bayes-negative' ? 'NEG' : 'POS';
    observe(a, 'e0', { pol });
    switch (variant) {
      case 'conflicting-figures':
        likelihood(0, R.sub(first[0], percent(1)));
        expected = refuse('CONFLICTING_FIGURES');
        break;
      case 'duplicate-evidence':
        observe(a, 'e0', { repeat: true, scope: `The log repeats the same positive ${a} reading for that observation.` });
        expected = refuse('DUPLICATE_EVIDENCE');
        break;
      case 'missing-base-rate': expected = refuse('MISSING_BASE_RATE'); break;
      case 'zero-support': expected = refuse('DEGENERATE_PARTITION'); break;
      default:
        expected = NON_ASSERTED[variant]
          ? refuse('SCOPED_OR_RETRACTED_PREMISE')
          : answer(posterior(p, pol === 'NEG' ? first.map(complement) : first));
    }
  } else if (variant === 'joint-likelihood' || variant === 'conflicting-joint') {
    events(a, b);
    priors();
    function jointRow(i, value) {
      const surface = literal(value, rng);
      const sentence = P.sentence(`Within ${names[i]}, both the ${a} and the ${b} read positive together at a rate expressed as ${surface}.`);
      P.numeric(`JOINT_LIK (e0:POS e1:POS) GIVEN h${i}`, value, surface, sentence, ctx);
    }
    rng.shuffle([0, 1]).forEach((i) => jointRow(i, joint[i]));
    if (variant === 'conflicting-joint') jointRow(0, R.add(joint[0], percent(1)));
    observe();
    observe(b, 'e1');
    expected = variant === 'conflicting-joint' ? refuse('CONFLICTING_FIGURES') : answer(posterior(p, joint));
  } else if (['equivalent-signals', 'tense-mismatch', 'embedded-qualifier'].includes(variant)) {
    events(a, b);
    priors();
    likelihoods(first);
    likelihoods(first, b, 'e1');
    const subject = `The ${a} and the ${b}`;
    const text = {
      'equivalent-signals': rng.pick([
        `${subject} are wired to the same switch.`,
        'These two signals always activate together.',
        'Both readings always move in lockstep.',
      ]),
      'tense-mismatch': `${subject} were wired to the same switch.`,
      'embedded-qualifier': rng.pick([
        `Both the ${a} and the ${b} on damaged panels always activate together.`,
        `Both the ${a} and the ${b} inside the locked enclosure always activate together.`,
      ]),
    }[variant];
    link(text, 'e0', 'EQUIV', 'e1');
    observe(a, 'e0', { scope: `On the selected ${item}, the ${a} is currently observed to be positive.` });
    observe(b, 'e1', { scope: `The ${b} is also currently observed to be positive on that ${item}.` });
    expected = variant === 'equivalent-signals'
      ? answer(posterior(p, first))
      : refuse('UNLICENSED_INDEPENDENCE', frechet(p, first, first));
  } else if (variant === 'conditional-independence' || variant === 'unlicensed-independence') {
    events(a, b);
    priors();
    likelihoods(first);
    likelihoods(second, b, 'e1');
    const independence = variant === 'conditional-independence';
    if (independence) {
      const sentence = P.sentence(`Within each category, the ${a} and the ${b} are conditionally independent.`);
      P.declare(`COND_INDEP (e0 e1) GIVEN * ${ctx}`, sentence);
    }
    observe();
    observe(b, 'e1');
    expected = independence
      ? answer(posterior(p, first.map((value, i) => R.mul(value, second[i]))))
      : refuse('UNLICENSED_INDEPENDENCE', frechet(p, first, second));
  } else if (['matched-context', 'context-mismatch', 'negated-membership'].includes(variant)) {
    const night = 'ctx_night_shift';
    const day = 'ctx_day_shift';
    const evidenceNoun = domain === 'industrial' ? 'panel' : item;
    queryCtx = variant === 'context-mismatch' ? day : night;
    events(a);
    priors({ scope: 'On the night shift', at: night });
    likelihoods(first, a, 'e0', 'On the night shift, ', night);
    const membership = {
      'matched-context': `is a night-shift ${evidenceNoun}`,
      'context-mismatch': `is a day-shift ${evidenceNoun}`,
      'negated-membership': `is not a night-shift ${evidenceNoun}`,
    }[variant];
    observe(a, 'e0', { scope: `The selected ${item} ${membership}, and its ${a} reads positive.` });
    expected = variant === 'matched-context' ? answer(posterior(p, first)) : refuse('CONTEXT_MISMATCH');
  } else if (['stated-posterior', 'posterior-conflict', 'reversed-conditional'].includes(variant)) {
    if (variant === 'reversed-conditional') events(a, b);
    else events(a);
    const computed = posterior(p, first);
    if (variant !== 'reversed-conditional') {
      priors();
      if (variant === 'posterior-conflict') likelihoods(first);
      else likelihood(0, first[0]);
    }
    const value = {
      'posterior-conflict': R.div(computed, R.fromInt(2n)),
      'stated-posterior': computed,
      'reversed-conditional': p,
    }[variant];
    const surface = literal(value, rng);
    const sentence = P.sentence(`Among ${item}s with a positive ${a}, the reported share classified as ${h0} is ${surface}.`);
    P.numeric('POSTERIOR_STATED h0 GIVEN e0 POS', value, surface, sentence, ctx);
    observe();
    if (variant === 'reversed-conditional') {
      priors();
      likelihoods(second, b, 'e1');
      observe(b, 'e1');
      expected = refuse('REVERSED_CONDITIONAL');
    } else {
      expected = variant === 'posterior-conflict' ? refuse('CONFLICTING_FIGURES') : answer(computed);
    }
  } else if (variant === 'two-hop-implication') {
    events(a, b, c);
    priors();
    likelihoods(first);
    link(`Whenever the ${a} fires, the ${b} fires.`, 'e0', 'IMPLIES', 'e1');
    link(`If the ${b} fires, then the ${c} fires.`, 'e1', 'IMPLIES', 'e2');
    observe();
    target = 'e2';
    targetText = `the ${c} reads positive`;
    expected = answer(ONE);
  } else if (variant === 'event-marginal' || variant === 'incomplete-factorization') {
    events(a);
    priors();
    likelihood(0, first[0]);
    if (variant === 'event-marginal') likelihood(1, first[1]);
    target = 'e0';
    targetText = `the ${a} reads positive on a randomly selected ${item}`;
    expected = variant === 'event-marginal'
      ? answer(R.add(R.mul(p, first[0]), R.mul(complement(p), first[1])))
      : refuse('INCOMPLETE_FACTORIZATION');
  } else if (['exclusive-signals', 'zero-joint-support', 'possible-zero-support'].includes(variant)) {
    events(a, b);
    priors();
    const low = [percent(rng.int(8, 39)), percent(rng.int(8, 39))];
    likelihoods(low);
    likelihood(0, variant === 'possible-zero-support' ? ZERO : low[0], b, 'e1');
    if (variant !== 'possible-zero-support') likelihood(1, low[1], b, 'e1');
    link(`The ${a} and the ${b} are mutually exclusive.`, 'e0', 'MUTEX', 'e1');
    if (variant !== 'possible-zero-support') observe();
    if (variant === 'exclusive-signals') {
      target = 'e1';
      targetText = `the ${b} reads positive`;
      expected = answer(ZERO);
    } else if (variant === 'zero-joint-support') {
      observe(b, 'e1');
      expected = refuse('DEGENERATE_PARTITION');
    } else {
      observe(b, 'e1');
      expected = refuse('ZERO_SUPPORT_POSSIBLE');
    }
  } else {
    throw new Error(`Unknown scenario variant ${variant}`);
  }

  P.sentence(rng.pick([
    `What is the probability that ${targetText}?`,
    `How likely is it that ${targetText}?`,
    `What is the chance that ${targetText}?`,
  ]));
  return { passage: P.text, bytecode: P.bytecode(`ASK_PROB ${target} GIVEN (${givens.join(' ')}) ${queryCtx}`), expected };
}
