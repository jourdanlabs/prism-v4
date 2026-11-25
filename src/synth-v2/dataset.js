import { createHash } from 'node:crypto';
import { R, buildReceipt, canonicalJSON, sha256hex, solve, toProgram } from '../kernel/index.js';
import { makeRng } from '../synth/rng.js';
import { DOMAINS } from './domains.js';
import { FAMILIES, parseDatasetLiteral, renderScenario } from './scenarios.js';

export const SCHEMA = 'prism.stage1.v2';
export const SPLITS = ['train', 'val', 'eval'];
export const DEFAULT_SEED = '20261123';
export const DEFAULT_COUNT = 5000;
export const REFUSAL_CODES = [
  'CONFLICTING_FIGURES', 'CONTEXT_MISMATCH', 'DEGENERATE_PARTITION', 'DUPLICATE_EVIDENCE',
  'INCOMPLETE_FACTORIZATION', 'MISSING_BASE_RATE', 'REVERSED_CONDITIONAL',
  'SCOPED_OR_RETRACTED_PREMISE', 'UNLICENSED_INDEPENDENCE', 'ZERO_SUPPORT_POSSIBLE',
];
export const ANSWER_MODES = [
  'COND_INDEP', 'EQUIV', 'EVENT_MARGINAL', 'IMPLIES', 'JOINT_LIK', 'MUTEX',
  'SINGLE_MARKER_BAYES', 'STATED_POSTERIOR',
];
const SYSTEM = 'Extract PRISM bytecode and give the exact rational probability interval as JSON. If the evidence does not license an answer, return REFUSED and its code, with exact bounds when available.';

const domainNames = Object.keys(DOMAINS);
// Assign all wording and counterfactuals of one reasoning family in one
// domain together. Each reasoning family has domains in every split.
const familyPlan = Object.keys(FAMILIES).flatMap((reasoning) =>
  domainNames.map((domain) => {
    const name = `${reasoning}/${domain}`;
    return { name, hash: sha256hex(name), reasoning, domain };
  }).sort((a, b) => a.hash.localeCompare(b.hash))
    .map((family, i) => ({ ...family, split: i < 4 ? 'train' : i === 4 ? 'val' : 'eval' }))
);
const slots = familyPlan.flatMap((family) => FAMILIES[family.reasoning].map((variant) => ({
  family, domain: family.domain, variant,
})));
const MIN_COUNT = 8 * familyPlan.length;

export function checkOptions({ seed, count }) {
  if (typeof seed !== 'string' || seed.length < 1 || seed.length > 128) {
    throw new Error('Seed must contain 1–128 characters.');
  }
  if (!Number.isSafeInteger(count) || count < MIN_COUNT) {
    throw new Error(`Count must be an integer of at least ${MIN_COUNT} to guarantee corpus coverage.`);
  }
}

function seededRng(seed, purpose, index = 0) {
  const hash = sha256hex(JSON.stringify([SCHEMA, seed, purpose, index]));
  return makeRng(Number.parseInt(hash.slice(0, 8), 16));
}

export function scenarioOrder(seed) {
  return seededRng(seed, 'schedule').shuffle(slots);
}

export function planRow(order, seed, index) {
  return { index, seed, ...order[index % order.length] };
}

export function passageSkeleton(passage) {
  return passage.toLowerCase()
    .replace(/\b(?:odds of \d+ to \d+ in favor|a \d+:\d+ ratio of successes to failures|\d+ out of \d+|\d+\/\d+|\d+(?:\.\d+)?%|\d+\.\d+)(?!\w)/g, '#')
    .replace(/\d+/g, '#')
    .replace(/\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|percent)\b/g, '#')
    .replace(/\s+/g, ' ').trim();
}

function answerModes(program, ops) {
  const links = program.decls.filter((d) => d.op === 'DET_LINK').map((d) => d.rel);
  if (links.length) return [...new Set(links)];
  if (ops.includes('COND_INDEP')) return ['COND_INDEP'];
  if (ops.includes('JOINT_LIK')) return ['JOINT_LIK'];
  if (ops.includes('POSTERIOR_STATED')) return ['STATED_POSTERIOR'];
  return [program.query.targetKind === 'ev' ? 'EVENT_MARGINAL' : 'SINGLE_MARKER_BAYES'];
}

function exactBounds(bounds) {
  return ['min', 'max'].map((end) => {
    const value = bounds[end];
    return R.toString(typeof value === 'string' ? R.parse(value) : value);
  });
}

export function verifyScenario({ passage, bytecode, expected }) {
  const program = toProgram(bytecode, passage.length);
  for (const decl of program.decls) {
    if (decl.value === undefined) continue;
    const text = passage.slice(decl.span.s, decl.span.e);
    const value = parseDatasetLiteral(text);
    if (value === null || !R.eq(value, decl.value)) {
      throw new Error(`${decl.op} cannot bind ${R.toString(decl.value)} to "${text}"`);
    }
  }
  const decision = solve(program, passage);
  const actual = { status: decision.status };
  if (decision.status === 'ANSWER') {
    actual.interval = exactBounds(decision.interval);
    if (actual.interval.some((v) => v !== R.toString(decision.value))) {
      throw new Error('Answer and interval disagree.');
    }
  } else {
    actual.code = decision.code;
    if (decision.details?.interval) actual.interval = exactBounds(decision.details.interval);
  }
  if (canonicalJSON(actual) !== canonicalJSON(expected)) {
    throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)} (${decision.details?.detail ?? passage})`);
  }
  return { program, decision };
}

export function buildRow(plan) {
  const { index, seed, family, domain, variant } = plan;
  const scenario = renderScenario({ domain, variant, rng: seededRng(seed, 'passage', index) });
  const { passage, bytecode, expected } = scenario;
  const { program, decision } = verifyScenario(scenario);
  const refusal = expected.status === 'REFUSED';
  const label = refusal ? expected : { ...expected, bytecode };
  const row = {
    messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: passage },
      { role: 'assistant', content: JSON.stringify(label) },
    ],
    metadata: {
      index,
      split: family.split,
      domain,
      family: family.name,
      family_sha256: family.hash,
      template_instance: `${family.name}/${variant}`,
      variant,
      kernel_receipt_sha256: buildReceipt({ X: passage, program, decision }).receipt_sha256,
      ...(refusal && { candidate_bytecode: bytecode }),
    },
  };
  const ops = [...program.decls.map((d) => d.op), program.query.op];
  return { row, expected, ops, modes: refusal ? [] : answerModes(program, ops) };
}

export class CorpusAudit {
  splits = Object.fromEntries(SPLITS.map((split) => [split, {
    hash: createHash('sha256'), rows: 0, answers: 0, refusals: 0,
    refusal_codes: {}, answer_modes: {},
  }]));
  passages = new Set();
  variants = new Set();
  domains = new Set();
  answerOps = new Set();
  refusalCodes = {};
  familySplits = new Map();
  templateSplits = new Map();
  skeletonSplits = new Map();

  add(result, line) {
    const { row, expected, ops, modes } = result;
    const { domain, family_sha256: familyHash, split, variant, template_instance: template } = row.metadata;
    const passage = row.messages[1].content;
    const passageHash = sha256hex(passage);
    if (this.passages.has(passageHash)) throw new Error(`Duplicate passage at row ${row.metadata.index}`);
    const bindings = [
      [this.familySplits, familyHash, 'Family'],
      [this.templateSplits, template, 'Template instance'],
      [this.skeletonSplits, sha256hex(passageSkeleton(passage)), 'Number-stripped skeleton'],
    ];
    for (const [owners, key, label] of bindings) {
      const previous = owners.get(key);
      if (previous && previous !== split) throw new Error(`${label} crosses splits: ${key}`);
    }
    for (const [owners, key] of bindings) owners.set(key, split);
    this.passages.add(passageHash);
    this.variants.add(variant);
    this.domains.add(domain);
    const run = this.splits[split];
    run.rows++;
    run.hash.update(line);
    if (expected.status === 'ANSWER') {
      run.answers++;
      for (const op of ops) this.answerOps.add(op);
      for (const mode of modes) run.answer_modes[mode] = (run.answer_modes[mode] ?? 0) + 1;
    } else {
      run.refusals++;
      run.refusal_codes[expected.code] = (run.refusal_codes[expected.code] ?? 0) + 1;
      this.refusalCodes[expected.code] = (this.refusalCodes[expected.code] ?? 0) + 1;
    }
  }

  manifest({ seed, count }) {
    if (this.passages.size !== count) throw new Error('Dataset count mismatch.');
    for (const [split, stats] of Object.entries(this.splits)) {
      if (stats.answers === 0 || stats.refusals / stats.rows < 0.30) {
        throw new Error(`${split} requires answers and at least 30% refusals.`);
      }
      for (const [kind, required, observed] of [
        ['refusal codes', REFUSAL_CODES, stats.refusal_codes],
        ['answer modes', ANSWER_MODES, stats.answer_modes],
      ]) {
        const missing = required.filter((label) => !observed[label]);
        if (missing.length) throw new Error(`${split} lacks required ${kind}: ${missing.join(', ')}.`);
      }
    }
    const requiredOps = [
      'PART_EXHAUSTIVE', 'PRIOR', 'LIK', 'POSTERIOR_STATED',
      'JOINT_LIK', 'DET_LINK', 'COND_INDEP', 'ASK_PROB',
    ];
    if (this.domains.size !== domainNames.length ||
      this.variants.size !== Object.values(FAMILIES).flat().length ||
      this.templateSplits.size !== slots.length ||
      requiredOps.some((op) => !this.answerOps.has(op))) {
      throw new Error('Dataset is missing required scenario or instruction coverage.');
    }
    return {
      schema: SCHEMA, seed, count,
      domains: [...this.domains].sort(),
      families: familyPlan,
      variants: [...this.variants].sort(),
      answer_ops: [...this.answerOps].sort(),
      refusal_codes: this.refusalCodes,
      split_audit: {
        unique_passages: this.passages.size,
        unique_number_stripped_skeletons: this.skeletonSplits.size,
        unique_template_instances: this.templateSplits.size,
        passages_crossing_splits: 0,
        skeletons_crossing_splits: 0,
        template_instances_crossing_splits: 0,
      },
      splits: Object.fromEntries(SPLITS.map((split) => {
        const { hash, ...stats } = this.splits[split];
        return [split, { ...stats, sha256: hash.digest('hex') }];
      })),
    };
  }
}
