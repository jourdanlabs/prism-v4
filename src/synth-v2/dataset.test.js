import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { R, sha256hex } from '../kernel/index.js';
import {
  ANSWER_MODES, REFUSAL_CODES, SPLITS, CorpusAudit, buildRow, passageSkeleton,
  planRow, scenarioOrder, verifyScenario,
} from './dataset.js';
import { parseDatasetLiteral } from './scenarios.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const seed = 'stage1-test';

function command(action, ...args) {
  return spawnSync(process.execPath, [`scripts/${action}-stage1-dataset.mjs`, ...args], {
    cwd: root, encoding: 'utf8',
  });
}

test('odds and success-to-failure ratios retain exact probability semantics', () => {
  for (const text of ['odds of 3 to 7 in favor', 'a 3:7 ratio of successes to failures', '30%', '3/10']) {
    assert.equal(R.toString(parseDatasetLiteral(text)), '3/10');
  }
});

test('changing the seed changes passages but never moves a family between splits', () => {
  const other = 'different-stage1-seed';
  const order = scenarioOrder(seed);
  const otherOrder = scenarioOrder(other);
  for (let i = 0; i < order.length; i++) {
    const { family } = planRow(order, seed, i);
    assert.equal(family.hash, sha256hex(family.name));
    assert.equal(otherOrder.find((slot) => slot.family.name === family.name).family.split, family.split);
  }
  assert.deepEqual(scenarioOrder(seed), order);
  assert.notDeepEqual(otherOrder, order);
  const row = buildRow(planRow(order, seed, 0)).row;
  assert.deepEqual(buildRow(planRow(order, seed, 0)).row, row);
  const sameSlot = otherOrder.findIndex((slot) =>
    slot.family.name === row.metadata.family && slot.domain === row.metadata.domain);
  assert.notEqual(buildRow(planRow(otherOrder, other, sameSlot)).row.messages[1].content, row.messages[1].content);
});

test('source text, refusal reason and exact Fréchet endpoints are verified', () => {
  const order = scenarioOrder(seed);
  const fixtures = new Map();
  for (let i = 0; i < 432 && fixtures.size < 3; i++) {
    const plan = planRow(order, seed, i);
    if (!['negated-membership', 'embedded-qualifier', 'tense-mismatch'].includes(plan.variant) ||
      fixtures.has(plan.variant) || plan.domain !== 'industrial') continue;
    fixtures.set(plan.variant, buildRow(plan).row);
  }
  assert.equal(fixtures.size, 3);
  for (const [variant, row] of fixtures) {
    const passage = row.messages[1].content;
    const bytecode = row.metadata.candidate_bytecode;
    const expected = JSON.parse(row.messages[2].content);
    assert.equal(expected.status, 'REFUSED');
    assert.equal(expected.bytecode, undefined);
    assert.match(bytecode, /^OBS /m);
    const { decision } = verifyScenario({ passage, bytecode, expected });
    if (variant === 'negated-membership') {
      assert.match(passage, /is not a night-shift panel/);
      assert.equal(decision.code, 'CONTEXT_MISMATCH');
      assert.match(decision.details.detail, /explicitly denied/);
    } else {
      assert.equal(decision.code, 'UNLICENSED_INDEPENDENCE');
      assert.ok(R.cmp(R.parse(expected.interval[0]), R.parse(expected.interval[1])) < 0);
      assert.throws(() => verifyScenario({
        passage, bytecode, expected: { ...expected, interval: ['0/1', '1/1'] },
      }), /Expected /);
    }
  }
});

test('CLI validates all rows and rejects relabeling and split leakage', () => {
  const temp = mkdtempSync(join(tmpdir(), 'prism-stage1-test-'));
  const dir = join(temp, 'dataset');
  try {
    const generated = command('generate', '--out', dir, '--seed', seed, '--count', '432');
    assert.equal(generated.status, 0, generated.stderr);
    const verified = command('validate', '--dir', dir);
    assert.equal(verified.status, 0, verified.stderr);
    const report = JSON.parse(verified.stdout);
    assert.equal(report.verified, 432);
    assert.equal(report.split_audit.skeletons_crossing_splits, 0);
    assert.equal(report.split_audit.template_instances_crossing_splits, 0);
    assert.equal(report.split_audit.passages_crossing_splits, 0);
    const owners = { passage: new Map(), skeleton: new Map(), template: new Map(), family: new Map() };
    for (const split of SPLITS) {
      const rows = readFileSync(join(dir, `${split}.jsonl`), 'utf8').trimEnd().split('\n').map(JSON.parse);
      const codes = new Set();
      for (const row of rows) {
        const passage = row.messages[1].content;
        const label = JSON.parse(row.messages[2].content);
        if (label.status === 'REFUSED') codes.add(label.code);
        for (const [kind, value] of Object.entries({
          passage,
          skeleton: passageSkeleton(passage),
          template: row.metadata.template_instance,
          family: row.metadata.family,
        })) {
          const prior = owners[kind].get(value);
          assert.ok(prior === undefined || prior === split, `${kind} overlaps ${prior} and ${split}`);
          owners[kind].set(value, split);
        }
      }
      assert.deepEqual([...codes].sort(), REFUSAL_CODES);
      assert.deepEqual(Object.keys(report.splits[split].answer_modes).sort(), ANSWER_MODES);
    }

    const train = join(dir, 'train.jsonl');
    const manifestPath = join(dir, 'manifest.json');
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    const original = readFileSync(train, 'utf8');
    const rows = original.trimEnd().split('\n').map(JSON.parse);
    const refused = rows.find((row) => {
      const label = JSON.parse(row.messages[2].content);
      return label.status === 'REFUSED' && label.code !== 'DUPLICATE_EVIDENCE';
    });
    const oldLabel = JSON.parse(refused.messages[2].content);
    refused.messages[2].content = JSON.stringify({ ...oldLabel, code: 'DUPLICATE_EVIDENCE' });
    const lines = rows.map((row) => JSON.stringify(row)).join('\n') + '\n';
    writeFileSync(train, lines);
    manifest.splits.train.sha256 = sha256hex(lines);
    writeFileSync(manifestPath, JSON.stringify(manifest));
    const invalid = command('validate', '--dir', dir);
    assert.equal(invalid.status, 1);
    assert.match(invalid.stderr, /Row differs from the verified passage, bytecode, verdict, or receipt/);

    const [first, ...rest] = original.trimEnd().split('\n');
    writeFileSync(train, rest.join('\n') + '\n');
    const heldOut = join(dir, 'eval.jsonl');
    const misplaced = JSON.parse(first);
    misplaced.metadata.split = 'eval';
    const evalRows = readFileSync(heldOut, 'utf8').trimEnd().split('\n').map(JSON.parse);
    evalRows.push(misplaced);
    evalRows.sort((a, b) => a.metadata.index - b.metadata.index);
    writeFileSync(heldOut, evalRows.map((row) => JSON.stringify(row)).join('\n') + '\n');
    const leaked = command('validate', '--dir', dir);
    assert.equal(leaked.status, 1);
    assert.match(leaked.stderr, /Scenario family is in the wrong split/);
  } finally {
    rmSync(temp, { recursive: true });
  }
});

test('audit rejects both number-only skeleton changes and a reused template instance across splits', () => {
  const result = buildRow(planRow(scenarioOrder(seed), seed, 0));
  const audit = new CorpusAudit();
  const add = (entry) => audit.add(entry, JSON.stringify(entry.row));
  const { row } = result;
  const otherSplit = SPLITS.find((split) => split !== row.metadata.split);
  const duplicate = structuredClone(result);
  duplicate.row.metadata = {
    ...row.metadata, split: otherSplit, family_sha256: 'another-family', template_instance: 'another-template',
  };
  duplicate.row.messages[1].content = row.messages[1].content.replace(/\d/g, (digit) => digit === '9' ? '8' : '9');
  assert.notEqual(duplicate.row.messages[1].content, row.messages[1].content);
  add(result);
  assert.throws(() => add(duplicate), /Number-stripped skeleton crosses splits/);
  duplicate.row.metadata.template_instance = row.metadata.template_instance;
  assert.throws(() => add(duplicate), /Template instance crosses splits/);
  assert.equal(passageSkeleton('25% of records'), passageSkeleton('odds of 1 to 2 in favor of records'));
});

test('audit requires each split to have every refusal code and answer mode', () => {
  const order = scenarioOrder(seed);
  const cases = order.map((_, i) => buildRow(planRow(order, seed, i)));
  for (const [variant, message] of [
    ['duplicate-evidence', /val lacks required refusal codes: DUPLICATE_EVIDENCE/],
    ['exclusive-signals', /val lacks required answer modes: MUTEX/],
  ]) {
    const audit = new CorpusAudit();
    const kept = cases.filter(({ row }) => !(row.metadata.split === 'val' && row.metadata.variant === variant));
    for (const result of kept) audit.add(result, JSON.stringify(result.row));
    assert.throws(() => audit.manifest({ seed, count: kept.length }), message);
  }
});
