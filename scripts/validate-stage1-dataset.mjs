import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { canonicalJSON } from '../src/kernel/index.js';
import {
  buildRow, checkOptions, CorpusAudit, planRow, scenarioOrder, SCHEMA, SPLITS,
} from '../src/synth-v2/dataset.js';

async function main() {
  const { values: args } = parseArgs({
    options: { dir: { type: 'string' }, help: { type: 'boolean' } },
  });
  if (args.help) {
    console.log('Usage: node scripts/validate-stage1-dataset.mjs --dir DATASET_DIRECTORY');
    return;
  }
  if (!args.dir) throw new Error('--dir must name the dataset directory.');
  const manifest = JSON.parse(await readFile(join(args.dir, 'manifest.json'), 'utf8'));
  if (manifest.schema !== SCHEMA) throw new Error('Unsupported dataset schema.');
  const { seed, count } = manifest;
  checkOptions({ seed, count });
  const order = scenarioOrder(seed);
  const audit = new CorpusAudit();
  const seen = new Set();

  for (const split of SPLITS) {
    const contents = await readFile(join(args.dir, `${split}.jsonl`), 'utf8');
    const lines = contents.split('\n');
    if (lines.pop() !== '') throw new Error(`${split}.jsonl must end with a newline.`);
    let previous = -1;
    for (const [i, line] of lines.entries()) {
      try {
        const input = JSON.parse(line);
        const index = input.metadata?.index;
        if (!Number.isSafeInteger(index) || index < 0 || index >= count ||
          index <= previous || seen.has(index)) {
          throw new Error('Row indices must be in range, ordered and unique.');
        }
        previous = index;
        seen.add(index);
        const plan = planRow(order, seed, index);
        if (plan.family.split !== split) throw new Error('Scenario family is in the wrong split.');
        const result = buildRow(plan);
        if (line !== JSON.stringify(result.row)) {
          throw new Error('Row differs from the verified passage, bytecode, verdict, or receipt.');
        }
        audit.add(result, line + '\n');
      } catch (error) {
        throw new Error(`${split}.jsonl:${i + 1}: ${error.message}`, { cause: error });
      }
    }
  }
  const actual = audit.manifest({ seed, count });
  if (canonicalJSON(actual) !== canonicalJSON(manifest)) {
    throw new Error('Manifest differs from the verified dataset.');
  }
  console.log(JSON.stringify({
    verified: actual.count,
    splits: actual.splits,
    refusal_codes: actual.refusal_codes,
    split_audit: actual.split_audit,
  }));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
