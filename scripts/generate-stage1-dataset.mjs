import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import {
  buildRow, checkOptions, CorpusAudit, DEFAULT_COUNT, DEFAULT_SEED, planRow, scenarioOrder, SPLITS,
} from '../src/synth-v2/dataset.js';

async function main() {
  const { values: args } = parseArgs({
    options: {
      out: { type: 'string' },
      seed: { type: 'string', default: DEFAULT_SEED },
      count: { type: 'string', default: String(DEFAULT_COUNT) },
      help: { type: 'boolean' },
    },
  });
  if (args.help) {
    console.log('Usage: node scripts/generate-stage1-dataset.mjs --out NEW_DIRECTORY [--seed SEED] [--count 5000]');
    return;
  }
  if (!args.out) throw new Error('--out must name a new output directory.');
  const options = { seed: args.seed, count: Number(args.count) };
  checkOptions(options);
  const order = scenarioOrder(options.seed);
  const audit = new CorpusAudit();
  const lines = Object.fromEntries(SPLITS.map((split) => [split, []]));
  for (let index = 0; index < options.count; index++) {
    const plan = planRow(order, options.seed, index);
    try {
      const result = buildRow(plan);
      const line = JSON.stringify(result.row) + '\n';
      audit.add(result, line);
      lines[plan.family.split].push(line);
    } catch (error) {
      throw new Error(`Row ${index} (${plan.domain}/${plan.variant}): ${error.message}`, { cause: error });
    }
  }
  const manifest = audit.manifest(options);
  await mkdir(args.out);
  for (const split of SPLITS) {
    await writeFile(join(args.out, `${split}.jsonl`), lines[split].join(''), { flag: 'wx' });
  }
  await writeFile(join(args.out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ out: args.out, count: manifest.count, splits: manifest.splits }));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
