# PRISM Stage-1 dataset handoff

Lane P3 ships an offline generator, a validator and unit tests. The default corpus has 5,000 synthetic passages across medical diagnostics, industrial and energy sensors, supply-chain logistics, finance and audit, legal evidence, and abstract urns and panels. Nine reasoning families and 29 scenario variants cover all eight required instructions in accepted bytecode, including two-hop implications, joint likelihood tables, conditional independence and directly stated posteriors.

## Generate and verify

Use Node 20 or later from the repository root. The generator creates a new directory under an existing parent. It accepts a seed of 1–128 characters and a row count of at least 432 so every run covers the required families and variants.

```sh
node scripts/generate-stage1-dataset.mjs --out stage1-default
node scripts/validate-stage1-dataset.mjs --dir stage1-default

node scripts/generate-stage1-dataset.mjs --out stage1-smoke --seed 20261123 --count 500
node scripts/validate-stage1-dataset.mjs --dir stage1-smoke

PRISM_OLLAMA_HOST=http://127.0.0.1:1 npm test
```

Generation and validation use the local JS kernel. The test command directs the two optional model integration tests to an inactive loopback endpoint so the rest of the suite runs offline.

## Output and checks

- `train.jsonl`, `val.jsonl` and `eval.jsonl` each contain chat records with `system`, `user` and `assistant` messages. The user message contains the passage. The assistant message contains a JSON answer with bytecode and `interval: [min, max]`, or a refusal with its exact code and bounds when the kernel reports them. Endpoints are canonical rational strings such as `"3/10"`.
- The `prism.stage1.v2` schema records the family hash, domain, scenario variant, template instance, split, row index and kernel receipt hash. Refusals carry the candidate program in metadata for verification. Some candidates deliberately assert an invalid license. Feed the `messages` field to the trainer and keep the metadata for audit.
- Each reasoning family has six domain-specific template families. Sorting those six SHA-256 family IDs assigns four to train, one to validation and one to eval. Each family stays with its domain and all its counterfactual variants across seeds. Every split contains each refusal code and each answer mode, including joint likelihood. General clause forms and vocabulary recur across families.
- The generator derives gold answers independently with exact arithmetic and checks the kernel's output. For the unlicensed pairs, it derives the Fréchet endpoints from the marginals and checks both bounds. The validator reconstructs each row from its seed and index, verifies source spans and covered numeric literals, replays the text and bytecode through the kernel, and compares each serialized record with the verified result. It rejects missing per-split coverage, duplicate passages, and leakage of normalized number-stripped skeletons or template instances. The manifest records those checks and binds the exact split JSONL bytes.
- Negated membership tests source-level context exclusion even when program context IDs coincide. Historical wiring and embedded restrictions both exercise license rejection and return `UNLICENSED_INDEPENDENCE`. Impossible evidence returns `DEGENERATE_PARTITION`, and evidence whose support can vanish returns `ZERO_SUPPORT_POSSIBLE`. The suite also covers conflicting figures, missing base rates, duplicate observations, quoted and retracted premises, hypothetical and approximate premises, reversed conditionals, and incomplete factors.

## Verification on seed 20261123

| Split | Rows | Answers | Refusals |
| --- | ---: | ---: | ---: |
| Train | 3,333 | 1,152 | 2,181 |
| Validation | 835 | 287 | 548 |
| Held-out eval | 832 | 285 | 547 |
| Total | 5,000 | 1,724 | 3,276 |

The full corpus passed replay with all 10 refusal codes and all eight answer modes in each split. Across the three splits, zero passages, zero number-stripped skeletons, and zero template instances overlap. The corpus contains 5,000 distinct passages, 4,999 distinct normalized skeletons, and 174 distinct template instances. Regression tests cover deterministic replay, exact bounds, refusal relabeling after a checksum update, a row moved into the held-out split, absent per-split coverage, and reused skeletons and template instances.

Generated corpora are reproducible build outputs. The measured coverage applies to these synthetic families and the kernel on this branch. Model training and held-out model scoring are future work.
