# PRISM v4 — Semantic-Witness Exact Probabilistic & Credal Decision Engine

**Forseti Research** · *Proof, or a refusal.*

PRISM v4 verifies probabilistic and decision-making claims from raw prose using a **fail-closed semantic cage**:

1. **Exact `BigInt` Rational & Credal Kernel:** Zero floating-point drift. Every posterior probability, interval bound, and utility check runs in exact integer/rational arithmetic.
2. **Verbatim Span Ownership & Direction Witnesses:** Every extracted prior, likelihood, and event literal must bind unambiguously to its grammatical owner and conditioning direction in the source text (`REVERSED_CONDITIONAL`, `INCOMPLETE_FACTORIZATION`, `UNLICENSED_INDEPENDENCE`, `SCOPED_OR_RETRACTED_PREMISE`).
3. **Counterfactual Invariance + Refusal-Only Cross-Model Witness Gate:** Candidate extractions are re-run under coherent literal swaps and independently checked by a second model family. A cross-model witness is strictly **refusal-only** (monotone downward) — it can only close a gate, never move an underdetermined premise to supported.
4. **Local Stdio MCP Server:** Run PRISM v4 locally as an MCP tool (`prism_verify_probability`, `prism_decide_from_belief`, `prism_decide_from_prose`).

## Evidence & Companion Repositories

- **Independent Benchmark (`ASSAY-013`):** [`donttrustme.ai/assay-013.html`](https://donttrustme.ai/assay-013.html) · `jourdanlabs/assay-013-prism-vs-jev-subq` (`MIT`)
  - Blind-100 unseen Bayesian & trap passages across 5 tolerance bands: local PRISM v4 scored **`40/50` exact**, **`0.0000` MAE**, and **`0/100` wrong releases** (`50/50` traps refused + `10` conservative refusals published openly in `misses.html`).
- **PRISM v5 (K-State Credal LP + Fréchet Anti-Cascade + NetVoI, `<2 ms` CPU):** `jourdanlabs/prism-v5` (`BUSL 1.1`)
- **Hosted PRISM Cloud API & Live Playground:** `https://forseti-prism-cloud.fly.dev/` (Commercial SaaS & VPC licensing via [Forseti Research / Forseti Enable](https://forsetienable.com))

## Quick Start

Requires **Node.js >= 20** (zero runtime npm dependencies).

```bash
# Run the deterministic verification suite
node --test test/prism-v4.test.js test/prism-v4-decision-mcp.test.js test/kernel.test.js test/credal.test.js

# Start the local stdio MCP server
npm run prism-mcp
```

## License

Source-available under **Business Source License 1.1 (BUSL 1.1)** — free for personal use, academic/independent research, benchmark verification, evaluation, and non-production testing. Converts to **Apache License 2.0** on **2030-11-24**. Commercial production use, hosted/managed service operation, or embedding into a commercial product requires a commercial license from **Forseti Research** (`https://forsetienable.com`). See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).
