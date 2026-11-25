// invariance-gate :: reference extractors for Phase-0 verification.
// These stand in for the neural Pointer-Factor head so the gate itself can be
// validated deterministically:
//   makeSemanticExtractor(template)   — genuine semantic binding: on any
//       counterfactual X' it re-reads the literal now sitting at each bound
//       site (via the gate-supplied remappedSites/remapSpan) and re-binds it
//       to the same owner. MUST pass the invariance law.
//   makePositionalExtractor(template) — the October A1-style bug: memorizes
//       the (owner -> value) binding from the original text and re-emits it
//       verbatim regardless of where the literals moved. MUST fail as
//       UNSTABLE_BINDING whenever literals actually move.

import * as R from '../kernel/rational.js';
import { toProgram } from '../kernel/parser.js';
import { literalsInSpan } from '../kernel/literals.js';
import { autoSentTags } from '../kernel/sentences.js';

const NUMERIC_OPS = new Set(['PRIOR', 'LIK', 'POSTERIOR_STATED']);

// The sentence-partition ledger is extractor-compiler output, not something
// the gate provides: like the live extractor, every reference extractor
// derives SENT_TAG decls from the text it is called on (original or
// counterfactual). A missing or malformed ledger is still possible — it just
// isn't the reference extractors' failure mode.
function withLedger(prog, X) {
  // re-derive on every call: template tags index the ORIGINAL text, the
  // ledger must cover the text actually being extracted
  prog.decls = prog.decls.filter((d) => d.op !== 'SENT_TAG');
  prog.decls.push(...autoSentTags(X, prog));
  return prog;
}

function clone(prog) {
  return JSON.parse(
    JSON.stringify(prog, (k, v) => (v && typeof v === 'object' && 'n' in v && 'd' in v ? { n: String(v.n), d: String(v.d) } : v)),
    (k, v) =>
      v && typeof v === 'object' && 'n' in v && 'd' in v && typeof v.n === 'string'
        ? { n: BigInt(v.n), d: BigInt(v.d) }
        : v
  );
}

export function makeSemanticExtractor(templateProgram) {
  const template = toProgram(templateProgram);
  return (X, info = {}) => {
    const prog = clone(template);
    if (info.slot === 'swap' && info.remappedSites) {
      // Re-bind each numeric decl to the literal now at its site position.
      // Decl order in the template corresponds to site order (position sort).
      const numericDecls = prog.decls.filter((d) => NUMERIC_OPS.has(d.op));
      for (const d of numericDecls) {
        const mapped = info.remapSpan ? info.remapSpan(d.span) : d.span;
        d.span = { s: mapped.s, e: mapped.e };
      }
      // values follow the literal now inside each decl's (remapped) span —
      // the gate passes remappedSites in position order; re-derive per-decl
      // by matching the site index the decl originally bound. Reference
      // extractors are handed the template's site order by construction:
      // we take the literal object each remapped site now holds in order of
      // the numeric decls' position-sorted sites.
      // NOTE: gate contract — remappedSites aligns with position-sorted
      // unique bound sites; multiple decls may share a site.
      const declSiteIdx = info.declSiteIdx;
      for (let i = 0; i < numericDecls.length; i++) {
        const d = numericDecls[i];
        if (declSiteIdx) {
          const rs = info.remappedSites[declSiteIdx[i]];
          if (rs) d.value = rs.value;
        }
      }
    }
    return withLedger(prog, X);
  };
}

export function makePositionalExtractor(templateProgram) {
  const template = toProgram(templateProgram);
  return (X, info = {}) => {
    const prog = clone(template);
    // The bug: spans are re-anchored (so spans still "look right") but values
    // are emitted from the memorized original binding — they do not track
    // the moved literals.
    if (info.slot === 'swap' && info.remapSpan) {
      for (const d of prog.decls) {
        if (d.span) {
          const m = info.remapSpan(d.span);
          d.span = { s: m.s, e: m.e };
        }
      }
    }
    return withLedger(prog, X);
  };
}

// makeMisbindingExtractor(template) — the PERSISTENT owner-swap bug: on every
// text (original and counterfactual alike) it rotates each numeric decl's
// bound site to the next decl's site within the same operator and binds the
// literal now sitting there. Values track sites perfectly, so numeric
// invariance alone would PASS it — but every binding is grammatically owned
// by the wrong hypothesis. The source-grounded ownership witness must catch
// it as UNSTABLE_BINDING.
export function makeMisbindingExtractor(templateProgram) {
  const template = toProgram(templateProgram);
  return (X, info = {}) => {
    const prog = clone(template);
    // keep decl->original-span so counterfactual calls can remap donor sites
    const numericDecls = prog.decls.filter((d) => NUMERIC_OPS.has(d.op));
    for (const op of NUMERIC_OPS) {
      const group = numericDecls.filter((d) => d.op === op && d.span);
      if (group.length < 2) continue;
      const origSpans = group.map((d) => d.span);
      for (let i = 0; i < group.length; i++) {
        const donorSpan = origSpans[(i + 1) % group.length];
        const mapped = info.remapSpan ? info.remapSpan(donorSpan) : donorSpan;
        const lits = literalsInSpan(X, mapped);
        group[i].span = { s: mapped.s, e: mapped.e };
        if (lits.length > 0) group[i].value = lits[0].value;
      }
    }
    // re-anchor non-numeric spans like the positional double does
    if (info.slot === 'swap' && info.remapSpan) {
      for (const d of prog.decls) {
        if (d.span && !NUMERIC_OPS.has(d.op)) {
          const m = info.remapSpan(d.span);
          d.span = { s: m.s, e: m.e };
        }
      }
    }
    return withLedger(prog, X);
  };
}
