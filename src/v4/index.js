// prism-v4 :: public entry point — Universal Semantic-Witness +
// Exact Symbolic Cage. Self-contained new module; v2 kernel/invariance/
// live exports are untouched and fully backward-compatible.
export {
  compileV4,
  makeV4OllamaExtractor,
  normCtxV4,
  ExtractionError,
  V4_EXTRACTOR_MODEL,
} from './extractor.js';
export {
  checkV4SemanticEntailment,
  verifyV4Witness,
  solveV4,
  buildV4Checks,
  V4_WITNESS_MODEL,
} from './witness.js';
export {
  verifyV4,
  V4_GATE_CODES,
  normSwapName,
  coherentSwapAssigns,
  coherentSwapsForProgram,
  alignEntityNames,
} from './gate.js';
export {
  decideFromBelief,
  decideFromProse,
  parseExactPayoff,
  POISONED_BELIEF_CODES,
} from './decision.js';
