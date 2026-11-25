export { verifyInvariance, GATE_CODES } from './gate.js';
export { applyPermutation, pairwiseSwaps, allPermutations } from './swap.js';
export { ablationCheck, heuristicAssertedHead, spanScoped } from './ablation.js';
export { complementCheck } from './complement.js';
export {
  makeSemanticExtractor,
  makePositionalExtractor,
} from './reference-extractors.js';
export {
  checkSemanticEntailment,
  verifyEntailmentWitness,
  solveWithWitness,
  programNeedsEntailment,
  passageSha256,
  programCanonicalSha256,
  WITNESS_MODEL,
} from './entailment.js';
