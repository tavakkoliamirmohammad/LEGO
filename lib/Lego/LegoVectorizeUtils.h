//===- LegoVectorizeUtils.h - Internal analysis utilities for lego-vectorize ===//
//
// Lowers loops over Lego-derived arith address expressions to MLIR vector
// dialect ops by symbolic stride analysis. Layout-agnostic: operates on
// post-LegoToArith IR (arith + memref + scf).
//
//===----------------------------------------------------------------------===//

#ifndef LEGO_LIB_LEGOVECTORIZEUTILS_H
#define LEGO_LIB_LEGOVECTORIZEUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <cstdint>

namespace mlir::lego {

enum class AccessKind {
  Unit,        // S(k) = k * elem_size                   (constant-stride contiguous)
  Strided,     // S(k) = k * c, c constant != elem_size  (constant-stride non-unit)
  Broadcast,   // S(k) = 0                                (loop-invariant)
  CrossBlock,  // piecewise unit-stride with single boundary (Tier B only — set later)
  NonAffine,   // simplification stalls; iv survives in S(k)
};

struct AccessClassification {
  AccessKind kind = AccessKind::NonAffine;
  int64_t stride = 0;        // for Strided
  int64_t boundary = -1;     // for CrossBlock: lane index of the discontinuity
  int64_t elementBytes = 0;  // sizeof(element) in bytes
};

// Symbolic stride solver — Tier A.
// Given a memref.load or memref.store op `memrefOp` and the candidate
// induction variable `iv`, returns a classification of
//   S(k) = simplify(addr(iv+k) - addr(iv))
// where addr is the index expression on memrefOp. Layout-agnostic — operates
// purely on the integer arithmetic DAG of the address expression.
//
// elementBytes is sizeof(element_type) in bytes (e.g. 8 for f64).
//
// This function does NOT mutate the input IR. It evaluates the index DAG
// symbolically as a linear expression in iv (AffineVal), then classifies
// the per-step difference (the coefficient of iv times elementBytes).
AccessClassification solveAccessTierA(Operation *memrefOp, Value iv,
                                      int64_t elementBytes);

// Speculative unroll: compute concrete addr(iv+0..L-1) for the given memref op,
// classify based on the actual address sequence.
//   - If all addresses differ by elementBytes from the previous: Unit.
//   - If they form a constant-stride pattern (uniform but != elementBytes): Strided(c).
//   - If the address is loop-invariant (all diffs == 0): Broadcast.
//   - If they partition into exactly two contiguous runs of unit stride with
//     a single jump between them: CrossBlock(boundary).
//   - Otherwise: NonAffine.
//
// Layout-agnostic — uses the lightweight AffineVal evaluator from Tier A
// extended to substitute concrete iv+k values.
//
// Like Tier A, this function does NOT mutate the input IR.
AccessClassification solveAccessTierB(Operation *memrefOp, Value iv,
                                      int64_t elementBytes, int64_t L);

}  // namespace mlir::lego

#endif  // LEGO_LIB_LEGOVECTORIZEUTILS_H
