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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::lego {

enum class AccessKind {
  Unit,        // S(k) = k * elem_size                   (constant-stride contiguous)
  Strided,     // S(k) = k * c, c constant != elem_size  (constant-stride non-unit)
  Broadcast,   // S(k) = 0                                (loop-invariant)
  CrossBlock,  // piecewise unit-stride with single boundary (Tier B only — set later)
  NonAffine,   // simplification stalls; iv survives in S(k)
};

// ---------------------------------------------------------------------------
// Lightweight symbolic integer evaluator (Tier A).
//
// Represents an index expression as (coeff * iv + constant + invariant_terms).
// Declared here so that pass code (computeMinDepDistance) can call evalAffine
// directly without going through solveAccessTierA.
// ---------------------------------------------------------------------------
struct AffineVal {
  bool valid = true;
  int64_t coeff = 0;
  int64_t constant = 0;
  llvm::SmallVector<Value, 2> invariant;

  static AffineVal nonAffine()            { AffineVal v; v.valid = false; return v; }
  static AffineVal constant_val(int64_t c){ AffineVal v; v.coeff = 0; v.constant = c; return v; }
  static AffineVal iv_val()               { AffineVal v; v.coeff = 1; v.constant = 0; return v; }
  static AffineVal invariant_val(Value inv) {
    AffineVal v; v.coeff = 0; v.constant = 0; v.invariant.push_back(inv); return v;
  }
};

// Evaluate the index expression `v` symbolically w.r.t. induction variable
// `iv`. Caches intermediate results in `cache`.
// Defined in LegoVectorizeAnalysis.cpp.
// evalLinearInIV: evaluate the index expression `v` symbolically w.r.t.
// induction variable `iv`. Returns an AffineVal representing the linear model
// `coeff * iv + constant + Σ invariant_i`. Named "LinearInIV" to be precise:
// this is a linear (not general MLIR affine) test — it checks whether `v` is
// linear in `iv`, which is exactly what the vectorizer needs.
// (Not to be confused with the MLIR affine dialect.)
// Defined in LegoVectorizeAnalysis.cpp.  Formerly named evalAffine (renamed
// in Finding 10 to distinguish from the MLIR affine dialect).
AffineVal evalLinearInIV(Value v, Value iv,
                         llvm::DenseMap<Value, AffineVal> &cache);

struct AccessClassification {
  AccessKind kind = AccessKind::NonAffine;
  int64_t stride = 0;           // for Strided
  int64_t boundary = -1;        // for CrossBlock: lane index of the discontinuity
  int64_t boundaryJump = 0;     // for CrossBlock: address jump at the discontinuity
                                //   = addrs[boundary] - addrs[boundary-1]
                                //   (in element-index units, not bytes).
                                //   The second-block base is at addrs[boundary],
                                //   i.e. baseIv + (addrs[boundary] - addrs[0]).
                                //   R12a threads this through so emission uses the
                                //   real jump rather than assuming contiguous bricks.
  int64_t block0Offset = 0;     // for CrossBlock: addrs[0] - iv_at_probe_start
                                //   (the offset from iv to the first probed address).
                                //   Normally 0 for Tier-B probing from iv=0.
  int64_t elementBytes = 0;     // sizeof(element) in bytes
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
//   Note: memref indices are in element units (not bytes), so unit stride means
//   consecutive element indices differ by 1, not by elementBytes.
//   - If all consecutive differences == 1: Unit.
//   - If they form a constant-stride pattern (uniform but != 1 and != 0): Strided(c).
//   - If the address is loop-invariant (all diffs == 0): Broadcast.
//   - If they partition into exactly two contiguous runs of unit stride (diff==1)
//     with a single jump between them: CrossBlock(boundary).
//   - Otherwise: NonAffine.
//
// Layout-agnostic — uses the lightweight AffineVal evaluator from Tier A
// extended to substitute concrete iv+k values.
//
// Like Tier A, this function does NOT mutate the input IR.
AccessClassification solveAccessTierB(Operation *memrefOp, Value iv,
                                      int64_t elementBytes, int64_t L);

// ---------------------------------------------------------------------------
// Cost model constants and register-lane helper (Finding 4).
//
// Centralises all tuning parameters so that a reader of the header sees the
// full cost model without scanning LegoVectorize.cpp.  The constants are
// compile-time values; no behavior change.
// ---------------------------------------------------------------------------
struct CostModel {
  // Penalty applied to pure gather-only loops.
  // SOURCE: Intel Optimization Reference Manual §2.5.5, Table 2-9.
  static constexpr double kStridedGatherPenalty  = 5.0;
  // SOURCE: Polychroniou et al., SIGMOD'15; Pandey et al., SC'19.
  static constexpr double kNonAffineGatherPenalty = 10.0;
  // ILP unroll multiplier for pure unit-stride loops.
  // SOURCE: Agner Fog §12.7; matches Clang's UnrollFactor for AVX-512.
  static constexpr int64_t kILPFactor = 4;
};

}  // namespace mlir::lego

#endif  // LEGO_LIB_LEGOVECTORIZEUTILS_H
