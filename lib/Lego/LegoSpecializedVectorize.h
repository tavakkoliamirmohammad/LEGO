//===- LegoSpecializedVectorize.h - Shared helpers for specialised passes -===//
//
// Internal-only helpers shared by the family of "specialised pattern"
// vectorise passes that LEGO ships:
//
//     LegoVectorizeCompact          (vector.compressstore)
//     LegoVectorizeArgmin           (paired (val,idx) reduction)
//     LegoVectorizeScan             (Hillis-Steele inclusive prefix)
//     LegoVectorizeFilteredReduce   (predicated reduction)
//     LegoVectorizeRLE              (edge-detect compaction)
//     LegoVectorizeScatterAdd       (gather/add/scatter — half-width policy)
//
// Each pass strip-mines an scf.for loop into (vec_loop, tail_loop) using
// the same bounds math, marker attributes, and skip-list convention.
// Centralising those keeps the per-pass file focused on its
// pattern-specific recogniser + emit.
//
//===----------------------------------------------------------------------===//

#ifndef LEGO_LIB_SPECIALIZED_VECTORIZE_H
#define LEGO_LIB_SPECIALIZED_VECTORIZE_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <cstdint>

namespace mlir::lego::specialised {

/// Default lane width policy used by every specialised pass *except*
/// LegoVectorizeScatterAdd (which intentionally uses half-width L on
/// AVX-512 because gather/scatter is the bottleneck — see comment in
/// LegoVectorizeScatterAdd.cpp).
///
/// The policy mirrors the lane widths the hand-written x86 codegen
/// expects from ``mlir::lego::createLegoVectorizePass(target)``:
///   avx512 → 16 lanes for 32-bit, 8 lanes for 64-bit
///   avx2   →  8 lanes for 32-bit, 4 lanes for 64-bit
///   neon, sve → 4 lanes for 32-bit, 2 lanes for 64-bit
///   default → 16 / elemBytes (16-byte register assumption).
inline int64_t getDefaultLanesForType(llvm::StringRef target,
                                       int64_t elemBytes) {
  if (target == "avx512") return elemBytes == 4 ? 16 : 8;
  if (target == "avx2")   return elemBytes == 4 ? 8  : 4;
  if (target == "neon" || target == "sve")
    return elemBytes == 4 ? 4 : 2;
  return 16 / std::max<int64_t>(elemBytes, 1);
}

/// Compute the strip-mined upper bound of a unit-step loop:
///     stripUb = lb + ((ub - lb) / L) * L     ;   tail spans [stripUb, ub)
///
/// Used by every specialised pass to split a loop into a vector body
/// (``[lb, stripUb)`` step ``L``) and a scalar tail (``[stripUb, ub)``
/// step 1).  The constants ``cL`` and ``stripUb`` are returned so the
/// caller can use them directly when building the two scf.for ops.
struct StripBounds {
  ::mlir::Value cL;        // the constant L as an index value
  ::mlir::Value stripUb;   // upper bound of the vector loop
};

inline StripBounds computeStripBounds(::mlir::OpBuilder &b,
                                      ::mlir::Location loc,
                                      ::mlir::Value lb,
                                      ::mlir::Value ub,
                                      int64_t L) {
  ::mlir::Value lenV   = ::mlir::arith::SubIOp::create(b, loc, ub, lb);
  ::mlir::Value cL     = ::mlir::arith::ConstantIndexOp::create(b, loc, L);
  ::mlir::Value chunks = ::mlir::arith::DivUIOp::create(b, loc, lenV, cL);
  ::mlir::Value stripBody =
      ::mlir::arith::MulIOp::create(b, loc, chunks, cL);
  ::mlir::Value stripUb =
      ::mlir::arith::AddIOp::create(b, loc, lb, stripBody);
  return {cL, stripUb};
}

} // namespace mlir::lego::specialised

#endif // LEGO_LIB_SPECIALIZED_VECTORIZE_H
