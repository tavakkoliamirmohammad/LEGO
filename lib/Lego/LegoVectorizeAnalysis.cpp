//===- LegoVectorizeAnalysis.cpp - Stride-analysis routines ---------------===//
//
// Pure-analysis helpers for the lego-vectorize pass. These functions inspect
// the integer-arithmetic DAG of a memref.load / memref.store index expression
// and classify its access pattern relative to an induction variable.
//
// Two analysis tiers:
//   Tier A (solveAccessTierA): symbolic AffineVal evaluator.
//     Represents the index as coeff*iv + constant + invariant_terms.
//     Classifies based on the per-step difference (= coeff).
//     Handles unit-stride, broadcast, and constant-stride (non-affine-free).
//
//   Tier B (solveAccessTierB): speculative concrete-unroll evaluator.
//     Probes addr(iv=0..L-1) concretely and inspects the resulting sequence.
//     Handles CrossBlock (piecewise unit-stride with single boundary) that
//     Tier A cannot detect symbolically because the index arithmetic contains
//     a modulo or other non-affine op.
//
// Neither tier mutates the input IR.
//
//===----------------------------------------------------------------------===//

#include "LegoAffineExtract.h"
#include "LegoVectorizeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"

#include <optional>

using namespace mlir;

// ---------------------------------------------------------------------------
// Tier-A stride solver implementation
// ---------------------------------------------------------------------------
namespace mlir::lego {

// (The custom ``AffineVal`` evaluator and ``evalLinearInIV`` were deleted
// here. Both ``solveAccessTierA`` (below) and ``computeMinDepDistance``
// in LegoVectorize.cpp now use the upstream-MLIR-based
// ``tryBuildAffineExpr`` from LegoAffineExtract.h. The legacy code lives
// in git history if it ever needs to be referenced.)


AccessClassification solveAccessTierA(Operation *memrefOp, Value iv,
                                      int64_t elementBytes) {
  AccessClassification cls;
  cls.elementBytes = elementBytes;

  // Extract the (first) index from a load or store.
  Value addr;
  if (auto load = dyn_cast<memref::LoadOp>(memrefOp)) {
    if (load.getIndices().empty()) return cls;  // scalar — NonAffine
    addr = load.getIndices().front();
  } else if (auto store = dyn_cast<memref::StoreOp>(memrefOp)) {
    if (store.getIndices().empty()) return cls;
    addr = store.getIndices().front();
  } else {
    return cls;
  }

  // Build an MLIR AffineExpr for the address relative to the IV. This walks
  // the SSA cone and returns nullopt for any op that isn't representable in
  // affine form (e.g. bitwise interleave for Z-Morton, ``i * j`` of two IVs,
  // data-dependent gather). The non-affine path falls through to Tier-B.
  auto extracted = tryBuildAffineExpr(addr, iv, memrefOp->getContext());
  if (!extracted) {
    cls.kind = AccessKind::NonAffine;
    return cls;
  }

  // Read off the coefficient of d0 (the IV). For pure-linear addresses this
  // is a compile-time integer; for ``d0 * symbol`` (runtime-valued stride) or
  // ``d0 floordiv c`` (non-linear in d0), getDim0Coefficient returns nullopt
  // and we fall back to NonAffine so Tier-B can probe the address sequence
  // concretely.
  std::optional<int64_t> coef = getDim0Coefficient(extracted->expr);
  if (!coef) {
    cls.kind = AccessKind::NonAffine;
    return cls;
  }

  // The per-step difference (S(iv+1) - S(iv)) is exactly the d0 coefficient
  // in element units. Multiply by elementBytes to get the byte-stride.
  int64_t byteStride = (*coef) * elementBytes;

  if (*coef == 0) {
    cls.kind = AccessKind::Broadcast;
  } else if (byteStride == elementBytes) {
    cls.kind = AccessKind::Unit;
    cls.stride = elementBytes;
  } else {
    cls.kind = AccessKind::Strided;
    cls.stride = byteStride;
  }

  return cls;
}

}  // namespace mlir::lego

// ---------------------------------------------------------------------------
// Tier-B speculative-unroll evaluator
// ---------------------------------------------------------------------------
namespace {

// Concrete-evaluation of an integer-arith DAG: given that the value of `iv`
// is fixed at `ivVal`, returns the concrete int64_t result of evaluating
// `root`, or std::nullopt if any node can't be evaluated (e.g. depends on
// non-iv block args, has a non-supported op).
static std::optional<int64_t>
evalConcreteIV(Value root, Value iv, int64_t ivVal,
                 llvm::DenseMap<Value, std::optional<int64_t>> &cache) {
  if (auto it = cache.find(root); it != cache.end()) return it->second;
  std::optional<int64_t> result;

  if (root == iv) {
    result = ivVal;
    cache[root] = result;
    return result;
  }
  Operation *defOp = root.getDefiningOp();
  if (!defOp) {
    // Other block argument — not evaluable.
    cache[root] = std::nullopt;
    return std::nullopt;
  }

  // Constants.
  if (auto cst = dyn_cast<arith::ConstantIndexOp>(defOp)) {
    result = cst.value();
  } else if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
      result = ia.getInt();
    }
  // Binary integer ops.
  } else if (defOp->getNumOperands() == 2 && defOp->getNumResults() == 1) {
    auto a = evalConcreteIV(defOp->getOperand(0), iv, ivVal, cache);
    auto b = evalConcreteIV(defOp->getOperand(1), iv, ivVal, cache);
    if (a && b) {
      if (isa<arith::AddIOp>(defOp))          result = *a + *b;
      else if (isa<arith::SubIOp>(defOp))     result = *a - *b;
      else if (isa<arith::MulIOp>(defOp))     result = *a * *b;
      else if (isa<arith::DivUIOp>(defOp) && *b != 0)
        result = (int64_t)((uint64_t)*a / (uint64_t)*b);
      else if (isa<arith::DivSIOp>(defOp) && *b != 0) result = *a / *b;
      else if (isa<arith::RemUIOp>(defOp) && *b != 0)
        result = (int64_t)((uint64_t)*a % (uint64_t)*b);
      else if (isa<arith::RemSIOp>(defOp) && *b != 0) result = *a % *b;
      else if (isa<arith::ShLIOp>(defOp))     result = *a << *b;
      else if (isa<arith::ShRUIOp>(defOp))
        result = (int64_t)((uint64_t)*a >> *b);
      else if (isa<arith::ShRSIOp>(defOp))    result = *a >> *b;
      else if (isa<arith::AndIOp>(defOp))     result = *a & *b;
      else if (isa<arith::OrIOp>(defOp))      result = *a | *b;
      else if (isa<arith::XOrIOp>(defOp))     result = *a ^ *b;
      // Otherwise: unsupported, leave result as nullopt.
    }
  } else if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(defOp)) {
    result = evalConcreteIV(defOp->getOperand(0), iv, ivVal, cache);
  }
  // (Add more ops as needed — the above covers most LEGO-generated index arithmetic.)

  cache[root] = result;
  return result;
}

}  // anonymous namespace

namespace mlir::lego {

AccessClassification solveAccessTierB(Operation *memrefOp, Value iv,
                                      int64_t elementBytes, int64_t L) {
  AccessClassification cls;
  cls.elementBytes = elementBytes;
  cls.kind = AccessKind::NonAffine;

  // Get the address index value (first index of memref.load/store).
  Value addr;
  if (auto load = dyn_cast<memref::LoadOp>(memrefOp)) {
    if (!load.getIndices().empty()) addr = load.getIndices().front();
  } else if (auto store = dyn_cast<memref::StoreOp>(memrefOp)) {
    if (!store.getIndices().empty()) addr = store.getIndices().front();
  }
  if (!addr) return cls;

  // Probe addr(iv = k) for k = 0..L-1.
  // (For v1, baseline=0: matches the "starting iteration" of the strip-mined
  // loop. The pattern test uses differences, so baseline only shifts all
  // addresses uniformly — it doesn't change classification.)
  llvm::DenseMap<Value, std::optional<int64_t>> cache;
  llvm::SmallVector<int64_t, 16> addrs;
  addrs.reserve(L);
  for (int64_t k = 0; k < L; ++k) {
    cache.clear();
    auto v = evalConcreteIV(addr, iv, /*ivVal=*/k, cache);
    if (!v) return cls;  // NonAffine — can't evaluate.
    addrs.push_back(*v);
  }
  if ((int64_t)addrs.size() < 2) return cls;

  // Compute consecutive differences and check uniformity.
  int64_t firstStep = addrs[1] - addrs[0];
  bool uniform = true;
  int64_t boundary = -1;
  int boundaryCount = 0;
  for (size_t i = 1; i < addrs.size(); ++i) {
    int64_t step = addrs[i] - addrs[i - 1];
    if (step != firstStep) {
      uniform = false;
      if (boundaryCount == 0) boundary = (int64_t)i;
      boundaryCount++;
    }
  }

  if (uniform) {
    // memref indices are in element units (not bytes), so unit stride means
    // consecutive element indices differ by 1, not by elementBytes.
    if (firstStep == 1) {
      cls.kind = AccessKind::Unit;
      cls.stride = elementBytes;
    } else if (firstStep == 0) {
      cls.kind = AccessKind::Broadcast;
    } else {
      cls.kind = AccessKind::Strided;
      cls.stride = firstStep;
    }
    return cls;
  }

  // Cross-block detection: (M+1) contiguous runs of unit stride with M jumps.
  //
  // GENERALITY NOTE: This detection covers ANY piecewise-affine access that
  // crosses M brick boundaries in L probe iterations, for M >= 1.
  //
  // For a brick of size B with layout: row*B + col (col=0..B-1, row=tile_id):
  //   addr(k) within brick r = base_r + k'  (unit-stride within each brick)
  //   boundary at lane = first index where the brick changes.
  // This covers all piecewise-linear layouts with up to L/2 boundaries in the
  // probe window, regardless of brick shape or element type.
  //
  // R12: boundaryCount > 1 is now handled as the multi-boundary CrossBlock case.
  // Threshold: boundaryCount <= L/2 (at most half the lanes are jump positions;
  // more would degenerate to nearly-all-jumps = NonAffine).
  {
    int64_t maxBoundaries = std::max(int64_t(1), (int64_t)addrs.size() / 2);
    if (boundaryCount >= 1 && boundaryCount <= maxBoundaries) {
      // Collect all boundary positions and verify all non-boundary steps are unit.
      llvm::SmallVector<int64_t, 4> bndPositions;
      bndPositions.reserve(boundaryCount);
      bool segmentsUnit = true;
      for (size_t i = 1; i < addrs.size(); ++i) {
        int64_t step = addrs[i] - addrs[i - 1];
        if (step != 1) {
          bndPositions.push_back((int64_t)i);  // record this boundary
        }
        // All non-jump steps must be unit stride.
        // (Jump steps are allowed at boundary positions.)
      }
      // Verify non-jump steps are unit.
      // Build a set of boundary positions for O(1) lookup.
      llvm::SmallVector<bool, 64> isBoundary(addrs.size(), false);
      for (int64_t b : bndPositions) isBoundary[b] = true;
      for (size_t i = 1; i < addrs.size(); ++i) {
        if (isBoundary[i]) continue;
        if (addrs[i] - addrs[i - 1] != 1) {
          segmentsUnit = false;
          break;
        }
      }

      if (segmentsUnit && (int64_t)bndPositions.size() == boundaryCount) {
        cls.kind = AccessKind::CrossBlock;

        // Fill the new multi-boundary fields (R12).
        cls.boundaries.clear();
        cls.boundaryJumps.clear();
        for (int64_t bpos : bndPositions) {
          cls.boundaries.push_back(bpos);
          // boundaryJump[k] = addrs[bpos] - addrs[0] (element-unit offset from
          // the strip start to the k-th segment's base).
          cls.boundaryJumps.push_back(addrs[bpos] - addrs[0]);
        }

        // Backward-compatible single-boundary fields.
        cls.boundary = bndPositions[0];
        cls.boundaryJump = addrs[bndPositions[0]] - addrs[0];
        cls.block0Offset = addrs[0];
        return cls;
      }
    }
  }

  // Otherwise: NonAffine.
  return cls;
}

}  // namespace mlir::lego
