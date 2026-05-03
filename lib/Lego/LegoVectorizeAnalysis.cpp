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

// AffineVal is declared in LegoVectorizeUtils.h; see there for documentation.

// Returns true if `v` is defined strictly outside the loop whose IV is `iv`.
// This is used to classify a Value as loop-invariant w.r.t. the inner loop.
// Strategy: `iv` is the block argument of the inner loop's body block. Any
// value defined in a block that does NOT dominate (is NOT a parent of) the
// inner loop is considered loop-invariant. Specifically:
//   - Block arguments of OTHER blocks (outer loop IVs, function args) are
//     invariant because they are fixed for the entire duration of the inner
//     loop's execution.
//   - Values defined by ops OUTSIDE the inner loop's region are invariant.
//
// For the @cpu_kernel(grid, tile) pattern:
//   outer scf.for %tile_id ...  ← %tile_id is loop-invariant w.r.t. inner
//     inner scf.for %local_i ...
//       %off = arith.muli %tile_id, %c16  ← ops inside inner using outer IV
//
// Here %tile_id is a block arg of the OUTER loop; it is invariant w.r.t.
// %local_i (the inner IV).
static bool isLoopInvariant(Value v, Value iv) {
  // If v IS the inner IV, it's not invariant.
  if (v == iv) return false;
  Operation *defOp = v.getDefiningOp();
  if (!defOp) {
    // v is a block argument (outer IV, function arg, etc.) — always invariant
    // w.r.t. the inner loop (since it's fixed when the inner loop runs).
    return true;
  }
  // For op-defined values: they are invariant if the defining op is NOT
  // inside the inner loop's body block. We detect this by checking if the op's
  // parent block is the same as iv's parent block (the inner loop body).
  // If the op is in a block that is NOT dominated by the inner loop body, it's
  // invariant.
  //
  // Simple conservative check: if defOp is in the same block as iv's use or
  // an ancestor, it could be invariant. We rely on evalAffine's recursive
  // structure — ops inside the loop that only depend on invariant values will
  // naturally evaluate to AffineVal with coeff=0.
  // This function is only called for non-defOp cases (block args) in the
  // main evalAffine block-arg handler below.
  return false;  // Op-defined: let evalAffine recurse to determine.
}

// Evaluate the index expression `v` symbolically.
// `iv` is the inner loop induction variable. Block args that are NOT iv
// but are defined outside the inner loop are treated as loop-invariant
// symbolic constants (AffineVal with coeff=0, invariant term set).
// Caches results in `cache` (keyed by Value) to avoid exponential blowup.
//
// Key fix for @cpu_kernel(grid, tile) patterns: the inner loop index is
//   %off = arith.muli %tile_id, %tile_size   (tile_id is outer loop IV)
//   %idx = arith.addi %off, %local_i          (local_i is inner loop IV = iv)
// Previously: %tile_id → NonAffine (bailed). Now: %tile_id → invariant_val,
// %off → {coeff=0, invariant=[%tile_id * tile_size conceptually]}, then
// %idx → {coeff=1, invariant=[%off]}.  coeff==1 → Unit stride. Correct!
AffineVal evalLinearInIV(Value v, Value iv,
                         llvm::DenseMap<Value, AffineVal> &cache) {
  // Cache hit?
  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;

  AffineVal result;

  // Block argument cases.
  Operation *defOp = v.getDefiningOp();
  if (!defOp) {
    // Block argument.
    if (v == iv) {
      // The inner loop's own IV: coeff=1.
      result = AffineVal::iv_val();
    } else {
      // A block arg that is NOT the inner IV. This includes:
      //   - Outer loop block args (tile_id, row_id, etc.)
      //   - Function/block arguments (buffer sizes, scalar args)
      // All are loop-invariant w.r.t. the inner loop: treat as opaque constant.
      result = AffineVal::invariant_val(v);
    }
    cache[v] = result;
    return result;
  }

  // Constant index.
  if (auto cst = dyn_cast<arith::ConstantIndexOp>(defOp)) {
    result = AffineVal::constant_val(cst.value());
    cache[v] = result;
    return result;
  }

  // Generic arith.constant returning index type.
  if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto iattr = dyn_cast<IntegerAttr>(cst.getValue())) {
      result = AffineVal::constant_val(iattr.getInt());
      cache[v] = result;
      return result;
    }
    result = AffineVal::nonAffine();
    cache[v] = result;
    return result;
  }

  // addi(a, b)  →  (ca+cb)*iv + (ka+kb) + (ia∪ib)
  if (auto add = dyn_cast<arith::AddIOp>(defOp)) {
    auto a = evalLinearInIV(add.getLhs(), iv, cache);
    auto b = evalLinearInIV(add.getRhs(), iv, cache);
    if (!a.valid || !b.valid) {
      result = AffineVal::nonAffine();
    } else {
      result.valid = true;
      result.coeff = a.coeff + b.coeff;
      result.constant = a.constant + b.constant;
      // Merge invariant terms from both sides.
      result.invariant.append(a.invariant);
      result.invariant.append(b.invariant);
    }
    cache[v] = result;
    return result;
  }

  // subi(a, b)  →  (ca-cb)*iv + (ka-kb) + (ia∪ib)
  if (auto sub = dyn_cast<arith::SubIOp>(defOp)) {
    auto a = evalLinearInIV(sub.getLhs(), iv, cache);
    auto b = evalLinearInIV(sub.getRhs(), iv, cache);
    if (!a.valid || !b.valid) {
      result = AffineVal::nonAffine();
    } else {
      result.valid = true;
      result.coeff = a.coeff - b.coeff;
      result.constant = a.constant - b.constant;
      result.invariant.append(a.invariant);
      result.invariant.append(b.invariant);
    }
    cache[v] = result;
    return result;
  }

  // muli(a, b):
  //   If one side has coeff==0 and no iv dependence → it's a pure invariant
  //   scalar that scales the other side.
  //   Case 1: a is pure-constant/invariant (coeff=0), b may have iv.
  //     result.coeff = a.constant * b.coeff  (only if a has no invariant terms,
  //     since invariant*iv would be non-affine).
  //   Case 2: b is pure-constant (coeff=0, no invariant), a may have iv.
  //   Case 3: Both have coeff != 0 → quadratic, not affine.
  //   Case 4: a has invariant terms and b has iv → NonAffine (iv * invariant).
  if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
    auto a = evalLinearInIV(mul.getLhs(), iv, cache);
    auto b = evalLinearInIV(mul.getRhs(), iv, cache);
    if (!a.valid || !b.valid) {
      result = AffineVal::nonAffine();
    } else if (a.coeff == 0 && a.invariant.empty()) {
      // a is a pure integer constant; b may have iv or invariant terms.
      result.valid = true;
      result.coeff = a.constant * b.coeff;
      result.constant = a.constant * b.constant;
      // Scale invariant terms of b by a.constant (conceptually; we track the
      // whole expression `v` as a new invariant term since we can't easily
      // fold the scale factor into the invariant Values list).
      // If b has invariant terms, the product is still loop-invariant → add v.
      if (!b.invariant.empty()) {
        // a * (invariant_terms) is invariant → track `v` as invariant.
        result.invariant.push_back(v);
      }
    } else if (b.coeff == 0 && b.invariant.empty()) {
      // b is a pure integer constant; a may have iv or invariant terms.
      result.valid = true;
      result.coeff = b.constant * a.coeff;
      result.constant = b.constant * a.constant;
      if (!a.invariant.empty()) {
        result.invariant.push_back(v);
      }
    } else if (a.coeff == 0 && b.coeff == 0) {
      // Both sides are invariant (no iv dependence) → result is invariant too.
      result.valid = true;
      result.coeff = 0;
      result.constant = 0;
      result.invariant.push_back(v);  // track the whole muli as invariant.
    } else {
      // iv appears in both sides (quadratic) or one side is invariant*iv.
      result = AffineVal::nonAffine();
    }
    cache[v] = result;
    return result;
  }

  // shli(a, constant) — left shift by a constant is equivalent to muli by 2^n.
  // Used by strength-reduction of power-of-2 multiplications.
  if (auto shl = dyn_cast<arith::ShLIOp>(defOp)) {
    auto a = evalLinearInIV(shl.getLhs(), iv, cache);
    auto b = evalLinearInIV(shl.getRhs(), iv, cache);
    // b must be a pure integer constant for this to be affine.
    if (!a.valid || !b.valid || b.coeff != 0 || !b.invariant.empty()) {
      result = AffineVal::nonAffine();
    } else {
      int64_t shift = b.constant;
      if (shift < 0 || shift >= 63) {
        result = AffineVal::nonAffine();
      } else {
        int64_t scale = int64_t(1) << shift;
        result.valid = true;
        result.coeff = a.coeff * scale;
        result.constant = a.constant * scale;
        if (!a.invariant.empty()) {
          result.invariant.push_back(v);
        }
      }
    }
    cache[v] = result;
    return result;
  }

  // index_cast (arith.index_cast or arith.index_castui) — treat as identity.
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(defOp)) {
    result = evalLinearInIV(defOp->getOperand(0), iv, cache);
    cache[v] = result;
    return result;
  }

  // For any other op (divui, remui, andi, ori, etc.):
  // Check if the op's result is loop-invariant by examining whether ALL of
  // its operands are loop-invariant (coeff=0, valid=true).
  // If so, treat the entire op as an opaque invariant term (like a block arg).
  // This handles cases like: %x = arith.andi %outer_iv, %mask (where outer_iv
  // is invariant w.r.t. inner iv) — the result is still invariant.
  {
    bool allInvariant = true;
    for (Value operand : defOp->getOperands()) {
      auto ov = evalLinearInIV(operand, iv, cache);
      if (!ov.valid || ov.coeff != 0) {
        allInvariant = false;
        break;
      }
    }
    if (allInvariant) {
      // All operands are loop-invariant → the op result is loop-invariant too.
      result.valid = true;
      result.coeff = 0;
      result.constant = 0;
      result.invariant.push_back(v);
      cache[v] = result;
      return result;
    }
  }

  // Anything else that depends on iv in a non-affine way → NonAffine.
  result = AffineVal::nonAffine();
  cache[v] = result;
  return result;
}

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

  // Evaluate S(iv) symbolically.
  llvm::DenseMap<Value, AffineVal> cache;
  AffineVal sym = evalLinearInIV(addr, iv, cache);

  if (!sym.valid) {
    cls.kind = AccessKind::NonAffine;
    return cls;
  }

  // The per-step difference is S(iv+1) - S(iv) = coeff (the coefficient of iv).
  // (S(iv+1) = coeff*(iv+1) + constant = coeff*iv + coeff + constant)
  // diff = S(iv+1) - S(iv) = coeff.
  //
  // The physical byte-stride for a flat-buffer access is coeff * elementBytes
  // (memref indices are in elements, not bytes).
  // We classify based on coeff * elementBytes:
  int64_t byteStride = sym.coeff * elementBytes;

  if (sym.coeff == 0) {
    // Address is loop-invariant.
    cls.kind = AccessKind::Broadcast;
  } else if (byteStride == elementBytes) {
    // Unit stride: advancing by one element per iteration.
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
