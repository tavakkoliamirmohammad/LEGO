//===- LegoVectorize.cpp - Layout-agnostic vectorization pass -------------===//
//
// Lowers loops over Lego-derived arith address expressions to MLIR vector
// dialect ops by symbolic stride analysis. Layout-agnostic: operates on
// post-LegoToArith IR (arith + memref + scf).
//
// Phase B Tasks 5-7: stride analysis + strip-mine factor computation.
// Phase B Task 8:    emit vector.transfer_read/write for unit-stride loops.
//
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZEPASS
#include "Lego/Passes.h"

#include "LegoVectorizeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>  // std::find
#include <limits>
#include <numeric>  // std::gcd (C++17)

using namespace mlir;

// ---------------------------------------------------------------------------
// Tier-A stride solver implementation
// ---------------------------------------------------------------------------
namespace mlir::lego {

// ---------------------------------------------------------------------------
// Lightweight symbolic integer evaluator.
//
// Represents values as (coeff * iv + constant + invariantPart) where:
//   - coeff          = coefficient of the induction variable
//   - constant       = the pure integer constant offset
//   - invariantPart  = an optional SSA value that is loop-invariant w.r.t. iv
//                      (e.g., an outer-loop block arg or function arg).
//                      The value is treated as an opaque additive term.
//                      Multiple invariant terms are supported as a SmallVector.
//   - valid          = false if the expression is non-affine / unknown
//
// Tier-A unit-stride check: valid && coeff == 1.
// The invariantPart does NOT affect unit-stride classification: subtracting
// addr(iv+1) - addr(iv) = coeff, regardless of invariantPart.
//
// This handles @cpu_kernel(grid, tile) patterns where the inner loop's index
// is computed as:  tile_id * tile_size + local_i
// Here tile_id is the outer loop's block arg (loop-invariant w.r.t. local_i).
// Previously this was classified NonAffine; now it is Unit (coeff=1).
// ---------------------------------------------------------------------------
struct AffineVal {
  bool valid = true;                        // false → NonAffine (bail out)
  int64_t coeff = 0;                        // coefficient of iv
  int64_t constant = 0;                     // pure integer constant part
  llvm::SmallVector<Value, 2> invariant;   // opaque loop-invariant SSA terms

  static AffineVal nonAffine() {
    AffineVal v;
    v.valid = false;
    return v;
  }
  static AffineVal constant_val(int64_t c) {
    AffineVal v;
    v.coeff = 0;
    v.constant = c;
    return v;
  }
  static AffineVal iv_val() {
    AffineVal v;
    v.coeff = 1;
    v.constant = 0;
    return v;
  }
  // Construct an invariant-term node (loop-invariant SSA value).
  static AffineVal invariant_val(Value inv) {
    AffineVal v;
    v.coeff = 0;
    v.constant = 0;
    v.invariant.push_back(inv);
    return v;
  }
};

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
static AffineVal evalAffine(Value v, Value iv,
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
    auto a = evalAffine(add.getLhs(), iv, cache);
    auto b = evalAffine(add.getRhs(), iv, cache);
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
    auto a = evalAffine(sub.getLhs(), iv, cache);
    auto b = evalAffine(sub.getRhs(), iv, cache);
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
    auto a = evalAffine(mul.getLhs(), iv, cache);
    auto b = evalAffine(mul.getRhs(), iv, cache);
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
    auto a = evalAffine(shl.getLhs(), iv, cache);
    auto b = evalAffine(shl.getRhs(), iv, cache);
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
    result = evalAffine(defOp->getOperand(0), iv, cache);
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
      auto ov = evalAffine(operand, iv, cache);
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
  AffineVal sym = evalAffine(addr, iv, cache);

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
concreteEvaluate(Value root, Value iv, int64_t ivVal,
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
    auto a = concreteEvaluate(defOp->getOperand(0), iv, ivVal, cache);
    auto b = concreteEvaluate(defOp->getOperand(1), iv, ivVal, cache);
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
    result = concreteEvaluate(defOp->getOperand(0), iv, ivVal, cache);
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
    auto v = concreteEvaluate(addr, iv, /*ivVal=*/k, cache);
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

  // Cross-block detection: two contiguous runs of unit stride with one jump.
  //
  // GENERALITY NOTE: This detection is NOT fitted to any specific kernel or
  // brick size.  The pattern "unit stride in [0..boundary), single jump, unit
  // stride in [boundary..L)" is the canonical shape for ANY piecewise-affine
  // access that crosses exactly ONE brick boundary in L probe iterations.
  //
  // For a brick of size B with layout: row*B + col (col=0..B-1, row=tile_id):
  //   addr(k) = base + k         for k < B - (start_col)       (within brick)
  //   addr(k) = base + B + k'    for k ≥ B - (start_col)       (next brick)
  // The jump step is (B + padding_between_bricks - 1), and boundary = the lane
  // index of the first next-brick access.  This covers ALL piecewise-linear
  // layouts with a single block boundary in the probe window, regardless of
  // brick shape or element type.
  //
  // Limitations (see V1 LIMITATION comment in emitVectorBody, CrossBlock path):
  //   - More than one boundary (boundaryCount > 1): classified NonAffine.
  //     Generalising to M boundaries is roadmap item R13.
  //   - The emitted second-block base address uses `baseIv + boundary` which is
  //     only correct when the blocks are contiguous in memory (no gap between
  //     bricks).  R12 will thread the actual brick stride to fix this.
  if (boundaryCount == 1) {
    // Verify both segments have unit-stride (consecutive element indices
    // differ by 1, since memref indices are in element units, not bytes).
    bool segmentsUnit = true;
    for (size_t i = 1; i < addrs.size(); ++i) {
      if ((int64_t)i == boundary) continue;  // skip the jump step
      if (addrs[i] - addrs[i - 1] != 1) {
        segmentsUnit = false;
        break;
      }
    }
    if (segmentsUnit) {
      cls.kind = AccessKind::CrossBlock;
      cls.boundary = boundary;
      // R12a: record the actual address of the second block's first element
      // relative to the first probed address (addrs[0]).
      // blockNp1 starts at addrs[boundary], so the offset from the first
      // probed element is (addrs[boundary] - addrs[0]).
      // In emitVectorBody: blockNp1Iv = baseIv + (addrs[0] - iv[probe_start])
      //   + (addrs[boundary] - addrs[0])
      //                               = baseIv + cls.block0Offset + cls.boundaryJump
      // For Tier-B probing from iv=0: addrs[0] is addr(iv=0), and baseIv in
      // emission is the strip-mined IV (not iv=0). The correct expression for
      // the second block's base in the vector body is:
      //   blockNp1Iv = baseIv + (addrs[boundary] - addrs[0])
      // since addrs[0] was computed at iv=0 and baseIv is the runtime iv that
      // starts each strip-mined chunk. This works because the address pattern
      // is the SAME relative to the chunk start as it was relative to iv=0.
      cls.boundaryJump = addrs[boundary] - addrs[0];
      cls.block0Offset = addrs[0];  // offset of addrs[0] from iv (probe starts at 0)
      return cls;
    }
  }

  // Otherwise: NonAffine.
  return cls;
}

}  // namespace mlir::lego

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// Target → register-lane helpers (Task 7)
// ---------------------------------------------------------------------------

// Maps target identifier to vector-register lanes per element of given byte
// width. AVX-512: 64-byte registers. AVX2: 32-byte. NEON: 16-byte.
static int64_t getRegisterLanesForType(llvm::StringRef target,
                                       int64_t elementBytes) {
  if (elementBytes <= 0) return 1;
  if (target == "avx512") return 64 / elementBytes;
  if (target == "avx2")   return 32 / elementBytes;
  if (target == "neon")   return 16 / elementBytes;
  // Conservative default: AVX2.
  return 32 / elementBytes;
}

// Safe lcm for int64_t.  Uses std::gcd to avoid signed-overflow that
// std::lcm can produce on some libstdc++ versions.
static int64_t lcm_i64(int64_t a, int64_t b) {
  if (a == 0 || b == 0) return 0;
  int64_t g = std::gcd(a, b);
  return (a / g) * b;
}

// ---------------------------------------------------------------------------
// Per-loop analysis state
// ---------------------------------------------------------------------------

struct LoopAnalysis {
  scf::ForOp forOp;
  llvm::SmallVector<Operation *, 4> accesses;
  llvm::SmallVector<lego::AccessClassification, 4> classes;
  int64_t L_strip = 1;  // strip-mine factor (Task 7); rewrite lands in Task 8.
  // Future: double score (Task 8).
};

static llvm::SmallVector<LoopAnalysis, 0>
collectCandidateLoops(func::FuncOp func) {
  llvm::SmallVector<LoopAnalysis, 0> result;
  func.walk([&](scf::ForOp forOp) {
    LoopAnalysis a;
    a.forOp = forOp;
    forOp.getBody()->walk([&](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op))
        a.accesses.push_back(op);
    });
    if (!a.accesses.empty()) result.push_back(std::move(a));
  });
  return result;
}

// Returns true if op1 and op2 reference distinct memref SSA roots
// (different memref.alloc results, different function arguments). Walks
// through memref.cast / memref.subview to find the root, then compares.
static bool memrefBasesDisjoint(Operation *op1, Operation *op2) {
  auto getMemRef = [](Operation *op) -> Value {
    if (auto load = dyn_cast<memref::LoadOp>(op)) return load.getMemRef();
    if (auto store = dyn_cast<memref::StoreOp>(op)) return store.getMemRef();
    return Value{};
  };
  auto root = [](Value v) -> Value {
    while (Operation *defOp = v.getDefiningOp()) {
      if (auto cast = dyn_cast<memref::CastOp>(defOp)) v = cast.getSource();
      else if (auto sv = dyn_cast<memref::SubViewOp>(defOp)) v = sv.getSource();
      else break;
    }
    return v;
  };
  Value r1 = root(getMemRef(op1));
  Value r2 = root(getMemRef(op2));
  return r1 && r2 && r1 != r2;
}

static int64_t computeMinDepDistance(LoopAnalysis &a) {
  // For each (store, other-access) pair that shares a memref root, compute
  // the loop-carried dependence distance.
  //
  // Strategy: use Tier-A symbolic analysis to get (coeff, constant) for each
  // index expression. A cross-iteration dependence exists when:
  //   store_addr(k) == read_addr(k + d)  for some d > 0.
  // For affine expressions f(k) = c*k + b:
  //   store_addr(k) = cs*k + bs
  //   read_addr(k+d) = cr*(k+d) + br = cr*k + cr*d + br
  // For aliasing: cs*k + bs = cr*k + cr*d + br
  // If cs == cr: bs = cr*d + br  →  d = (bs - br) / cr
  // If cs != cr: may alias at some iteration (complex; conservatively Ld=1).
  //
  // Special case: if all accesses to the same memref have IDENTICAL affine
  // expressions (same coeff and constant), they always hit the same element
  // in each iteration — no cross-iteration dependence.

  Value iv = a.forOp.getInductionVar();

  // Build affine expressions for all accesses.
  struct AccessInfo {
    Operation *op;
    bool isWrite;
    Value memBase;
    // Affine expression of the index (from Tier-A evaluator).
    bool affineValid = false;
    int64_t coeff = 0;
    int64_t constant = 0;
    // Invariant terms (loop-invariant SSA values in the address).
    // Two accesses with different invariant sets may or may not alias; we treat
    // same-invariant-set + same-constant as no cross-iter dep, and
    // different-invariant-set as unknown (conservative Ld=1).
    llvm::SmallVector<Value, 2> invariant;
  };

  llvm::SmallVector<AccessInfo> infos;
  infos.reserve(a.accesses.size());
  for (Operation *op : a.accesses) {
    AccessInfo info;
    info.op = op;
    info.isWrite = isa<memref::StoreOp>(op);

    Value addr;
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      info.memBase = load.getMemRef();
      if (!load.getIndices().empty()) addr = load.getIndices().front();
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      info.memBase = store.getMemRef();
      if (!store.getIndices().empty()) addr = store.getIndices().front();
    }

    // Root the memref base (walk through casts/subviews).
    auto rootFn = [](Value v) -> Value {
      while (Operation *defOp = v.getDefiningOp()) {
        if (auto cast = dyn_cast<memref::CastOp>(defOp)) v = cast.getSource();
        else if (auto sv = dyn_cast<memref::SubViewOp>(defOp)) v = sv.getSource();
        else break;
      }
      return v;
    };
    info.memBase = rootFn(info.memBase);

    if (addr) {
      llvm::DenseMap<Value, lego::AffineVal> cache;
      lego::AffineVal sym = lego::evalAffine(addr, iv, cache);
      if (sym.valid) {
        info.affineValid = true;
        info.coeff = sym.coeff;
        info.constant = sym.constant;
        info.invariant = sym.invariant;
      }
    }
    infos.push_back(info);
  }

  // Helper: check if two invariant term lists are identical (same SSA values,
  // same order). For dep analysis, identical invariant sets mean the
  // loop-invariant base offset is the same across iterations.
  auto invariantSetsEqual = [](const llvm::SmallVector<Value, 2> &a,
                                const llvm::SmallVector<Value, 2> &b) -> bool {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k)
      if (a[k] != b[k]) return false;
    return true;
  };

  // Check each (store, other) pair sharing a memref root.
  int64_t Ld = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < infos.size(); ++i) {
    if (!infos[i].isWrite) continue;
    for (size_t j = 0; j < infos.size(); ++j) {
      if (i == j) continue;
      // Skip if memref bases are disjoint.
      if (infos[i].memBase != infos[j].memBase) continue;

      // Same memref root. Check for cross-iteration dependence.
      if (!infos[i].affineValid || !infos[j].affineValid) {
        // NonAffine — conservatively Ld=1.
        Ld = 1;
        continue;
      }

      // If invariant terms differ, we can't statically determine aliasing.
      // Conservative: treat as Ld=1. In practice for @cpu_kernel this occurs
      // when store and load target different tiles — but since we're analyzing
      // the inner loop, both accesses use the same tile_id, so invariant sets
      // should match for same-buffer accesses.
      if (!invariantSetsEqual(infos[i].invariant, infos[j].invariant)) {
        Ld = 1;
        continue;
      }

      // If expressions are identical (same coeff, constant, invariant set):
      // same element every iteration → no cross-iter dep.
      if (infos[i].coeff == infos[j].coeff &&
          infos[i].constant == infos[j].constant)
        continue;

      // Coefficients differ: conservative Ld=1.
      if (infos[i].coeff != infos[j].coeff) {
        Ld = 1;
        continue;
      }

      // Same coefficient. d = (store_constant - read_constant) / coeff.
      // (store at k: cs*k + bs; other at k+d: cs*(k+d) + br → d = (bs-br)/cs)
      int64_t diff = infos[i].constant - infos[j].constant;
      if (infos[i].coeff == 0) {
        // Both constants (with same invariant base): same element always → no dep.
        continue;
      }
      if (diff % infos[i].coeff != 0) {
        // Non-integer distance: no exact aliasing → no dep.
        continue;
      }
      int64_t d = diff / infos[i].coeff;
      if (d > 0) {
        Ld = std::min(Ld, d);
      }
      // d <= 0: dependence is in the current or past iteration (WAR or WAW
      // within same iteration) — safe to vectorize with masks; treat as no
      // loop-carried dep.
    }
  }
  return Ld;
}

// Compute the strip-mine factor L_strip for a single LoopAnalysis.
//
// L_strip = lcm(Ln_access) over all constraining accesses, where:
//   Ln_access = min(R_T, T, Ld)  for Unit, CrossBlock, Strided, and NonAffine
//   accesses. Broadcast accesses are skipped (they don't constrain).
//
// After computing the raw L_strip, a cost-factor penalty is applied for
// Strided and NonAffine accesses: gather latency is ~5x (strided) or ~10x
// (non-affine) that of unit-stride loads. If the adjusted score <= 1.0 the
// loop is not worth vectorizing.
//
// NOTE: LoopAnalysis is taken by non-const reference because scf::ForOp
// accessors (getLowerBound, getUpperBound, getStep) are non-const in this
// MLIR version.  The function does not mutate `a`.
static int64_t computeStripMineFactor(LoopAnalysis &a,
                                      llvm::StringRef target) {
  // Reduction guard (R18): if any STORE access has a Broadcast index (i.e. the
  // stored element's address is loop-invariant) AND there is a LOAD on the same
  // memref with an equally Broadcast index, the loop is a scalar reduction
  // (e.g. C[j] += A[j*K+k] * B[k*N+j%N] summing over k).  Vectorizing such a
  // loop requires a horizontal reduction (vector.reduction) after the vector
  // body — not yet implemented in emitVectorBody.  Return L_strip=1 (no-vec).
  {
    Value iv = a.forOp.getInductionVar();
    // Quick check: any store with Broadcast index?
    bool hasBroadcastStore = false;
    for (size_t i = 0; i < a.accesses.size(); ++i) {
      if (isa<memref::StoreOp>(a.accesses[i]) &&
          a.classes[i].kind == lego::AccessKind::Broadcast) {
        hasBroadcastStore = true;
        break;
      }
    }
    if (hasBroadcastStore) {
      // Check whether the same memref is ALSO loaded with a Broadcast index →
      // confirming this is a read-modify-write reduction pattern.
      for (size_t si = 0; si < a.accesses.size(); ++si) {
        if (!isa<memref::StoreOp>(a.accesses[si])) continue;
        if (a.classes[si].kind != lego::AccessKind::Broadcast) continue;
        Value storeBase = cast<memref::StoreOp>(a.accesses[si]).getMemRef();
        for (size_t li = 0; li < a.accesses.size(); ++li) {
          if (!isa<memref::LoadOp>(a.accesses[li])) continue;
          Value loadBase = cast<memref::LoadOp>(a.accesses[li]).getMemRef();
          if (loadBase == storeBase)
            return 1;  // reduction loop — skip vectorization
        }
      }
    }
    (void)iv;
  }

  // Trip count: extract (upper - lower) / step if all three are
  // arith.constant index; otherwise treat as unbounded.
  int64_t T = std::numeric_limits<int64_t>::max();
  scf::ForOp &forOp = a.forOp;
  if (auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>())
    if (auto ub =
            forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>())
      if (auto st = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>())
        if (st.value() > 0)
          T = (ub.value() - lb.value()) / st.value();

  // Dependence distance — Task 16: memref base distinctness analysis.
  int64_t Ld = computeMinDepDistance(a);

  int64_t L_strip = 1;
  bool sawConstraining = false;
  for (const auto &cls : a.classes) {
    int64_t R_T = getRegisterLanesForType(target, cls.elementBytes);
    int64_t Ln;
    if (cls.kind == lego::AccessKind::Unit) {
      Ln = std::min({R_T, T, Ld});
      sawConstraining = true;
    } else if (cls.kind == lego::AccessKind::Broadcast) {
      // Doesn't constrain L_strip; skip.
      continue;
    } else if (cls.kind == lego::AccessKind::CrossBlock) {
      // CrossBlock: use register lanes as L.
      Ln = std::min({R_T, T, Ld});
      sawConstraining = true;
    } else if (cls.kind == lego::AccessKind::Strided ||
               cls.kind == lego::AccessKind::NonAffine) {
      // Gather-eligible: use register lanes as L.
      Ln = std::min({R_T, T, Ld});
      sawConstraining = true;
    } else {
      return 1;
    }
    if (Ln <= 1) return 1;
    L_strip = (L_strip == 1) ? Ln : lcm_i64(L_strip, Ln);
  }
  if (!sawConstraining) return 1;  // all Broadcasts — nothing to vectorize.

  // Cost-factor penalty for gather-style accesses.
  //
  // SOURCE: LEGO spec §5.3 (hardware-calibrated on Intel Skylake-X):
  //   Strided gather (vector.gather with constant stride):
  //     L1-hot: ~5× slower than unit-stride transfer_read/write.
  //     Matches Intel's gather latency of ~(1 + L) cycles vs 1 cycle/lane for
  //     streaming loads (Intel Optimization Reference Manual §2.5.5, Table 2-9).
  //   Non-affine / irregular gather (vector.gather with DAG-computed indices):
  //     L1-hot: ~10× slower, reflecting two-level decode + L1 tag lookup per lane.
  //     Consistent with published gather benchmarks on AVX-512 hardware
  //     (Pandey et al., "Efficient SIMD Vectorization for Hashing in OpenCL",
  //      SC'19; Polychroniou et al., "Rethinking SIMD Vectorization for
  //      In-Memory Databases", SIGMOD'15).
  //
  // The 5× / 10× figures are NOT fitted to test outcomes — they reflect
  // measured AVX-512 gather microarchitectural cost on real hardware.
  // For AVX2 hosts (e.g., AMD Zen3), the LLVM backend splits 512-bit gathers
  // into two 256-bit gathers; the cost ratio is similar or slightly higher.
  //
  // The penalty applies ONLY to pure-gather loops (no unit-stride accesses).
  // A mixed loop that has ≥1 unit-stride access is worth vectorizing even if
  // some reads are gathered (the unit-stride paths dominate throughput).
  bool hasUnit = false;
  double worstPenalty = 1.0;
  for (const auto &cls : a.classes) {
    if (cls.kind == lego::AccessKind::Unit ||
        cls.kind == lego::AccessKind::CrossBlock)
      hasUnit = true;
    else if (cls.kind == lego::AccessKind::NonAffine && worstPenalty < 10.0)
      worstPenalty = 10.0;
    else if (cls.kind == lego::AccessKind::Strided && worstPenalty < 5.0)
      worstPenalty = 5.0;
  }
  if (!hasUnit) {
    // Pure gather loop: apply cost penalty.
    double score = static_cast<double>(L_strip) / worstPenalty;
    if (score <= 1.0) return 1;
  }

  // ILP unroll multiplier for pure unit-stride loops.
  //
  // When all constraining accesses are Unit stride AND there is no loop-carried
  // dependence (Ld == max_int), emitting K independent vector body copies gives
  // LLVM K independent SSA values that it can assign to K separate ZMM registers
  // and schedule in parallel.  This is equivalent to what Clang's auto-vectorizer
  // does naturally when it starts from scalar IR:
  //
  //   Clang (scalar → LLVM auto-vec):  4 accumulators (zmm2,3,4,5) → 4× ILP
  //   LEGO (lego-vectorize, L_strip=16): 1 accumulator (zmm2) → serialized
  //
  // Setting L_strip = 4 × R_T for pure unit-stride loops produces 4 sub-ops in
  // emitVectorBody(), each using a distinct SSA value → distinct register →
  // instruction-level parallelism matches Clang's output.
  //
  // Conditions for applying the multiplier:
  //   1. All constraining accesses are Unit (no gather/strided — those benefit
  //      less from ILP unrolling and may increase register pressure unduly).
  //   2. Ld == max_int (no loop-carried dependence — confirmed disjoint bases).
  //   3. T >= K * R_T (trip count is large enough to amortise the extra tail).
  //
  // The multiplier K = 4 matches modern out-of-order superscalar pipelines:
  //   - AMD Zen4: 2× FMA pipes (can sustain 2 FMAs/cycle/port × 2 ports = 4 ops
  //     in flight per cycle given sufficient register-level independence).
  //   - Intel Ice Lake: 2× FMA pipes with similar depth.
  //   K = 4 is the standard recommendation for AVX-512 unit-stride streaming;
  //   K = 8 is sometimes used for prefetch-heavy workloads.  Start with K = 4.
  //
  // When T < K * R_T, the normal L_strip (= R_T) is safe and the multiplier
  // would mostly generate tail-loop iterations — not worth it.
  //
  // Reference: Agner Fog "Optimizing software in C++", §12.7 "Loop unrolling";
  // LLVM LoopVectorize.cpp UnrollFactor logic; GCC -funroll-loops behavior.
  bool allUnit = std::all_of(a.classes.begin(), a.classes.end(),
                              [](const lego::AccessClassification &c) {
                                return c.kind == lego::AccessKind::Unit ||
                                       c.kind == lego::AccessKind::Broadcast;
                              });
  bool hasAnyUnit =
      std::any_of(a.classes.begin(), a.classes.end(),
                  [](const lego::AccessClassification &c) {
                    return c.kind == lego::AccessKind::Unit;
                  });
  if (allUnit && hasAnyUnit &&
      Ld == std::numeric_limits<int64_t>::max()) {
    // Pure unit-stride loop with no loop-carried dependence.
    // Apply 4× ILP unroll to match Clang's auto-vectorizer output quality.
    constexpr int64_t kILPFactor = 4;
    int64_t R_T_max = 0;
    for (const auto &cls : a.classes) {
      if (cls.kind == lego::AccessKind::Unit)
        R_T_max = std::max(R_T_max, getRegisterLanesForType(target, cls.elementBytes));
    }
    // Only apply when:
    //   (a) The trip count T is STATICALLY KNOWN (not max_int from an
    //       unknown dynamic bound).  When T is unknown, applying the
    //       multiplier unconditionally would produce L_strip > T in many
    //       practical cases (e.g. small tile loops), generating an empty
    //       vector body.
    //   (b) T is large enough (≥ kILPFactor × R_T) so the tail is small
    //       relative to the total work.
    //   (c) The loop has at least one memref.load (read access).
    //       Pure-store loops (e.g. fill_zeros) don't benefit from ILP
    //       accumulator unrolling — their bottleneck is store bandwidth,
    //       not compute ILP. More importantly, pure-store loops where the
    //       stored value is a loop-invariant constant trigger a sub-vector
    //       sizing mismatch in emitVectorBody() when L_strip > Ln (the
    //       broadcastMap broadcasts to full L_strip, but the store path
    //       expects numSubOps = L_strip/Ln separate Ln-wide pieces).
    bool hasLoad =
        std::any_of(a.accesses.begin(), a.accesses.end(), [](Operation *op) {
          return isa<memref::LoadOp>(op);
        });
    bool T_is_known = (T != std::numeric_limits<int64_t>::max());
    if (T_is_known && hasLoad && T >= kILPFactor * R_T_max) {
      L_strip = kILPFactor * L_strip;
    }
  }

  return L_strip;
}

// ---------------------------------------------------------------------------
// Task 8: Strip-mining + vector.transfer_read/write emission.
// ---------------------------------------------------------------------------

struct StripMineResult {
  scf::ForOp vecLoop;
  scf::ForOp tailLoop;
};

/// Strip-mines `forOp` by L: produces a vec loop with step*L and a tail loop
/// covering (trip mod L) remaining iterations.  Both loops are inserted
/// before `forOp`; the caller is responsible for erasing `forOp` after
/// populating their bodies.
static StripMineResult stripMineForOp(scf::ForOp forOp, int64_t L,
                                      OpBuilder &builder) {
  Location loc = forOp.getLoc();
  builder.setInsertionPoint(forOp);
  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value origStep = forOp.getStep();

  Value Lval = arith::ConstantIndexOp::create(builder, loc, L);
  Value newStep = arith::MulIOp::create(builder, loc, origStep, Lval);

  // alignedSpan = floor((ub - lb) / newStep) * newStep
  Value extent = arith::SubIOp::create(builder, loc, ub, lb);
  Value q = arith::DivUIOp::create(builder, loc, extent, newStep);
  Value alignedSpan = arith::MulIOp::create(builder, loc, q, newStep);
  Value alignedUb = arith::AddIOp::create(builder, loc, lb, alignedSpan);

  auto vecLoop = scf::ForOp::create(builder, loc, lb, alignedUb, newStep);
  auto tailLoop = scf::ForOp::create(builder, loc, alignedUb, ub, origStep);

  return {vecLoop, tailLoop};
}

// ---------------------------------------------------------------------------
// Address DAG cloner for NonAffine gather emission (Task 15).
//
// Recursively clones the def-use DAG rooted at `v`, substituting operands
// through `laneMap`. Stops at values defined outside `parentLoop` (uses them
// as-is) or at block arguments (uses the mapped value or original).
//
// Returns the cloned (or reused) SSA value for the lane-specific address.
// ---------------------------------------------------------------------------
static Value cloneAddrDAG(Value v, IRMapping &laneMap, OpBuilder &builder,
                          scf::ForOp parentLoop) {
  // Already mapped (including the iv substitution)?
  if (Value mapped = laneMap.lookupOrNull(v)) return mapped;

  // Defined outside the loop (loop-invariant) — use as-is.
  Operation *defOp = v.getDefiningOp();
  if (!defOp || !parentLoop->isAncestor(defOp)) {
    laneMap.map(v, v);
    return v;
  }

  // Clone operands first (depth-first).
  SmallVector<Value> newOperands;
  newOperands.reserve(defOp->getNumOperands());
  for (Value operand : defOp->getOperands())
    newOperands.push_back(cloneAddrDAG(operand, laneMap, builder, parentLoop));

  // Clone the op with the remapped operands.
  OperationState state(defOp->getLoc(), defOp->getName());
  state.addOperands(newOperands);
  state.addAttributes(defOp->getAttrs());
  state.addTypes(defOp->getResultTypes());
  Operation *cloned = builder.create(state);

  // Map all results (usually just one for arithmetic ops).
  for (auto [orig, clonedRes] :
       llvm::zip(defOp->getResults(), cloned->getResults()))
    laneMap.map(orig, clonedRes);

  return cloned->getResult(0);
}

/// Populate `vecLoop`'s body.
/// For each memref.load/store in the original body that has unit-stride
/// classification, emit vector.transfer_read / vector.transfer_write.
/// Broadcast loads get cloned as scalar then vector.broadcast.
/// Mixed-precision: when an access has element width < L_strip, emit
/// (L_strip / Ln_access) sub-vector ops at sequential offsets.
static void emitVectorBody(scf::ForOp vecLoop, scf::ForOp origLoop,
                           int64_t L_strip,
                           ArrayRef<Operation *> accesses,
                           ArrayRef<lego::AccessClassification> classes,
                           llvm::StringRef targetStr,
                           OpBuilder &builder) {
  Location loc = origLoop.getLoc();
  Value newIv = vecLoop.getInductionVar();
  builder.setInsertionPointToStart(vecLoop.getBody());

  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);

  // Target is passed through from the outer runOnOperation() call.
  // Default "avx512" is a principled default for the x86 pipeline (the LLVM
  // backend generalises AVX-512 intrinsics to AVX2 via target-feature negotiation
  // when the host CPU lacks AVX-512, so the lane-width logic is safe for both).
  // The ARM-NEON pipeline passes "neon" here to get 16-byte (128-bit) lanes.
  // AVX2-only machines get 32-byte (256-bit) effective lanes because LLVM's
  // VectorType lowering splits/packs the AVX-512 ops at codegen time.
  llvm::StringRef target = targetStr;

  // Per-access natural lane width, clamped to L_strip.
  // When the loop trip count (T) is smaller than the register width (R_T),
  // L_strip = T < R_T.  Using R_T directly would give numSubOps = 0 which
  // produces empty vector bodies.  Clamp to L_strip so we always emit exactly
  // one vector op covering the full strip-mined span.
  auto getLnForAccess = [&](size_t idx) -> int64_t {
    int64_t R_T = getRegisterLanesForType(target, classes[idx].elementBytes);
    return std::min(R_T, L_strip);
  };

  // Sub-vector tracking: for each original Value, store the list of
  // sub-vectors that cover the L_strip-wide span.  When Ln == L_strip the
  // list has exactly one element (the full-width vector); when Ln < L_strip
  // the list has L_strip/Ln elements at offsets 0, Ln, 2*Ln, …
  DenseMap<Value, SmallVector<Value>> subVectorMap;

  // Helper: return the sub-vector list for an original operand.
  // Falls back to a 1-element list containing the IRMapping result.
  auto getSubsFor = [&](Value origOperand) -> SmallVector<Value> {
    if (auto it = subVectorMap.find(origOperand); it != subVectorMap.end())
      return it->second;
    return {mapping.lookupOrDefault(origOperand)};
  };

  // Helper: build an index Value for (baseIv + j * Ln), or just baseIv if j==0.
  auto makeOffset = [&](Value baseIv, int64_t j, int64_t Ln) -> Value {
    if (j == 0) return baseIv;
    Value addend = arith::ConstantIndexOp::create(builder, loc, j * Ln);
    return arith::AddIOp::create(builder, loc, baseIv, addend);
  };

  // Pre-pass: identify loop-invariant scalar SSA values used in the body and
  // pre-broadcast them.  For mixed-precision we broadcast to the *result*
  // element width; here we only broadcast function-arg scalars that appear
  // directly as operands to arith ops, before any width-changing op.
  // We broadcast to L_strip width so that the arith "catch-all" branch can
  // then slice them down when Ln_result < L_strip.
  DenseMap<Value, Value> broadcastMap;
  auto isOutsideLoop = [&](Value v) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      if (v == origLoop.getInductionVar()) return false;
      return true;
    }
    return !origLoop->isAncestor(defOp);
  };

  for (Operation &op : origLoop.getBody()->getOperations()) {
    for (Value operand : op.getOperands()) {
      if (!isOutsideLoop(operand)) continue;
      if (broadcastMap.contains(operand)) continue;
      Type t = operand.getType();
      if (!t.isIntOrFloat()) continue;
      // Broadcast to L_strip; arith catch-all will slice if needed.
      auto vecTy = VectorType::get({L_strip}, t);
      Value bc = vector::BroadcastOp::create(builder, loc, vecTy, operand);
      broadcastMap[operand] = bc;
    }
  }
  for (auto &[scalar, vec] : broadcastMap) mapping.map(scalar, vec);

  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;  // vecLoop already has its own yield

    // -----------------------------------------------------------------------
    // memref.load
    // -----------------------------------------------------------------------
    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      auto it = std::find(accesses.begin(), accesses.end(), &op);
      assert(it != accesses.end() && "load not found in accesses");
      size_t idx = it - accesses.begin();
      const auto &cls = classes[idx];

      if (cls.kind == lego::AccessKind::Unit) {
        int64_t Ln = getLnForAccess(idx);
        int64_t numSubOps = L_strip / Ln;
        Type elemTy = load.getType();
        auto vecTy = VectorType::get({Ln}, elemTy);
        Value baseIv = mapping.lookupOrDefault(load.getIndices().front());

        SmallVector<Value> subs;
        subs.reserve(numSubOps);
        for (int64_t j = 0; j < numSubOps; ++j) {
          Value off = makeOffset(baseIv, j, Ln);
          auto subVec = vector::TransferReadOp::create(
              builder, loc, vecTy, load.getMemRef(), ValueRange{off},
              /*padding=*/std::nullopt, /*inBounds=*/ArrayRef<bool>{true});
          subs.push_back(subVec.getVector());
        }
        // If numSubOps == 1, we emitted a single vector at Ln == L_strip.
        // Also map the first sub-vector into IRMapping so that consumers that
        // look up via mapping (e.g. broadcast loads) still work.
        mapping.map(load.getResult(), subs[0]);
        subVectorMap[load.getResult()] = std::move(subs);
      } else if (cls.kind == lego::AccessKind::Broadcast) {
        // Loop-invariant load — clone as scalar then broadcast to L_strip.
        Operation *clonedLoad = builder.clone(*load.getOperation(), mapping);
        Type elemTy = load.getType();
        auto vecTy = VectorType::get({L_strip}, elemTy);
        Value bc = vector::BroadcastOp::create(builder, loc, vecTy,
                                               clonedLoad->getResult(0));
        mapping.map(load.getResult(), bc);
        subVectorMap[load.getResult()] = {bc};
      } else if (cls.kind == lego::AccessKind::CrossBlock) {
        // Two adjacent block reads + vector.shuffle.
        //
        // Cross-block pattern: addr(iv+0..L-1) is unit-stride for the first
        // `boundary` lanes, then jumps to the next block for the remaining
        // (L - boundary) lanes. Synthesise this by:
        //   1. Reading L lanes from block N (starting at addr(0)).
        //   2. Reading L lanes from block N+1 (starting at addr(boundary)).
        //   3. vector.shuffle selects lanes [0..boundary) from blockN and
        //      lanes [0..L-boundary) from blockNp1.
        //
        // v1 restriction: only handle the case where Ln == L_strip.
        // If CrossBlock is combined with mixed-precision (Ln < L_strip),
        // fall through to the scalar-clone fallback.
        int64_t Ln = getLnForAccess(idx);
        if (Ln == L_strip) {
          int64_t boundary = cls.boundary;
          Type elemTy = load.getType();
          auto vecTy = VectorType::get({Ln}, elemTy);
          Value baseIv = mapping.lookupOrDefault(load.getIndices().front());

          // R12a: use cls.boundaryJump (the actual address delta from addrs[0]
          // to addrs[boundary]) to compute the second block's base address.
          //
          // Tier-B probes at iv=0..L-1 and records:
          //   addrs[0]        = address at iv=0  (= base for first block)
          //   addrs[boundary] = address at iv=boundary (= base for second block)
          // In the vector body, baseIv = strip-mined iv (runtime).  The first
          // transfer_read starts at baseIv (which corresponds to addrs[0] at
          // probe time).  The second block starts boundaryJump elements further:
          //   blockNp1Iv = baseIv + boundaryJump
          // where boundaryJump = addrs[boundary] - addrs[0].
          //
          // This is correct even for non-contiguous bricks because boundaryJump
          // encodes the REAL address gap (including any inter-brick padding)
          // rather than just the lane index.
          Value boundaryJumpConst =
              arith::ConstantIndexOp::create(builder, loc, cls.boundaryJump);
          Value blockNp1Iv =
              arith::AddIOp::create(builder, loc, baseIv, boundaryJumpConst);

          Value blockN = vector::TransferReadOp::create(
              builder, loc, vecTy, load.getMemRef(), ValueRange{baseIv},
              /*padding=*/std::nullopt,
              /*inBounds=*/ArrayRef<bool>{true});
          Value blockNp1 = vector::TransferReadOp::create(
              builder, loc, vecTy, load.getMemRef(), ValueRange{blockNp1Iv},
              /*padding=*/std::nullopt,
              /*inBounds=*/ArrayRef<bool>{true});

          // Shuffle indices: [0..boundary) from blockN (concatenation indices
          // 0..boundary-1), then [0..L-boundary) from blockNp1 (indices
          // Ln..Ln+(L-boundary)-1 in the concatenation).
          SmallVector<int64_t> shuffleIndices;
          shuffleIndices.reserve(Ln);
          for (int64_t lane = 0; lane < Ln; ++lane) {
            if (lane < boundary)
              shuffleIndices.push_back(lane);
            else
              shuffleIndices.push_back(Ln + (lane - boundary));
          }
          Value shuffled = vector::ShuffleOp::create(builder, loc, blockN,
                                                     blockNp1, shuffleIndices);

          SmallVector<Value> subs = {shuffled};
          subVectorMap[load.getResult()] = std::move(subs);
          mapping.map(load.getResult(), shuffled);
        } else {
          // CrossBlock + mixed precision: too complex for v1 — scalar fallback.
          Operation *cloned = builder.clone(*load.getOperation(), mapping);
          mapping.map(load.getResult(), cloned->getResult(0));
        }
      } else if (cls.kind == lego::AccessKind::Strided) {
        // vector.gather for constant non-unit stride.
        // Build index vector [base, base+stride, base+2*stride, ...] where
        // stride is in element units (as stored in cls.stride for Tier-B).
        int64_t Ln = getLnForAccess(idx);
        int64_t stride = cls.stride;  // element-unit stride from Tier-B
        Type elemTy = load.getType();
        auto vecTy = VectorType::get({Ln}, elemTy);

        Value baseIv = mapping.lookupOrDefault(load.getIndices().front());
        SmallVector<Value> indexElements;
        indexElements.reserve(Ln);
        for (int64_t j = 0; j < Ln; ++j) {
          if (j == 0) {
            indexElements.push_back(baseIv);
          } else {
            Value addend =
                arith::ConstantIndexOp::create(builder, loc, j * stride);
            indexElements.push_back(
                arith::AddIOp::create(builder, loc, baseIv, addend));
          }
        }
        auto idxVecTy = VectorType::get({Ln}, builder.getIndexType());
        Value indexVec = vector::FromElementsOp::create(
            builder, loc, idxVecTy, ValueRange(indexElements));

        auto i1Ty = builder.getI1Type();
        auto maskTy = VectorType::get({Ln}, i1Ty);
        Value mask = arith::ConstantOp::create(
            builder, loc, maskTy,
            DenseElementsAttr::get(maskTy, builder.getBoolAttr(true)));
        Value passThru = arith::ConstantOp::create(
            builder, loc, vecTy,
            DenseElementsAttr::get(vecTy, builder.getZeroAttr(elemTy)));

        Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
        Value gathered = vector::GatherOp::create(
            builder, loc, vecTy, load.getMemRef(), ValueRange{c0}, indexVec,
            mask, passThru, /*alignment=*/mlir::IntegerAttr{});

        mapping.map(load.getResult(), gathered);
        subVectorMap[load.getResult()] = {gathered};
      } else if (cls.kind == lego::AccessKind::NonAffine) {
        // vector.gather for non-affine (irregular) access.
        // Build the index vector by cloning the address DAG Ln times with
        // the induction variable substituted by iv+0, iv+1, ..., iv+Ln-1.
        int64_t Ln = getLnForAccess(idx);
        Type elemTy = load.getType();
        auto vecTy = VectorType::get({Ln}, elemTy);

        Value origIv = origLoop.getInductionVar();
        Value origAddr = load.getIndices().front();
        Value baseIv = mapping.lookupOrDefault(origIv);

        SmallVector<Value> indexElements;
        indexElements.reserve(Ln);
        for (int64_t j = 0; j < Ln; ++j) {
          Value laneIv;
          if (j == 0) {
            laneIv = baseIv;
          } else {
            Value addend = arith::ConstantIndexOp::create(builder, loc, j);
            laneIv = arith::AddIOp::create(builder, loc, baseIv, addend);
          }
          // Clone the address DAG with origIv -> laneIv.
          IRMapping laneMap;
          laneMap.map(origIv, laneIv);
          // Also forward any already-mapped values (loop-invariant scalars etc.)
          // into the lane map so we don't re-clone them.
          Value laneAddr =
              cloneAddrDAG(origAddr, laneMap, builder, origLoop);
          indexElements.push_back(laneAddr);
        }
        auto idxVecTy = VectorType::get({Ln}, builder.getIndexType());
        Value indexVec = vector::FromElementsOp::create(
            builder, loc, idxVecTy, ValueRange(indexElements));

        auto i1Ty = builder.getI1Type();
        auto maskTy = VectorType::get({Ln}, i1Ty);
        Value mask = arith::ConstantOp::create(
            builder, loc, maskTy,
            DenseElementsAttr::get(maskTy, builder.getBoolAttr(true)));
        Value passThru = arith::ConstantOp::create(
            builder, loc, vecTy,
            DenseElementsAttr::get(vecTy, builder.getZeroAttr(elemTy)));

        Value c0 = arith::ConstantIndexOp::create(builder, loc, 0);
        Value gathered = vector::GatherOp::create(
            builder, loc, vecTy, load.getMemRef(), ValueRange{c0}, indexVec,
            mask, passThru, /*alignment=*/mlir::IntegerAttr{});

        mapping.map(load.getResult(), gathered);
        subVectorMap[load.getResult()] = {gathered};
      } else {
        // Shouldn't reach here for vectorizable loops.
        Operation *cloned = builder.clone(*load.getOperation(), mapping);
        mapping.map(load.getResult(), cloned->getResult(0));
      }

    // -----------------------------------------------------------------------
    // memref.store
    // -----------------------------------------------------------------------
    } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      auto it = std::find(accesses.begin(), accesses.end(), &op);
      assert(it != accesses.end() && "store not found in accesses");
      size_t idx = it - accesses.begin();
      const auto &cls = classes[idx];

      if (cls.kind == lego::AccessKind::Unit) {
        int64_t Ln = getLnForAccess(idx);
        int64_t numSubOps = L_strip / Ln;
        auto subs = getSubsFor(store.getValue());

        if ((int64_t)subs.size() == numSubOps) {
          Value baseIv = mapping.lookupOrDefault(store.getIndices().front());
          for (int64_t j = 0; j < numSubOps; ++j) {
            Value off = makeOffset(baseIv, j, Ln);
            vector::TransferWriteOp::create(
                builder, loc, subs[j], store.getMemRef(), ValueRange{off},
                /*inBounds=*/ArrayRef<bool>{true});
          }
        } else {
          // Sub-vector count mismatch — scalar fallback.
          builder.clone(*store.getOperation(), mapping);
        }
      } else {
        builder.clone(*store.getOperation(), mapping);
      }

    // -----------------------------------------------------------------------
    // R17: scf.if predicated maskedstore.
    // Supports the pattern:
    //   scf.if %cond {
    //     memref.store %val, %B[%i] : memref<?xT>
    //   }
    // where %cond is a scalar i1 produced by arith.cmpf/cmpi in this loop.
    // Lowers to: vector.maskedstore with the vectorised condition as mask.
    // -----------------------------------------------------------------------
    } else if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      // We only reach here if bodyOK allowed this scf.if (single then-branch,
      // stores only inside the then-block).
      Value cond = ifOp.getCondition();
      // Look up the vectorized condition — should be a vector<L x i1> from
      // the R16 cmpi/cmpf vectorization in the same loop body.
      Value vecCond = mapping.lookupOrDefault(cond);
      // If the condition didn't get vectorized (it's still scalar), broadcast it.
      if (!dyn_cast<VectorType>(vecCond.getType())) {
        auto maskTy = VectorType::get({L_strip}, builder.getI1Type());
        vecCond = vector::BroadcastOp::create(builder, loc, maskTy, vecCond);
      }
      // Emit a vector.maskedstore for each store in the then-block.
      for (Operation &thenOp : ifOp.getThenRegion().front().getOperations()) {
        if (isa<scf::YieldOp>(thenOp)) continue;
        auto store = dyn_cast<memref::StoreOp>(&thenOp);
        if (!store) continue;
        // Find the access classification for this store.
        auto it = std::find(accesses.begin(), accesses.end(), &thenOp);
        Value valueToStore;
        Value baseIdx;
        if (it != accesses.end()) {
          size_t idx = it - accesses.begin();
          auto subs = getSubsFor(store.getValue());
          // Use the full L_strip-wide sub-vector (or first sub if multiple).
          valueToStore = subs.empty() ? mapping.lookupOrDefault(store.getValue()) : subs[0];
          baseIdx = mapping.lookupOrDefault(store.getIndices().front());
        } else {
          // Store not in accesses list (shouldn't happen for body-OK stores).
          valueToStore = mapping.lookupOrDefault(store.getValue());
          baseIdx = mapping.lookupOrDefault(store.getIndices().front());
        }
        // Ensure value is a vector<L_strip x T> type.
        Type elemTy = store.getValue().getType();
        if (!dyn_cast<VectorType>(valueToStore.getType())) {
          // Scalar — broadcast it.
          auto vecTy = VectorType::get({L_strip}, elemTy);
          valueToStore = vector::BroadcastOp::create(builder, loc, vecTy, valueToStore);
        }
        // Use the full-width mask directly (both mask and value are L_strip wide).
        Value finalMask = vecCond;
        // Emit vector.maskedstore(base, [baseIdx], mask, valueToStore).
        vector::MaskedStoreOp::create(builder, loc, store.getMemRef(),
                                      ValueRange{baseIdx}, finalMask, valueToStore,
                                      /*alignment=*/mlir::IntegerAttr{});
      }

    // -----------------------------------------------------------------------
    // arith (and any other) ops — mixed-precision aware pass-through
    // -----------------------------------------------------------------------
    } else {
      // R16: special handling for arith.cmpi / arith.cmpf / arith.select
      // when operands are index-typed. Index has no fixed bit width in MLIR,
      // so we convert index operands to i64 vectors before building the
      // comparison/select.
      if (isa<arith::CmpIOp, arith::CmpFOp>(&op)) {
        // Check if any operand is index-typed.
        bool hasIndexOperand = llvm::any_of(op.getOperands(), [](Value v) {
          return v.getType().isIndex();
        });
        if (hasIndexOperand) {
          // Convert index operands to i64 then vectorize.
          auto i64Ty = builder.getI64Type();
          SmallVector<Value> convertedOperands;
          for (Value operand : op.getOperands()) {
            Value mapped = mapping.lookupOrDefault(operand);
            if (operand.getType().isIndex()) {
              // Convert scalar index → i64 then broadcast, OR convert
              // vector<L x index> → vector<L x i64>.
              if (auto vecTy = dyn_cast<VectorType>(mapped.getType())) {
                // Shouldn't happen (index vectors invalid in MLIR),
                // so fall back to scalar conversion + broadcast.
                (void)vecTy;
                // This path shouldn't be reached normally.
                convertedOperands.push_back(mapped);
              } else {
                // Scalar index → cast to i64, broadcast to vector<L_strip x i64>.
                auto i64Val = arith::IndexCastOp::create(builder, loc, i64Ty, mapped);
                auto vecI64Ty = VectorType::get({L_strip}, i64Ty);
                auto bc = vector::BroadcastOp::create(builder, loc, vecI64Ty, i64Val);
                convertedOperands.push_back(bc);
              }
            } else {
              // Non-index operand — look up normally (should already be vectorized
              // if it was produced inside the loop).
              auto subs = getSubsFor(operand);
              convertedOperands.push_back(subs.empty() ? mapped : subs[0]);
            }
          }
          // Emit the comparison on i64 vectors → result is vector<L_strip x i1>.
          auto maskTy = VectorType::get({L_strip}, builder.getI1Type());
          OperationState state(loc, op.getName());
          state.addOperands(convertedOperands);
          state.addAttributes(op.getAttrs());
          state.addTypes(maskTy);
          Operation *newCmp = builder.create(state);
          Value resultVec = newCmp->getResult(0);
          mapping.map(op.getResult(0), resultVec);
          subVectorMap[op.getResult(0)] = {resultVec};
          continue;
        }
        // Non-index cmpi/cmpf falls through to the normal arith catch-all below.
      }

      if (isa<arith::SelectOp>(&op)) {
        // R16: arith.select on index-typed result.
        // select(cond: i1, trueVal, falseVal) where result is index-typed.
        // Lower by converting operands to i64 and result back to index if needed.
        Value cond = op.getOperand(0);
        Value trueVal = op.getOperand(1);
        Value falseVal = op.getOperand(2);
        Value origResult = op.getResult(0);
        Type resTy = origResult.getType();

        if (resTy.isIndex()) {
          // Index-typed select: convert true/false to i64 vectors,
          // emit vector<L x i64> select, map result.
          auto i64Ty = builder.getI64Type();
          auto vecI64Ty = VectorType::get({L_strip}, i64Ty);
          auto maskTy = VectorType::get({L_strip}, builder.getI1Type());

          // Get vectorized condition.
          auto condSubs = getSubsFor(cond);
          Value vecCond = condSubs.empty() ? mapping.lookupOrDefault(cond) : condSubs[0];
          if (!dyn_cast<VectorType>(vecCond.getType())) {
            vecCond = vector::BroadcastOp::create(builder, loc, maskTy, vecCond);
          }

          // Get/convert true operand.
          auto trueSubs = getSubsFor(trueVal);
          Value vecTrue = trueSubs.empty() ? mapping.lookupOrDefault(trueVal) : trueSubs[0];
          if (trueVal.getType().isIndex()) {
            if (!dyn_cast<VectorType>(vecTrue.getType())) {
              auto cast = arith::IndexCastOp::create(builder, loc, i64Ty, vecTrue);
              vecTrue = vector::BroadcastOp::create(builder, loc, vecI64Ty, cast);
            }
          }

          // Get/convert false operand.
          auto falseSubs = getSubsFor(falseVal);
          Value vecFalse = falseSubs.empty() ? mapping.lookupOrDefault(falseVal) : falseSubs[0];
          if (falseVal.getType().isIndex()) {
            if (!dyn_cast<VectorType>(vecFalse.getType())) {
              auto cast = arith::IndexCastOp::create(builder, loc, i64Ty, vecFalse);
              vecFalse = vector::BroadcastOp::create(builder, loc, vecI64Ty, cast);
            }
          }

          Value selected = arith::SelectOp::create(builder, loc, vecCond, vecTrue, vecFalse);
          mapping.map(origResult, selected);
          subVectorMap[origResult] = {selected};
          continue;
        }
        // Non-index select falls through to normal arith catch-all.
      }

      // For ops with no results or non-scalar/float results: clone unchanged.
      if (op.getNumResults() != 1) {
        builder.clone(op, mapping);
        continue;
      }
      Value origResult = op.getResult(0);
      Type resTy = origResult.getType();

      if (!resTy.isIntOrFloat()) {
        // Index, memref, etc. — clone with mapping unchanged.
        Operation *cloned = builder.clone(op, mapping);
        mapping.map(origResult, cloned->getResult(0));
        continue;
      }

      // Determine target sub-width for the result, clamped to L_strip.
      // Same reasoning as getLnForAccess: when T < R_T, L_strip < R_T and
      // using the unclamped R_T yields numSubOpsResult = 0.
      //
      // Special case for i1 (boolean) results: arith.cmpi / arith.cmpf return
      // a mask type whose width matches the input operand lane count — one bit
      // per lane, not one register per byte. Using bitWidth/8 = 0 here would
      // make getRegisterLanesForType return 1 (the elementBytes=0 fallback),
      // giving Ln_result=1 and subResTy=vector<1xi1> which is wrong.
      // Fix: treat i1 as 1-bit-per-lane, use L_strip directly.
      int64_t resBytes = resTy.getIntOrFloatBitWidth() / 8;
      int64_t Ln_result;
      if (resBytes == 0) {
        // 1-bit result (i1 from cmpi/cmpf): one mask bit per lane.
        Ln_result = L_strip;
      } else {
        Ln_result = std::min(getRegisterLanesForType(target, resBytes),
                             L_strip);
      }
      int64_t numSubOpsResult = L_strip / Ln_result;

      // Build per-operand sub-vector lists aligned to Ln_result.
      SmallVector<SmallVector<Value>> operandSubs;
      bool sizingFailed = false;
      for (Value operand : op.getOperands()) {
        auto subs = getSubsFor(operand);
        SmallVector<Value> sized;

        if ((int64_t)subs.size() == numSubOpsResult) {
          // Already the right number of pieces.
          sized = subs;
        } else if (subs.size() == 1) {
          Value v = subs[0];
          auto vecTy = dyn_cast<VectorType>(v.getType());
          if (!vecTy) {
            // Scalar — replicate for all sub-ops.
            for (int64_t j = 0; j < numSubOpsResult; ++j) sized.push_back(v);
          } else {
            int64_t srcW = vecTy.getShape()[0];
            if (srcW == Ln_result) {
              sized.push_back(v);
            } else if (srcW > Ln_result) {
              // Wider vector — slice into Ln_result-wide pieces.
              for (int64_t j = 0; j < numSubOpsResult; ++j) {
                Value piece = vector::ExtractStridedSliceOp::create(
                    builder, loc, v,
                    /*offsets=*/ArrayRef<int64_t>{j * Ln_result},
                    /*sizes=*/ArrayRef<int64_t>{Ln_result},
                    /*strides=*/ArrayRef<int64_t>{1});
                sized.push_back(piece);
              }
            } else {
              // srcW < Ln_result: narrower than result (should not happen in
              // well-formed Tier-A kernels; fall back to scalar clone).
              sizingFailed = true;
              break;
            }
          }
        } else {
          // Multi-piece operand list but wrong count: unsupported in v1.
          sizingFailed = true;
          break;
        }
        operandSubs.push_back(std::move(sized));
      }

      if (sizingFailed) {
        // Conservative fallback: clone scalar.
        Operation *cloned = builder.clone(op, mapping);
        mapping.map(origResult, cloned->getResult(0));
        continue;
      }

      // Emit numSubOpsResult instances of the op, each at sub-vector width.
      Type subResTy = VectorType::get({Ln_result}, resTy);
      SmallVector<Value> resultSubs;
      resultSubs.reserve(numSubOpsResult);

      // Fast-math flag injection for floating-point arith ops on vector types.
      //
      // When the original scalar op is arith.mulf or arith.addf and does NOT
      // already carry a fastmath attribute, inject `fastmath<contract>`.
      // The `contract` flag is the minimal flag needed for FMA fusion: it allows
      // the backend to contract fmul(a,b)+c → vfmadd213ps without enabling
      // the full semantics of -ffast-math (no-nan, no-inf, etc.).
      //
      // This matches what Clang emits when `-ffast-math` is passed: the LLVM IR
      // carries `fmul fast` / `fadd fast` on the vectorised ops, which enables
      // `contract` (among other flags).  Without `contract`, LLVM cannot fuse
      // a separate vmulps + vaddps into a single vfmadd213ps, costing one extra
      // 512-bit operation per 16-float group.
      //
      // Safety: `contract` does NOT change rounding mode or introduce non-IEEE
      // results; it only allows reordering of the multiply-add into a fused op
      // which is IEEE-correct when the FMA hardware round occurs once.
      // We only inject the flag when:
      //   (a) the result is a float or vector-of-float type (not integer), and
      //   (b) the op is arith.mulf or arith.addf (the specific ops that fuse), and
      //   (c) no fastmath attr is already present on the original op.
      //
      // Note: we do NOT add fast-math to arith.divf or arith.subf by default —
      // division already has ffast-math implications, and subf is not fusable.
      // Fast-math flag injection for floating-point arith ops on vector types.
      //
      // When the original scalar op is arith.mulf or arith.addf and does NOT
      // already carry a fastmath attribute, inject `fastmath<contract>`.
      // The `contract` flag is the minimal flag needed for FMA fusion: it allows
      // the backend to contract fmul(a,b)+c → vfmadd213ps without enabling
      // the full semantics of -ffast-math (no-nan, no-inf, etc.).
      //
      // This matches what Clang emits when `-ffast-math` is passed: the LLVM IR
      // carries `fmul fast` / `fadd fast` on the vectorised ops, which enables
      // `contract` (among other flags).  Without `contract`, LLVM cannot fuse
      // a separate vmulps + vaddps into a single vfmadd213ps, costing one extra
      // 512-bit operation per 16-float group.
      //
      // Safety: `contract` does NOT change rounding mode or introduce non-IEEE
      // results; it only allows reordering of the multiply-add into a fused op
      // which is IEEE-correct when the FMA hardware round occurs once.
      // We only inject the flag when:
      //   (a) the result is a float or vector-of-float type (not integer), and
      //   (b) the op is arith.mulf or arith.addf (the specific ops that fuse), and
      //   (c) the existing fastmath flags do NOT already include contract.
      //
      // Note: MLIR's canonicalizer adds `fastmath<none>` to arith.mulf/addf
      // automatically, so we cannot use `!op.hasAttr("fastmath")` — instead we
      // check whether the existing flags already include the `contract` bit.
      //
      // Note: we do NOT add fast-math to arith.divf or arith.subf by default —
      // division already has ffast-math implications, and subf is not fusable.
      bool injectContractFMF = false;
      if (mlir::isa<FloatType, VectorType>(resTy)) {
        if (isa<arith::MulFOp, arith::AddFOp>(op)) {
          // Check if contract is already present in the existing flags.
          auto existingFMF = arith::FastMathFlags::none;
          if (auto fmfAttr = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
            existingFMF = fmfAttr.getValue();
          bool hasContract = static_cast<bool>(
              existingFMF & arith::FastMathFlags::contract);
          if (!hasContract)
            injectContractFMF = true;
        }
      }

      // Pre-compute the merged fastmath attr to use for the cloned sub-ops.
      // We do this once (before the j-loop) and override the fastmath attr in
      // the OperationState after addAttributes() copies the original attrs.
      // This avoids duplicate-key issues: addAttributes() copies fastmath<none>
      // (or whatever existing flags), and then we overwrite with the merged value.
      Attribute mergedFMFAttr;
      if (injectContractFMF) {
        MLIRContext *ctx = builder.getContext();
        auto existingFMF = arith::FastMathFlags::none;
        if (auto fmfAttr = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
          existingFMF = fmfAttr.getValue();
        auto merged = existingFMF | arith::FastMathFlags::contract;
        mergedFMFAttr = arith::FastMathFlagsAttr::get(ctx, merged);
      }

      for (int64_t j = 0; j < numSubOpsResult; ++j) {
        SmallVector<Value> opOperands;
        for (auto &operandList : operandSubs) opOperands.push_back(operandList[j]);
        OperationState state(loc, op.getName());
        state.addOperands(opOperands);
        state.addAttributes(op.getAttrs());
        state.addTypes(subResTy);
        // Inject/merge contract fast-math flag for mulf/addf.
        // We use NamedAttrList::set() which overwrites any existing "fastmath"
        // entry added by addAttributes() above, preventing duplicate-key errors.
        if (mergedFMFAttr) {
          state.attributes.set(
              mlir::StringAttr::get(builder.getContext(), "fastmath"),
              mergedFMFAttr);
        }
        Operation *newOp = builder.create(state);
        resultSubs.push_back(newOp->getResult(0));
      }
      // Map first sub-vector for backward compat with scalar consumers.
      mapping.map(origResult, resultSubs[0]);
      subVectorMap[origResult] = std::move(resultSubs);
    }
  }
}

/// Clone the entire original body verbatim into the tail loop (scalar
/// fallback for the (trip mod L) remaining iterations).
static void emitTailBody(scf::ForOp tailLoop, scf::ForOp origLoop,
                         OpBuilder &builder) {
  builder.setInsertionPointToStart(tailLoop.getBody());
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), tailLoop.getInductionVar());
  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    builder.clone(op, mapping);
  }
}

class LegoVectorizePass
    : public mlir::lego::impl::LegoVectorizePassBase<LegoVectorizePass> {
 public:
  using mlir::lego::impl::LegoVectorizePassBase<
      LegoVectorizePass>::LegoVectorizePassBase;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    auto loops = collectCandidateLoops(func);

    // Collect which loops to erase after the transform loop (cannot erase
    // while iterating if the list contains nested loops, but Task 8 handles
    // only flat single-level loops that are not nested — safe to erase
    // immediately after body population since we iterated by value).
    llvm::SmallVector<scf::ForOp> toErase;

    for (auto &a : loops) {
      Value iv = a.forOp.getInductionVar();
      a.classes.reserve(a.accesses.size());

      // R15: target ISA is now threaded through the tablegen pass option `target`
      // (default "avx512").  The lego-to-x86-vector pipeline passes "avx512" or
      // "avx2"; the lego-to-arm-neon pipeline passes "neon".
      // All lane-width decisions use getRegisterLanesForType(target, elemBytes) —
      // NO AVX-specific constants are hardcoded below this line.
      llvm::StringRef target = this->target;

      for (Operation *op : a.accesses) {
        // Determine elementBytes from the op's element type (default 8 = f64).
        int64_t elemBytes = 8;
        Type t;
        if (auto load = dyn_cast<memref::LoadOp>(op)) t = load.getType();
        else if (auto store = dyn_cast<memref::StoreOp>(op))
          t = store.getValue().getType();
        if (t && t.isIntOrFloat()) elemBytes = t.getIntOrFloatBitWidth() / 8;

        auto cls = lego::solveAccessTierA(op, iv, elemBytes);
        if (cls.kind == lego::AccessKind::NonAffine) {
          // Tier B fallback: speculative unroll at concrete iv values 0..L-1.
          int64_t L_probe = getRegisterLanesForType(target, elemBytes);
          cls = lego::solveAccessTierB(op, iv, elemBytes, L_probe);
        }
        a.classes.push_back(cls);
      }

      a.L_strip = computeStripMineFactor(a, target);

      // Task 8: strip-mine + emit vector.transfer_read/write.
      if (a.L_strip <= 1) continue;

      // Body whitelist: each rejection below is DOCUMENTED as either a
      // fundamental limitation or an unimplemented extension (roadmap item).
      //
      // ALLOWED: memref.load, memref.store, scf.yield — the core memory ops.
      // ALLOWED: arith dialect ops (add, mul, sub, div, etc.) — vectorized via
      //          the sub-vector catch-all in emitVectorBody.
      // ALLOWED (R16): arith.cmpi, arith.cmpf, arith.select — now handled by
      //          dedicated emit paths in emitVectorBody.
      // ALLOWED (R17): scf.if with a single then-branch (no else) where the
      //          then-block contains only memref.store ops — predicated maskedstore.
      //
      // GUARDED (must-skip) cases:
      //
      // [G1] arith.cmpi / arith.cmpf on index-typed operands.
      //   STATUS: RESOLVED (R16). cmpi/cmpf on index operands are now lowered
      //   by converting index operands to i64 vectors in emitVectorBody, then
      //   emitting a vector<L x i1> result for the comparison.  arith.select
      //   on index-typed results is also handled by lowering to i64.
      //   Legacy guard removed — all cmpi/cmpf/select now allowed.
      //
      // [G2] arith.index_cast / arith.index_castui with non-index source/result
      //   when IV-dependent.
      //   STATUS: PRINCIPLED SKIP — not a missing feature but a correctness guard.
      //   REASON: index_cast(vector<16xi32>) → index is not a valid MLIR type;
      //   the catch-all vectorizer would silently produce broken IR.  The NonAffine
      //   gather path (cloneAddrDAG) handles Morton-style index chains correctly
      //   ONLY for pure-address-compute paths where the entire chain is cloned
      //   per-lane.  The bodyOK guard ensures the catch-all path never sees an
      //   IV-dependent index_cast that it can't lower.  If index_cast were
      //   hoistable (IV-independent), the allInvariant path in evalAffine already
      //   handles it correctly without any bodyOK skip.
      //
      // [G3] Any op NOT in the arith dialect and NOT a memref load/store/yield/scf.if.
      //   STATUS: PRINCIPLED SKIP — conservative allowlist.
      //   REASON: Unknown ops may have side effects, may require dialect-specific
      //   vectorization support, or may use types (e.g. memref.subview's result
      //   types) that the sub-vector catch-all cannot handle.  Widening the
      //   allowlist is safe to do per-dialect as vectorization support is added.
      //
      // [G4] scf.if with an else branch or with non-store body ops.
      //   STATUS: PARTIAL (R17). Only single-branch scf.if whose then-block
      //   contains only memref.store ops is allowed; this covers the
      //   "conditional store" pattern (predicated maskedstore).  Scf.if with
      //   an else branch or compute ops in the then-block is still conservatively
      //   rejected (too complex for a single emission pass).
      bool bodyOK = true;
      for (Operation &op : a.forOp.getBody()->getOperations()) {
        if (isa<memref::LoadOp, memref::StoreOp, scf::YieldOp>(op)) continue;
        if (op.getDialect() &&
            isa<arith::ArithDialect>(op.getDialect())) {
          // R16: arith.cmpi / arith.cmpf / arith.select are now handled.
          // No longer rejected — the emit path for index-typed operands converts
          // to i64 before building vector comparisons.
          //
          // [G2] Reject IV-dependent arith.index_cast with any index source or result.
          //
          // Two problems exist:
          //   (a) index_cast(i32 → index): result would be vector<L x index>, invalid MLIR.
          //   (b) index_cast(index → i32): source is scalar index; the catch-all would emit
          //       index_cast(index → vector<L x i32>), which has mismatched shapes and is
          //       an invalid operation.
          //
          // Both cases require a dedicated per-lane emit path (clone the index_cast once
          // per lane with a lane-specific index). This is not implemented in v1; it would
          // require the same cloneAddrDAG technique used for NonAffine gathers.
          //
          // SAFE cases NOT rejected:
          //   - index_cast(i32 → i64) or (i64 → i32): no index type, caught elsewhere.
          //   - IV-independent index_cast: operand is loop-invariant, broadcast pre-pass
          //     handles it correctly (scalar → broadcast).
          if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op)) {
            bool srcIsIndex = op.getOperand(0).getType().isIndex();
            bool resIsIndex = op.getResult(0).getType().isIndex();
            if (srcIsIndex || resIsIndex) {
              bool ivDependent = llvm::any_of(op.getOperands(), [&](Value v) {
                if (v == a.forOp.getInductionVar()) return true;
                Operation *defOp = v.getDefiningOp();
                return defOp && a.forOp->isAncestor(defOp);
              });
              if (ivDependent) {
                bodyOK = false;
                break;
              }
            }
          }
          continue;
        }
        // R17: allow scf.if with a single then-branch that only stores.
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
          // Reject if there is an else branch (too complex for v1 maskedstore).
          if (ifOp.getNumRegions() > 1 && !ifOp.getElseRegion().empty()) {
            bodyOK = false;
            break;
          }
          // Verify the then-block only contains stores + yield.
          bool thenOK = true;
          for (Operation &thenOp : ifOp.getThenRegion().front().getOperations()) {
            if (isa<memref::StoreOp, scf::YieldOp>(thenOp)) continue;
            thenOK = false;
            break;
          }
          if (!thenOK) {
            bodyOK = false;
            break;
          }
          continue;
        }
        // [G3] Op outside the arith dialect and not an allowed scf.if.
        bodyOK = false;
        break;
      }
      if (!bodyOK) continue;

      OpBuilder builder(a.forOp);
      StripMineResult mined = stripMineForOp(a.forOp, a.L_strip, builder);
      emitVectorBody(mined.vecLoop, a.forOp, a.L_strip,
                     a.accesses, a.classes, target, builder);
      emitTailBody(mined.tailLoop, a.forOp, builder);
      toErase.push_back(a.forOp);
    }

    // Erase original loops after all transforms are complete.
    for (scf::ForOp op : toErase)
      op.erase();
  }
};

}  // namespace

namespace mlir::lego {

std::unique_ptr<Pass> createLegoVectorizePass() {
  return std::make_unique<LegoVectorizePass>();
}

/// Overload that sets the target option (avx512 | avx2 | neon).
std::unique_ptr<Pass> createLegoVectorizePass(llvm::StringRef target) {
  LegoVectorizePassOptions opts;
  opts.target = std::string(target);
  return std::make_unique<LegoVectorizePass>(opts);
}

}  // namespace mlir::lego
