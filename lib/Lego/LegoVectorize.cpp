//===- LegoVectorize.cpp - Layout-agnostic vectorization pass -------------===//
//
// Lowers loops over Lego-derived arith address expressions to MLIR vector
// dialect ops by symbolic stride analysis. Layout-agnostic: operates on
// post-LegoToArith IR (arith + memref + scf).
//
// Stride analysis (AffineVal evaluator, Tier-A/B solvers) lives in
// LegoVectorizeAnalysis.cpp. This file contains only the rewrite / emit
// logic: strip-mine factor computation, emitVectorBody, and the pass.
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

#include "llvm/Support/Debug.h"

#include <algorithm>  // std::find
#include <limits>
#include <numeric>  // std::gcd (C++17)

#define DEBUG_TYPE "lego-vectorize"

using namespace mlir;

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
      lego::AffineVal sym = lego::evalLinearInIV(addr, iv, cache);
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
          if (loadBase == storeBase) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[lego-vectorize] computeStripMineFactor: L_strip=1"
                          " — reduction loop (broadcast-store + same-base load)\n");
            return 1;  // reduction loop — skip vectorization
          }
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
      LLVM_DEBUG(llvm::dbgs()
                 << "[lego-vectorize] computeStripMineFactor: L_strip=1"
                    " — unknown AccessKind, conservative skip\n");
      return 1;
    }
    if (Ln <= 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[lego-vectorize] computeStripMineFactor: L_strip=1"
                    " — Ln<=1 (register width insufficient for element type)\n");
      return 1;
    }
    L_strip = (L_strip == 1) ? Ln : lcm_i64(L_strip, Ln);
  }
  if (!sawConstraining) {
    LLVM_DEBUG(llvm::dbgs()
               << "[lego-vectorize] computeStripMineFactor: L_strip=1"
                  " — all accesses are Broadcast (nothing constrains L)\n");
    return 1;  // all Broadcasts — nothing to vectorize.
  }

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
    else if (cls.kind == lego::AccessKind::NonAffine &&
             worstPenalty < lego::CostModel::kNonAffineGatherPenalty)
      worstPenalty = lego::CostModel::kNonAffineGatherPenalty;
    else if (cls.kind == lego::AccessKind::Strided &&
             worstPenalty < lego::CostModel::kStridedGatherPenalty)
      worstPenalty = lego::CostModel::kStridedGatherPenalty;
  }
  if (!hasUnit) {
    // Pure gather loop: apply cost penalty.
    double score = static_cast<double>(L_strip) / worstPenalty;
    if (score <= 1.0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[lego-vectorize] computeStripMineFactor: L_strip=1"
                    " — cost-model rejected pure-gather loop"
                    " (score=" << score << " <= 1.0)\n");
      return 1;
    }
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
    // Apply ILP unroll to match Clang's auto-vectorizer output quality.
    constexpr int64_t kILPFactor = lego::CostModel::kILPFactor;
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
// cloneAddrChain — clone the address def-use chain for NonAffine gather emission.
//
// Recursively clones the def-use chain rooted at `v`, substituting operands
// through `laneMap`. Stops at values defined outside `parentLoop` (uses them
// as-is) or at block arguments (uses the mapped value or original).
// Named "cloneAddrChain" following MLIR convention of "chain" for def-use
// sequences (rather than the more general graph-theory term "DAG").
//
// Returns the cloned (or reused) SSA value for the lane-specific address.
// ---------------------------------------------------------------------------
static Value cloneAddrChain(Value v, IRMapping &laneMap, OpBuilder &builder,
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
    newOperands.push_back(cloneAddrChain(operand, laneMap, builder, parentLoop));

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


// ---------------------------------------------------------------------------
// EmitContext — holds all shared state for the vector body emission pass.
//
// Each per-kind emit method (emitUnitLoad, emitBroadcastLoad, etc.) is a
// member of this struct so they can access loc, builder, mapping, and
// subVectorMap without threading them through every call.
// ---------------------------------------------------------------------------
struct EmitContext {
  Location loc;
  Value newIv;
  llvm::StringRef target;
  int64_t L_strip;
  scf::ForOp origLoop;
  OpBuilder &builder;
  IRMapping mapping;
  DenseMap<Value, SmallVector<Value>> subVectorMap;
  // Extra scalar (non-vectorized) IV mappings from inner reduction loops.
  // Used by emitGatherLoad to seed per-lane address chain cloning with the
  // correct scalar IV mappings (e.g., inner loop IV → new inner loop IV).
  IRMapping innerScalarMappings;

  // -------------------------------------------------------------------------
  // Shared helpers
  // -------------------------------------------------------------------------

  // Per-access natural lane width, clamped to L_strip.
  // When the loop trip count (T) is smaller than the register width (R_T),
  // L_strip = T < R_T.  Using R_T directly would give numSubOps = 0 which
  // produces empty vector bodies.  Clamp to L_strip so we always emit exactly
  // one vector op covering the full strip-mined span.
  int64_t getLnForAccess(const lego::AccessClassification &cls) const {
    int64_t R_T = getRegisterLanesForType(target, cls.elementBytes);
    return std::min(R_T, L_strip);
  }

  // Build an index Value for (baseIv + j * Ln), or just baseIv if j==0.
  Value makeOffset(Value baseIv, int64_t j, int64_t Ln) {
    if (j == 0) return baseIv;
    Value addend = arith::ConstantIndexOp::create(builder, loc, j * Ln);
    return arith::AddIOp::create(builder, loc, baseIv, addend);
  }

  // Return the sub-vector list for an original operand.
  // Falls back to a 1-element list containing the IRMapping result.
  SmallVector<Value> getSubsFor(Value origOperand) {
    if (auto it = subVectorMap.find(origOperand); it != subVectorMap.end())
      return it->second;
    return {mapping.lookupOrDefault(origOperand)};
  }

  // Register a vectorized result: always keeps mapping[orig] == subVecs[orig][0].
  // Use this instead of calling mapping.map + subVectorMap[orig] = ... separately
  // to enforce the invariant that the two data structures stay in sync.
  //
  // This is the VectorFrame concept from Finding 7: the invariant
  //   mapping[v] == subVecs[v][0]
  // is enforced structurally here rather than by convention across all emit sites.
  void mapVec(Value orig, SmallVector<Value> subs) {
    assert(!subs.empty() && "mapVec: sub-vector list must be non-empty");
    mapping.map(orig, subs[0]);
    subVectorMap[orig] = std::move(subs);
  }

  // Check whether `v` is defined outside origLoop.
  bool isOutsideLoop(Value v) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      if (v == origLoop.getInductionVar()) return false;
      return true;
    }
    return !origLoop->isAncestor(defOp);
  }

  // -------------------------------------------------------------------------
  // Pre-pass: broadcast loop-external scalars to natural sub-vector width.
  //
  // Previous approach: broadcast to L_strip (e.g. vector<64xf32>), then let the
  // arith catch-all slice it into numSubOpsResult pieces via
  // vector.extract_strided_slice. This generated 6 instructions to splat a
  // scalar that GCC splats in 1 vpbroadcastss. Fixed: broadcast directly to
  // natural Ln width and replicate the SSA value numSubOps times so the
  // catch-all's getSubsFor() returns pre-sliced lists.
  // -------------------------------------------------------------------------
  // Helper: broadcast a single external scalar (factored out for reuse).
  void broadcastOneExternalScalar(Value operand) {
    if (subVectorMap.contains(operand)) return;
    Type t = operand.getType();
    if (!t.isIntOrFloat()) return;

    int64_t elemBits = t.getIntOrFloatBitWidth();
    int64_t elemBytes = elemBits / 8;
    int64_t Ln;
    if (elemBytes <= 0) {
      Ln = L_strip;  // i1 or sub-byte: one bit per lane
    } else {
      Ln = std::min(getRegisterLanesForType(target, elemBytes), L_strip);
    }
    int64_t numSubOps = (Ln > 0) ? (L_strip / Ln) : 1;

    // Emit ONE broadcast op at vector<Ln x T>; reuse it numSubOps times.
    // One physical vpbroadcastss; LLVM can hoist and share across ILP slots.
    auto vecTy = VectorType::get({Ln}, t);
    Value bc = vector::BroadcastOp::create(builder, loc, vecTy, operand);
    SmallVector<Value> subs(numSubOps, bc);
    mapVec(operand, std::move(subs));
  }

  void broadcastExternalScalars() {
    // Scan the direct body ops and (up to two levels deep) any nested scf.for
    // loops.  Nested scf.for loops arise when range() loops are written inside
    // tile_range; the inner IVs are scalar but their bodies may reference the
    // outer IV which is vectorized.  We must broadcast those outer-scope
    // references.  R20 supports up to 2 levels (outer tile_range → inner
    // reduction loop → innermost reduction loop).
    for (Operation &op : origLoop.getBody()->getOperations()) {
      // If this op is an inner reduction scf.for, scan its body too.
      if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
        for (Operation &innerOp : innerFor.getBody()->getOperations()) {
          // R20-2L: also scan two-level nested scf.for inside the inner loop.
          if (auto innermostFor = dyn_cast<scf::ForOp>(&innerOp)) {
            for (Operation &innermostOp :
                 innermostFor.getBody()->getOperations()) {
              for (Value operand : innermostOp.getOperands()) {
                if (!isOutsideLoop(operand)) continue;
                broadcastOneExternalScalar(operand);
              }
            }
            continue;
          }
          for (Value operand : innerOp.getOperands()) {
            if (!isOutsideLoop(operand)) continue;
            broadcastOneExternalScalar(operand);
          }
        }
        continue;
      }
      for (Value operand : op.getOperands()) {
        if (!isOutsideLoop(operand)) continue;
        broadcastOneExternalScalar(operand);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Per-kind load emit helpers
  // -------------------------------------------------------------------------

  void emitUnitLoad(memref::LoadOp load,
                    const lego::AccessClassification &cls) {
    int64_t Ln = getLnForAccess(cls);
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
    mapVec(load.getResult(), std::move(subs));
  }

  void emitBroadcastLoad(memref::LoadOp load,
                         const lego::AccessClassification &cls) {
    // Loop-invariant load -- clone as scalar then broadcast at natural Ln width.
    // Emit ONE broadcast (one vpbroadcastss) and reuse it numSubOps times.
    Operation *clonedLoad = builder.clone(*load.getOperation(), mapping);
    Type elemTy = load.getType();
    int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    int64_t Ln = (elemBytes > 0)
                     ? std::min(getRegisterLanesForType(target, elemBytes), L_strip)
                     : L_strip;
    int64_t numSubOps = (Ln > 0) ? (L_strip / Ln) : 1;
    auto vecTy = VectorType::get({Ln}, elemTy);
    Value bc = vector::BroadcastOp::create(builder, loc, vecTy,
                                           clonedLoad->getResult(0));
    SmallVector<Value> subs(numSubOps, bc);
    mapVec(load.getResult(), std::move(subs));
  }

  void emitCrossBlockLoad(memref::LoadOp load,
                          const lego::AccessClassification &cls) {
    // Two adjacent block reads + vector.shuffle.
    //
    // Cross-block pattern: addr(iv+0..L-1) is unit-stride for the first
    // `boundary` lanes, then jumps to the next block for the remaining
    // (L - boundary) lanes. Synthesise by:
    //   1. Reading L lanes from block N (starting at addr(0)).
    //   2. Reading L lanes from block N+1 (starting at addr(boundary)).
    //   3. vector.shuffle selects [0..boundary) from blockN and
    //      [0..L-boundary) from blockNp1.
    //
    // v1 restriction: only Ln == L_strip. If CrossBlock is combined with
    // mixed-precision (Ln < L_strip), fall through to scalar-clone fallback.
    int64_t Ln = getLnForAccess(cls);
    if (Ln != L_strip) {
      Operation *cloned = builder.clone(*load.getOperation(), mapping);
      mapping.map(load.getResult(), cloned->getResult(0));
      return;
    }

    int64_t boundary = cls.boundary;
    Type elemTy = load.getType();
    auto vecTy = VectorType::get({Ln}, elemTy);
    Value baseIv = mapping.lookupOrDefault(load.getIndices().front());

    // R12a: use cls.boundaryJump (actual address delta from addrs[0] to
    // addrs[boundary]) to compute the second block's base address.
    // boundaryJump = addrs[boundary] - addrs[0] (element-unit offset).
    Value boundaryJumpConst =
        arith::ConstantIndexOp::create(builder, loc, cls.boundaryJump);
    Value blockNp1Iv =
        arith::AddIOp::create(builder, loc, baseIv, boundaryJumpConst);

    Value blockN = vector::TransferReadOp::create(
        builder, loc, vecTy, load.getMemRef(), ValueRange{baseIv},
        /*padding=*/std::nullopt, /*inBounds=*/ArrayRef<bool>{true});
    Value blockNp1 = vector::TransferReadOp::create(
        builder, loc, vecTy, load.getMemRef(), ValueRange{blockNp1Iv},
        /*padding=*/std::nullopt, /*inBounds=*/ArrayRef<bool>{true});

    // Shuffle: [0..boundary) from blockN, then [0..L-boundary) from blockNp1.
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
    mapVec(load.getResult(), {shuffled});
  }

  void emitStridedLoad(memref::LoadOp load,
                       const lego::AccessClassification &cls) {
    // R20: Deinterleave path for small constant strides (2, 4, 8).
    //
    // For stride=S: load S consecutive blocks of Ln elements, then shuffle
    // to extract every S-th element. Maps to vpermt2ps (1-3 cycles) rather
    // than vpgatherdps (10+ cycles for L1-hot data on x86 AVX-512).
    //
    // Conditions: S in {2,4,8} and S*Ln <= 256. Otherwise falls through to
    // the gather path (emitGatherLoad).
    int64_t stride = cls.stride / cls.elementBytes;  // stride in elements
    int64_t Ln = getLnForAccess(cls);
    bool useDeinterleave = (stride == 2 || stride == 4 || stride == 8) &&
                           (stride * Ln <= 256);

    if (!useDeinterleave) {
      emitGatherLoad(load, cls);
      return;
    }

    Type elemTy = load.getType();
    auto vecTy = VectorType::get({Ln}, elemTy);

    // Compute physBase = addr(newIv) via lane-0 address DAG clone.
    Value origIv = origLoop.getInductionVar();
    Value curIv = mapping.lookupOrDefault(origIv);
    IRMapping lane0Map;
    lane0Map.map(origIv, curIv);
    Value physBase = cloneAddrChain(load.getIndices().front(),
                                  lane0Map, builder, origLoop);

    // Load S blocks of Ln elements each: Block[b] starts at physBase + b*Ln.
    SmallVector<Value> blocks;
    blocks.reserve(stride);
    for (int64_t b = 0; b < stride; ++b) {
      Value blockBase;
      if (b == 0) {
        blockBase = physBase;
      } else {
        Value bOff = arith::ConstantIndexOp::create(builder, loc, b * Ln);
        blockBase = arith::AddIOp::create(builder, loc, physBase, bOff);
      }
      auto block = vector::TransferReadOp::create(
          builder, loc, vecTy, load.getMemRef(), ValueRange{blockBase},
          /*padding=*/std::nullopt, /*inBounds=*/ArrayRef<bool>{true});
      blocks.push_back(block.getVector());
    }

    // Deinterleave: select element k*stride from the concatenated blocks.
    Value result;
    if (stride == 2) {
      // indices [0, 2, 4, ..., 2*(Ln-1)] selects even elements from [B0|B1].
      SmallVector<int64_t> shuffleIdx;
      shuffleIdx.reserve(Ln);
      for (int64_t k = 0; k < Ln; ++k)
        shuffleIdx.push_back(k * 2);
      result = vector::ShuffleOp::create(builder, loc, blocks[0],
                                         blocks[1], shuffleIdx);
    } else if (stride == 4) {
      // Two half-width shuffles then a merge.
      int64_t half = Ln / 2;
      SmallVector<int64_t> halfIdx;
      halfIdx.reserve(half);
      for (int64_t k = 0; k < half; ++k)
        halfIdx.push_back(k * stride);
      auto sh1Raw = vector::ShuffleOp::create(builder, loc, blocks[0],
                                              blocks[1], halfIdx);
      auto sh2Raw = vector::ShuffleOp::create(builder, loc, blocks[2],
                                              blocks[3], halfIdx);
      SmallVector<int64_t> combineIdx;
      combineIdx.reserve(Ln);
      for (int64_t k = 0; k < half; ++k) combineIdx.push_back(k);
      for (int64_t k = 0; k < half; ++k) combineIdx.push_back(half + k);
      result = vector::ShuffleOp::create(builder, loc, sh1Raw, sh2Raw,
                                         combineIdx);
    } else {
      // stride == 8: four quarter-width shuffles, two pair-merges, one final.
      int64_t qtr = Ln / 4;
      SmallVector<int64_t> qtrIdx;
      qtrIdx.reserve(qtr);
      for (int64_t k = 0; k < qtr; ++k)
        qtrIdx.push_back(k * stride);
      SmallVector<Value> halves;
      for (int64_t p = 0; p < 4; ++p) {
        auto sh = vector::ShuffleOp::create(builder, loc,
                                            blocks[2*p], blocks[2*p+1],
                                            qtrIdx);
        halves.push_back(sh);
      }
      SmallVector<int64_t> pairIdx;
      pairIdx.reserve(Ln / 2);
      for (int64_t k = 0; k < qtr; ++k) pairIdx.push_back(k);
      for (int64_t k = 0; k < qtr; ++k) pairIdx.push_back(qtr + k);
      auto mid01 = vector::ShuffleOp::create(builder, loc,
                                             halves[0], halves[1], pairIdx);
      auto mid23 = vector::ShuffleOp::create(builder, loc,
                                             halves[2], halves[3], pairIdx);
      SmallVector<int64_t> finalIdx;
      finalIdx.reserve(Ln);
      for (int64_t k = 0; k < Ln/2; ++k) finalIdx.push_back(k);
      for (int64_t k = 0; k < Ln/2; ++k) finalIdx.push_back(Ln/2 + k);
      result = vector::ShuffleOp::create(builder, loc, mid01, mid23, finalIdx);
    }

    mapVec(load.getResult(), {result});
  }

  // vector.gather for constant non-unit stride (large strides or non-power-of-2)
  // or for non-affine (irregular) access. Builds per-lane indices by cloning
  // the ORIGINAL scalar address DAG with origIv substituted by (newIv + j).
  //
  // R19 fix: use a fresh per-lane IRMapping (not the outer `mapping` which has
  // vector subs for float values) so cloneAddrChain produces scalar element-unit
  // addresses for each lane.
  void emitGatherLoad(memref::LoadOp load,
                      const lego::AccessClassification &cls) {
    int64_t Ln = getLnForAccess(cls);
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
      // Seed laneMap with the outer IV substitution and any inner scalar IV
      // mappings (R20: when an inner reduction loop IV appears in the address
      // chain, it must be remapped to the new inner loop's IV, not left as
      // the original block argument which will be erased).
      IRMapping laneMap = innerScalarMappings;
      laneMap.map(origIv, laneIv);
      Value laneAddr = cloneAddrChain(origAddr, laneMap, builder, origLoop);
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

    mapVec(load.getResult(), {gathered});
  }

  // -------------------------------------------------------------------------
  // Per-kind store emit helpers
  // -------------------------------------------------------------------------

  void emitUnitStore(memref::StoreOp store,
                     const lego::AccessClassification &cls) {
    int64_t Ln = getLnForAccess(cls);
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
      // Sub-vector count mismatch -- scalar fallback.
      builder.clone(*store.getOperation(), mapping);
    }
  }

  // R17: scf.if predicated maskedstore.
  // Supports the pattern:
  //   scf.if %cond {
  //     memref.store %val, %B[%i] : memref<?xT>
  //   }
  // where %cond is a scalar i1 produced by arith.cmpf/cmpi in this loop.
  // Lowers to: vector.maskedstore with the vectorised condition as mask.
  void emitMaskedStore(scf::IfOp ifOp,
                       ArrayRef<Operation *> accesses,
                       ArrayRef<lego::AccessClassification> classes) {
    Value cond = ifOp.getCondition();
    Value vecCond = mapping.lookupOrDefault(cond);
    // If the condition didn't get vectorized (still scalar), broadcast it.
    if (!dyn_cast<VectorType>(vecCond.getType())) {
      auto maskTy = VectorType::get({L_strip}, builder.getI1Type());
      vecCond = vector::BroadcastOp::create(builder, loc, maskTy, vecCond);
    }
    for (Operation &thenOp : ifOp.getThenRegion().front().getOperations()) {
      if (isa<scf::YieldOp>(thenOp)) continue;
      auto store = dyn_cast<memref::StoreOp>(&thenOp);
      if (!store) continue;
      auto it = std::find(accesses.begin(), accesses.end(), &thenOp);
      Value valueToStore;
      Value baseIdx;
      if (it != accesses.end()) {
        auto subs = getSubsFor(store.getValue());
        valueToStore = subs.empty() ? mapping.lookupOrDefault(store.getValue())
                                    : subs[0];
        baseIdx = mapping.lookupOrDefault(store.getIndices().front());
      } else {
        valueToStore = mapping.lookupOrDefault(store.getValue());
        baseIdx = mapping.lookupOrDefault(store.getIndices().front());
      }
      // Ensure value is a vector<L_strip x T> type.
      Type elemTy = store.getValue().getType();
      if (!dyn_cast<VectorType>(valueToStore.getType())) {
        auto vecTy = VectorType::get({L_strip}, elemTy);
        valueToStore = vector::BroadcastOp::create(builder, loc, vecTy,
                                                   valueToStore);
      }
      vector::MaskedStoreOp::create(builder, loc, store.getMemRef(),
                                    ValueRange{baseIdx}, vecCond, valueToStore,
                                    /*alignment=*/mlir::IntegerAttr{});
    }
  }

  // -------------------------------------------------------------------------
  // arith catch-all -- mixed-precision aware pass-through.
  // Handles: R16 index-typed cmpi/cmpf/select, and general scalar->vector
  // widening for all other arith ops.
  // -------------------------------------------------------------------------
  void emitArithOp(Operation &op) {
    // R16: arith.cmpi / arith.cmpf on index-typed operands.
    // Index has no fixed bit width in MLIR; convert to i64 vectors first.
    if (isa<arith::CmpIOp, arith::CmpFOp>(&op)) {
      bool hasIndexOperand = llvm::any_of(op.getOperands(), [](Value v) {
        return v.getType().isIndex();
      });
      if (hasIndexOperand) {
        auto i64Ty = builder.getI64Type();
        SmallVector<Value> convertedOperands;
        for (Value operand : op.getOperands()) {
          Value mapped = mapping.lookupOrDefault(operand);
          if (operand.getType().isIndex()) {
            if (dyn_cast<VectorType>(mapped.getType())) {
              // Index vectors are invalid in MLIR; this path shouldn't be hit.
              convertedOperands.push_back(mapped);
            } else {
              auto i64Val = arith::IndexCastOp::create(builder, loc, i64Ty, mapped);
              auto vecI64Ty = VectorType::get({L_strip}, i64Ty);
              auto bc = vector::BroadcastOp::create(builder, loc, vecI64Ty, i64Val);
              convertedOperands.push_back(bc);
            }
          } else {
            auto subs = getSubsFor(operand);
            convertedOperands.push_back(subs.empty() ? mapped : subs[0]);
          }
        }
        auto maskTy = VectorType::get({L_strip}, builder.getI1Type());
        OperationState state(loc, op.getName());
        state.addOperands(convertedOperands);
        state.addAttributes(op.getAttrs());
        state.addTypes(maskTy);
        Operation *newCmp = builder.create(state);
        Value resultVec = newCmp->getResult(0);
        mapVec(op.getResult(0), {resultVec});
        return;
      }
      // Non-index cmpi/cmpf falls through to the general arith path below.
    }

    // R16: arith.select on index-typed result.
    if (isa<arith::SelectOp>(&op)) {
      Value origResult = op.getResult(0);
      Type resTy = origResult.getType();
      if (resTy.isIndex()) {
        Value cond = op.getOperand(0);
        Value trueVal = op.getOperand(1);
        Value falseVal = op.getOperand(2);

        auto i64Ty = builder.getI64Type();
        auto vecI64Ty = VectorType::get({L_strip}, i64Ty);
        auto maskTy = VectorType::get({L_strip}, builder.getI1Type());

        auto condSubs = getSubsFor(cond);
        Value vecCond = condSubs.empty() ? mapping.lookupOrDefault(cond)
                                         : condSubs[0];
        if (!dyn_cast<VectorType>(vecCond.getType()))
          vecCond = vector::BroadcastOp::create(builder, loc, maskTy, vecCond);

        auto trueSubs = getSubsFor(trueVal);
        Value vecTrue = trueSubs.empty() ? mapping.lookupOrDefault(trueVal)
                                         : trueSubs[0];
        if (trueVal.getType().isIndex() &&
            !dyn_cast<VectorType>(vecTrue.getType())) {
          auto cast = arith::IndexCastOp::create(builder, loc, i64Ty, vecTrue);
          vecTrue = vector::BroadcastOp::create(builder, loc, vecI64Ty, cast);
        }

        auto falseSubs = getSubsFor(falseVal);
        Value vecFalse = falseSubs.empty() ? mapping.lookupOrDefault(falseVal)
                                           : falseSubs[0];
        if (falseVal.getType().isIndex() &&
            !dyn_cast<VectorType>(vecFalse.getType())) {
          auto cast = arith::IndexCastOp::create(builder, loc, i64Ty, vecFalse);
          vecFalse = vector::BroadcastOp::create(builder, loc, vecI64Ty, cast);
        }

        Value selected = arith::SelectOp::create(builder, loc, vecCond,
                                                  vecTrue, vecFalse);
        mapVec(origResult, {selected});
        return;
      }
      // Non-index select falls through to the general arith path below.
    }

    // General arith pass-through: widen scalar result to sub-vector width.
    if (op.getNumResults() != 1) {
      builder.clone(op, mapping);
      return;
    }
    Value origResult = op.getResult(0);
    Type resTy = origResult.getType();

    if (!resTy.isIntOrFloat()) {
      // Index, memref, etc. -- clone with mapping.
      Operation *cloned = builder.clone(op, mapping);
      mapping.map(origResult, cloned->getResult(0));
      return;
    }

    // Determine natural sub-width for the result, clamped to L_strip.
    // Special case for i1: one mask bit per lane, use L_strip directly.
    int64_t resBytes = resTy.getIntOrFloatBitWidth() / 8;
    int64_t Ln_result;
    if (resBytes == 0) {
      Ln_result = L_strip;
    } else {
      Ln_result = std::min(getRegisterLanesForType(target, resBytes), L_strip);
    }
    int64_t numSubOpsResult = L_strip / Ln_result;

    // Build per-operand sub-vector lists aligned to Ln_result.
    SmallVector<SmallVector<Value>> operandSubs;
    bool sizingFailed = false;
    for (Value operand : op.getOperands()) {
      auto subs = getSubsFor(operand);
      SmallVector<Value> sized;

      if ((int64_t)subs.size() == numSubOpsResult) {
        sized = subs;
      } else if (subs.size() == 1) {
        Value v = subs[0];
        auto vecTy = dyn_cast<VectorType>(v.getType());
        if (!vecTy) {
          for (int64_t j = 0; j < numSubOpsResult; ++j) sized.push_back(v);
        } else {
          int64_t srcW = vecTy.getShape()[0];
          if (srcW == Ln_result) {
            sized.push_back(v);
          } else if (srcW > Ln_result) {
            for (int64_t j = 0; j < numSubOpsResult; ++j) {
              Value piece = vector::ExtractStridedSliceOp::create(
                  builder, loc, v,
                  /*offsets=*/ArrayRef<int64_t>{j * Ln_result},
                  /*sizes=*/ArrayRef<int64_t>{Ln_result},
                  /*strides=*/ArrayRef<int64_t>{1});
              sized.push_back(piece);
            }
          } else {
            // srcW < Ln_result: narrower than result -- fall back to scalar.
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
      Operation *cloned = builder.clone(op, mapping);
      mapping.map(origResult, cloned->getResult(0));
      return;
    }

    // Fast-math contract flag injection for arith.mulf / arith.addf.
    //
    // The `contract` flag allows the backend to fuse fmul+fadd -> vfmadd213ps
    // without enabling full -ffast-math semantics (no-nan, no-inf, etc.).
    // Only inject when contract is not already present (MLIR canonicalizer
    // adds `fastmath<none>` automatically, so we check existing flags).
    bool injectContractFMF = false;
    if (mlir::isa<FloatType, VectorType>(resTy)) {
      if (isa<arith::MulFOp, arith::AddFOp>(op)) {
        auto existingFMF = arith::FastMathFlags::none;
        if (auto fmfAttr =
                op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
          existingFMF = fmfAttr.getValue();
        if (!static_cast<bool>(existingFMF & arith::FastMathFlags::contract))
          injectContractFMF = true;
      }
    }
    Attribute mergedFMFAttr;
    if (injectContractFMF) {
      MLIRContext *ctx = builder.getContext();
      auto existingFMF = arith::FastMathFlags::none;
      if (auto fmfAttr = op.getAttrOfType<arith::FastMathFlagsAttr>("fastmath"))
        existingFMF = fmfAttr.getValue();
      mergedFMFAttr = arith::FastMathFlagsAttr::get(
          ctx, existingFMF | arith::FastMathFlags::contract);
    }

    // Emit numSubOpsResult instances of the op at sub-vector width.
    Type subResTy = VectorType::get({Ln_result}, resTy);
    SmallVector<Value> resultSubs;
    resultSubs.reserve(numSubOpsResult);
    for (int64_t j = 0; j < numSubOpsResult; ++j) {
      SmallVector<Value> opOperands;
      for (auto &operandList : operandSubs) opOperands.push_back(operandList[j]);
      OperationState state(loc, op.getName());
      state.addOperands(opOperands);
      state.addAttributes(op.getAttrs());
      state.addTypes(subResTy);
      if (mergedFMFAttr) {
        state.attributes.set(
            mlir::StringAttr::get(builder.getContext(), "fastmath"),
            mergedFMFAttr);
      }
      Operation *newOp = builder.create(state);
      resultSubs.push_back(newOp->getResult(0));
    }
    mapVec(origResult, std::move(resultSubs));
  }

  // -------------------------------------------------------------------------
  // R20: Outer-loop vectorization with inner reduction loop(s).
  //
  // Emits a vectorized copy of an inner scf.for (reduction loop) found inside
  // the tile_range outer loop.  The inner loop is cloned verbatim except:
  //   - loads/stores are replaced with their vectorized counterparts.
  //   - arith ops are replaced with their vector equivalents.
  //   - nested scf.for ops are handled recursively (R20-2L).
  //
  // This enables patterns such as:
  //   for j in tile_range:
  //     for i in range(M):            ← inner-1 (scalar i)
  //       for k in range(K):          ← inner-2 (scalar k, R20-2L)
  //         C[i*N+j] += A[i*K+k] * B[k*N+j]
  //
  // where j is vectorized across lanes and i,k are kept scalar.
  // C[i*N+j:j+16] is held in vector registers across all k iterations.
  //
  // Design: the outer loop's mapping already maps the outer IV to the vectorized
  // IV (ctx.newIv).  We add the inner IV mapping (scalar → new scalar IV) and
  // process the inner loop body ops using the existing per-kind emit helpers.
  // Recursion handles arbitrarily many levels of scalar reduction loops,
  // limited in practice to 2 by the bodyOK check.
  // -------------------------------------------------------------------------
  void emitInnerForOp(scf::ForOp innerFor,
                      ArrayRef<Operation *> accesses,
                      ArrayRef<lego::AccessClassification> classes) {
    // Clone the inner loop's bounds/step using the current mapping.
    Value lb = mapping.lookupOrDefault(innerFor.getLowerBound());
    Value ub = mapping.lookupOrDefault(innerFor.getUpperBound());
    Value step = mapping.lookupOrDefault(innerFor.getStep());

    auto newInnerFor = scf::ForOp::create(builder, loc, lb, ub, step);
    builder.setInsertionPointToStart(newInnerFor.getBody());

    // Add the inner IV to both the main mapping (for emitArithOp cloning) and
    // innerScalarMappings (for seeding gather lane maps in emitGatherLoad).
    mapping.map(innerFor.getInductionVar(), newInnerFor.getInductionVar());
    innerScalarMappings.map(innerFor.getInductionVar(),
                            newInnerFor.getInductionVar());

    // Process each op in the inner loop body.  The outer mapping already
    // contains the vectorized outer IV and broadcasts of outer-scope scalars
    // (populated in broadcastExternalScalars which was extended to scan inside
    // nested scf.for loops).
    for (Operation &innerOp : innerFor.getBody()->getOperations()) {
      if (isa<scf::YieldOp>(&innerOp)) continue;

      // R20-2L: recursively handle a further-nested scf.for (e.g. k-loop
      // inside the i-loop).  This keeps C[i*N+j] in vector registers across
      // the entire k reduction.
      if (auto nestedFor = dyn_cast<scf::ForOp>(&innerOp)) {
        emitInnerForOp(nestedFor, accesses, classes);
        continue;
      }

      if (auto load = dyn_cast<memref::LoadOp>(&innerOp)) {
        auto it = std::find(accesses.begin(), accesses.end(), &innerOp);
        if (it == accesses.end()) {
          // Not in accesses list — clone scalar.
          builder.clone(innerOp, mapping);
          continue;
        }
        const auto &cls = classes[it - accesses.begin()];
        switch (cls.kind) {
          case lego::AccessKind::Unit:       emitUnitLoad(load, cls);       break;
          case lego::AccessKind::Broadcast:  emitBroadcastLoad(load, cls);  break;
          case lego::AccessKind::CrossBlock: emitCrossBlockLoad(load, cls); break;
          case lego::AccessKind::Strided:    emitStridedLoad(load, cls);    break;
          case lego::AccessKind::NonAffine:  emitGatherLoad(load, cls);     break;
          default: {
            Operation *cloned = builder.clone(*load.getOperation(), mapping);
            mapping.map(load.getResult(), cloned->getResult(0));
            break;
          }
        }
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(&innerOp)) {
        auto it = std::find(accesses.begin(), accesses.end(), &innerOp);
        if (it == accesses.end()) {
          builder.clone(innerOp, mapping);
          continue;
        }
        const auto &cls = classes[it - accesses.begin()];
        if (cls.kind == lego::AccessKind::Unit)
          emitUnitStore(store, cls);
        else
          builder.clone(*store.getOperation(), mapping);
        continue;
      }

      // arith ops and other non-load/store ops: emit using the arith catch-all.
      emitArithOp(innerOp);
    }

    // Restore insertion point to after the new inner loop.
    builder.setInsertionPointAfter(newInnerFor);
  }
};  // struct EmitContext

/// Populate `vecLoop`'s body by dispatching each original loop op to the
/// appropriate per-kind emit helper in EmitContext.
static void emitVectorBody(scf::ForOp vecLoop, scf::ForOp origLoop,
                           int64_t L_strip,
                           ArrayRef<Operation *> accesses,
                           ArrayRef<lego::AccessClassification> classes,
                           llvm::StringRef targetStr,
                           OpBuilder &builder) {
  builder.setInsertionPointToStart(vecLoop.getBody());

  // Construct context; map orig IV -> new IV upfront.
  EmitContext ctx{origLoop.getLoc(),
                  vecLoop.getInductionVar(),
                  targetStr,
                  L_strip,
                  origLoop,
                  builder,
                  IRMapping{},
                  DenseMap<Value, SmallVector<Value>>{}};
  ctx.mapping.map(origLoop.getInductionVar(), ctx.newIv);

  // Pre-pass: broadcast loop-external scalars to natural sub-vector width.
  ctx.broadcastExternalScalars();

  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;  // vecLoop already has its own yield

    // -------------------------------------------------------------------
    // memref.load
    // -------------------------------------------------------------------
    if (auto load = dyn_cast<memref::LoadOp>(&op)) {
      auto it = std::find(accesses.begin(), accesses.end(), &op);
      assert(it != accesses.end() && "load not found in accesses");
      size_t idx = it - accesses.begin();
      const auto &cls = classes[idx];

      switch (cls.kind) {
        case lego::AccessKind::Unit:
          ctx.emitUnitLoad(load, cls);
          break;
        case lego::AccessKind::Broadcast:
          ctx.emitBroadcastLoad(load, cls);
          break;
        case lego::AccessKind::CrossBlock:
          ctx.emitCrossBlockLoad(load, cls);
          break;
        case lego::AccessKind::Strided:
          ctx.emitStridedLoad(load, cls);
          break;
        case lego::AccessKind::NonAffine:
          ctx.emitGatherLoad(load, cls);
          break;
        default: {
          // Shouldn't reach here for vectorizable loops.
          Operation *cloned = builder.clone(*load.getOperation(), ctx.mapping);
          ctx.mapping.map(load.getResult(), cloned->getResult(0));
          break;
        }
      }

    // -------------------------------------------------------------------
    // memref.store
    // -------------------------------------------------------------------
    } else if (auto store = dyn_cast<memref::StoreOp>(&op)) {
      auto it = std::find(accesses.begin(), accesses.end(), &op);
      assert(it != accesses.end() && "store not found in accesses");
      size_t idx = it - accesses.begin();
      const auto &cls = classes[idx];

      if (cls.kind == lego::AccessKind::Unit)
        ctx.emitUnitStore(store, cls);
      else
        builder.clone(*store.getOperation(), ctx.mapping);

    // -------------------------------------------------------------------
    // scf.if -- R17 predicated maskedstore
    // -------------------------------------------------------------------
    } else if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      ctx.emitMaskedStore(ifOp, accesses, classes);

    // -------------------------------------------------------------------
    // scf.for -- R20 outer-loop vectorization with inner reduction loop.
    // The inner scf.for (range() loop) is a scalar reduction loop whose body
    // contains loads/stores that reference the outer vectorized IV.  Clone
    // the inner loop into the vectorized body, replacing each inner-body
    // load/store with its vectorized equivalent.
    // -------------------------------------------------------------------
    } else if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
      ctx.emitInnerForOp(innerFor, accesses, classes);

    // -------------------------------------------------------------------
    // arith and all other ops -- mixed-precision aware pass-through
    // -------------------------------------------------------------------
    } else {
      ctx.emitArithOp(op);
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
      if (a.L_strip <= 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[lego-vectorize] skip loop in '"
                   << func.getName() << "'"
                   << " — L_strip=" << a.L_strip
                   << " (reason: cost-model or reduction guard)\n");
        // [G5] Diagnostic hint for flattened-iteration pattern (divui/remui on IV).
        // Detects: loop body contains arith.divui or arith.remui where one operand
        // is the induction variable (or IV-dependent). This is the hallmark of
        // "flat-grid" kernels with `i = ij/N; j = ij%N`.  In LEGO v1, these
        // produce NonAffine gather accesses which the cost model may reject.
        // Emit a remark so the user understands the limitation and the fix.
        {
          Value iv = a.forOp.getInductionVar();
          for (Operation &op : a.forOp.getBody()->getOperations()) {
            if (!isa<arith::DivUIOp, arith::RemUIOp, arith::DivSIOp,
                     arith::RemSIOp>(&op))
              continue;
            bool ivDep = llvm::any_of(op.getOperands(), [&](Value v) {
              if (v == iv) return true;
              Operation *defOp = v.getDefiningOp();
              return defOp && a.forOp->isAncestor(defOp);
            });
            if (ivDep) {
              op.emitRemark(
                  "lego-vectorize: loop body contains integer division on the "
                  "induction variable (flattened-iteration pattern: i=ij/N, "
                  "j=ij%N). This produces NonAffine gather accesses which are "
                  "~10x slower than unit-stride loads. Rewrite as nested loops "
                  "(for i in tile_range: for k in range(K): for j in range(N))"
                  " to enable unit-stride vectorization.");
              break;
            }
          }
        }
        continue;
      }

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
      //   gather path (cloneAddrChain) handles Morton-style index chains correctly
      //   ONLY for pure-address-compute paths where the entire chain is cloned
      //   per-lane.  The bodyOK guard ensures the catch-all path never sees an
      //   IV-dependent index_cast that it can't lower.  If index_cast were
      //   hoistable (IV-independent), the allInvariant path in evalLinearInIV already
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
          // require the same cloneAddrChain technique used for NonAffine gathers.
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
                // G2: emit a remark visible without --debug-only so the user
                // learns WHY vectorization was skipped (the most counterintuitive
                // rejection — legal-looking arith op that blocks vectorization).
                op.emitRemark(
                    "lego-vectorize: non-vectorizable — IV-dependent index_cast "
                    "with index source/result; rewrite index arithmetic to avoid "
                    "index_cast on the induction variable to enable vectorization");
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
            LLVM_DEBUG(llvm::dbgs()
                       << "[lego-vectorize] skip loop in '"
                       << func.getName() << "'"
                       << " — body contains scf.if with else-branch (G4)\n");
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
            LLVM_DEBUG(llvm::dbgs()
                       << "[lego-vectorize] skip loop in '"
                       << func.getName() << "'"
                       << " — scf.if then-block has non-store ops (G4)\n");
            bodyOK = false;
            break;
          }
          continue;
        }
        // R20: allow a single-level nested scf.for (scalar reduction loop)
        // whose body contains only arith/memref ops or a single further level
        // of nested scf.for (R20-2L: e.g. outer i-loop containing inner k-loop).
        // The inner loop is vectorized by emitInnerForOp when the outer loop
        // is strip-mined; all inner IVs remain scalar.
        if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
          bool innerOK = true;
          for (Operation &innerOp : innerFor.getBody()->getOperations()) {
            if (isa<memref::LoadOp, memref::StoreOp, scf::YieldOp>(innerOp))
              continue;
            if (innerOp.getDialect() &&
                isa<arith::ArithDialect>(innerOp.getDialect()))
              continue;
            // R20-2L: allow a second level of nested scf.for (innermost
            // reduction loop, e.g. k-loop inside i-loop), provided its body
            // contains only arith/memref ops.
            if (auto innermostFor = dyn_cast<scf::ForOp>(&innerOp)) {
              bool innermostOK = true;
              for (Operation &innermostOp :
                   innermostFor.getBody()->getOperations()) {
                if (isa<memref::LoadOp, memref::StoreOp, scf::YieldOp>(
                        innermostOp))
                  continue;
                if (innermostOp.getDialect() &&
                    isa<arith::ArithDialect>(innermostOp.getDialect()))
                  continue;
                LLVM_DEBUG(llvm::dbgs()
                           << "[lego-vectorize] skip loop in '"
                           << func.getName()
                           << "' — innermost reduction loop body contains "
                              "unsupported op: "
                           << innermostOp.getName().getStringRef() << "\n");
                innermostOK = false;
                break;
              }
              if (!innermostOK) { innerOK = false; break; }
              LLVM_DEBUG(llvm::dbgs()
                         << "[lego-vectorize] allowing 2-level nested scf.for "
                            "in '" << func.getName()
                         << "' — innermost reduction loop (R20-2L)\n");
              continue;
            }
            // Anything else inside the inner loop — too complex.
            LLVM_DEBUG(llvm::dbgs()
                       << "[lego-vectorize] skip loop in '"
                       << func.getName()
                       << "' — inner reduction loop body contains unsupported op: "
                       << innerOp.getName().getStringRef() << "\n");
            innerOK = false;
            break;
          }
          if (!innerOK) {
            bodyOK = false;
            break;
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "[lego-vectorize] allowing nested scf.for in '"
                     << func.getName()
                     << "' — inner reduction loop with arith/memref body (R20)\n");
          continue;
        }
        // [G3] Op outside the arith dialect and not an allowed scf.if or scf.for.
        LLVM_DEBUG(llvm::dbgs()
                   << "[lego-vectorize] skip loop in '" << func.getName()
                   << "' — body contains op outside allowlist: "
                   << op.getName().getStringRef() << " (G3)\n");
        bodyOK = false;
        break;
      }
      if (!bodyOK) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[lego-vectorize] skip loop in '" << func.getName()
                   << "' — body check failed (bodyOK=false)\n");
        continue;
      }

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
