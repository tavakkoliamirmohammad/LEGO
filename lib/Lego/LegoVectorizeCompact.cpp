//===- LegoVectorizeCompact.cpp - Vectorize stream-filter loops ----------===//
//
// Recognises the canonical *compaction* / stream-filter pattern that the
// clang auto-vectoriser scalarises entirely:
//
//     k = 0
//     for i in 0..N:
//         if cond[i] > threshold:
//             B[k] = A[i]
//             k = k + 1
//
// The data-dependent output position ``k`` (loop-carried) is what defeats
// clang's auto-vec. AVX-512 has the exact instruction needed —
// ``vpcompressps`` — but clang's heuristic budget for "compress" patterns
// is conservative enough to skip them.
//
// LEGO recognises the shape directly and rewrites to a strip-mined vector
// loop using **upstream MLIR vector dialect ops only**:
//
//     scf.for %i = 0 to N step L iter_args(%k = 0) {
//       %vec_v   = vector.transfer_read A[%i] : vector<L x T>
//       %vec_c   = vector.transfer_read cond[%i] : vector<L x T>
//       %vec_thr = vector.broadcast %threshold
//       %mask    = arith.cmpf ogt, %vec_c, %vec_thr : vector<L x i1>
//       vector.compressstore B[%k], %mask, %vec_v
//       %popcnt  = math.ctpop ( vector.bitcast %mask -> iL ) : iL
//       %k_new   = %k + popcnt_idx
//       scf.yield %k_new
//     }
//     // Tail loop preserves the original scalar body for (N mod L) iterations.
//
// ``vector.compressstore`` lowers cleanly to AVX-512 ``vpcompressps`` on x86
// and ARM SVE ``compact + st1`` on aarch64 — no architecture-specific
// intrinsics in the recogniser.
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZECOMPACTPASS
#include "Lego/Passes.h"
#include "LegoSpecializedVectorize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

using mlir::lego::specialised::computeStripBounds;
using mlir::lego::specialised::getDefaultLanesForType;

/// The decoded shape of a compaction loop.  Holds the values the rewrite
/// needs without re-walking the IR.
struct CompactShape {
  // Data side: B[k] = A[i] (or any pointwise expression of A[i]).
  memref::LoadOp     dataLoad;       // load A[i]
  memref::StoreOp    dataStore;      // store B[k]
  // Predicate side: cond[i] cmp threshold
  memref::LoadOp     condLoad;       // load cond[i]
  Value              condCmp;        // arith.cmpf result (i1)
  arith::CmpFOp      cmpOp;
  Value              threshold;      // RHS of the cmp (loop-invariant)
  // The scf.if op whose then-branch is the predicated store + counter inc.
  scf::IfOp          ifOp;
  // The arith.addi inside the then-branch: k + 1 → yielded.
  arith::AddIOp      counterInc;
};

/// Marker attribute placed on the vec-loop and tail-loop after a successful
/// rewrite, so the greedy driver does not match the tail again.
static constexpr llvm::StringRef kCompactDoneAttr = "lego.compact_done";

/// Inspect ``forOp`` and decide whether it matches the compaction pattern.
/// Returns ``std::nullopt`` if the shape does not match.
static std::optional<CompactShape> matchCompactPattern(scf::ForOp forOp) {
  // Avoid re-matching loops that have already been rewritten (the tail loop
  // shares the original scalar body and would otherwise loop forever).
  if (forOp->hasAttr(kCompactDoneAttr)) return std::nullopt;

  // Must have exactly one iter_arg of index type.
  if (forOp.getNumRegionIterArgs() != 1) return std::nullopt;
  Value k = forOp.getRegionIterArgs().front();
  if (!k.getType().isIndex()) return std::nullopt;

  // Step must be 1 (we'll strip-mine).
  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1) {
    return std::nullopt;
  }

  // Walk the body.  Expected ops (in any reasonable order):
  //   - memref.load for A[i] (and possibly more)
  //   - memref.load for cond[i]
  //   - arith.cmpf (or similar) producing i1
  //   - scf.if(%p) -> index
  //   - scf.yield %k_new
  Block *body = forOp.getBody();
  scf::IfOp ifOp;
  arith::CmpFOp cmpOp;
  for (Operation &op : body->getOperations()) {
    if (auto i = dyn_cast<scf::IfOp>(&op)) {
      if (ifOp) return std::nullopt;        // only one scf.if
      ifOp = i;
    } else if (auto c = dyn_cast<arith::CmpFOp>(&op)) {
      if (cmpOp) return std::nullopt;       // only one cmp
      cmpOp = c;
    }
  }
  if (!ifOp || !cmpOp) return std::nullopt;
  if (ifOp.getCondition() != cmpOp.getResult()) return std::nullopt;

  // The scf.if must yield a single index value.
  if (ifOp.getNumResults() != 1) return std::nullopt;
  if (!ifOp.getResult(0).getType().isIndex()) return std::nullopt;

  // Only a then-branch may exist (no else, or else is just yield %k).
  Region &thenReg = ifOp.getThenRegion();
  if (thenReg.empty()) return std::nullopt;
  Block *thenBlk = &thenReg.front();

  // The else branch must yield %k unchanged (or be missing → implicit).
  if (!ifOp.getElseRegion().empty()) {
    Block *elseBlk = &ifOp.getElseRegion().front();
    if (elseBlk->getOperations().size() != 1) return std::nullopt;
    auto elseYield = dyn_cast<scf::YieldOp>(elseBlk->getTerminator());
    if (!elseYield || elseYield.getNumOperands() != 1) return std::nullopt;
    if (elseYield.getOperand(0) != k) return std::nullopt;
  }

  // Walk the then-branch.  Expected: optional memref.load(s) for the value
  // being stored (cpu_dsl emits A[i] inside the if), one memref.store
  // (B[k] = A[i]), one arith.addi (k + 1), and scf.yield.
  memref::StoreOp dataStore;
  arith::AddIOp   counterInc;
  for (Operation &op : thenBlk->getOperations()) {
    if (auto s = dyn_cast<memref::StoreOp>(&op)) {
      if (dataStore) return std::nullopt;
      dataStore = s;
    } else if (auto a = dyn_cast<arith::AddIOp>(&op)) {
      if (counterInc) return std::nullopt;
      counterInc = a;
    } else if (isa<scf::YieldOp, arith::ConstantOp, memref::LoadOp>(&op)) {
      // memref.load is allowed (cpu_dsl emits A[i] inside the if-then);
      // we hoist it out during emission.
    } else {
      return std::nullopt;                  // unknown op rejects the pattern
    }
  }
  if (!dataStore || !counterInc) return std::nullopt;

  // The store address must be ``k``.
  if (dataStore.getIndices().size() != 1) return std::nullopt;
  if (dataStore.getIndices().front() != k) return std::nullopt;

  // The counter increment must be ``addi(k, 1)``.
  Value lhs = counterInc.getLhs();
  Value rhs = counterInc.getRhs();
  Value otherOperand;
  if (lhs == k) otherOperand = rhs;
  else if (rhs == k) otherOperand = lhs;
  else return std::nullopt;
  APInt incVal;
  if (!matchPattern(otherOperand, m_ConstantInt(&incVal)) || incVal != 1)
    return std::nullopt;

  // The yield from the then branch must yield counterInc.
  auto thenYield = dyn_cast<scf::YieldOp>(thenBlk->getTerminator());
  if (!thenYield || thenYield.getNumOperands() != 1) return std::nullopt;
  if (thenYield.getOperand(0) != counterInc.getResult()) return std::nullopt;

  // Locate the data load in the body or in the then-branch.  ``A[i]`` is
  // typically loaded outside the if; ``cond[i]`` likewise.
  memref::LoadOp dataLoad;
  memref::LoadOp condLoad;
  // The cmp's LHS is cond[i]; its RHS is the threshold (typically constant).
  Value cmpLhs = cmpOp.getLhs();
  Value cmpRhs = cmpOp.getRhs();
  // Threshold must be loop-invariant (constant or defined outside the loop).
  Value threshold = cmpRhs;
  if (auto *defOp = threshold.getDefiningOp()) {
    if (forOp->isAncestor(defOp)) return std::nullopt;
  }
  // condLoad is the load that defines cmpLhs.
  condLoad = cmpLhs.getDefiningOp<memref::LoadOp>();
  if (!condLoad) return std::nullopt;
  if (condLoad.getIndices().size() != 1) return std::nullopt;
  if (condLoad.getIndices().front() != forOp.getInductionVar())
    return std::nullopt;

  // dataLoad — the source value being stored.  Must be a memref.load on the
  // same induction variable (unit-stride read).
  Value storedVal = dataStore.getValueToStore();
  dataLoad = storedVal.getDefiningOp<memref::LoadOp>();
  if (!dataLoad) return std::nullopt;
  if (dataLoad.getIndices().size() != 1) return std::nullopt;
  if (dataLoad.getIndices().front() != forOp.getInductionVar())
    return std::nullopt;

  return CompactShape{dataLoad,  dataStore, condLoad,  cmpOp.getResult(),
                      cmpOp,     threshold, ifOp,      counterInc};
}

/// Rewrite a recognised compaction loop to its vectorised form.
/// Inserts the new vector loop + scalar tail loop, then deletes the
/// original.
struct VectorizeCompactPattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeCompactPattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchCompactPattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    Type elemTy = shape->dataLoad.getType();
    if (!elemTy.isIntOrFloat()) return failure();
    int64_t elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    int64_t L = getDefaultLanesForType(target, elemBytes);
    if (L < 2) return failure();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value k0 = forOp.getInits().front();
    auto sb = computeStripBounds(rewriter, loc, lb, ub, L);
    Value cL = sb.cL;
    Value stripUb = sb.stripUb;

    // Build the vector loop: scf.for %i = %lb to %stripUb step %L
    //                          iter_args(%k = %k0) -> (index)
    auto vecLoop = scf::ForOp::create(
        rewriter, loc, lb, stripUb, cL, ValueRange{k0});
    vecLoop->setAttr(kCompactDoneAttr, rewriter.getUnitAttr());

    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();
      Value k = vecLoop.getRegionIterArgs().front();

      // vector.transfer_read of A[i] and cond[i], width L.
      auto vecTy = VectorType::get({L}, elemTy);
      Value padding = arith::ConstantOp::create(
          rewriter, loc, elemTy, rewriter.getZeroAttr(elemTy));

      Value vecV = vector::TransferReadOp::create(
                       rewriter, loc, vecTy, shape->dataLoad.getMemRef(),
                       ValueRange{iv}, padding,
                       /*inBounds=*/ArrayRef<bool>{true})
                       .getResult();
      Value vecC = vector::TransferReadOp::create(
                       rewriter, loc, vecTy, shape->condLoad.getMemRef(),
                       ValueRange{iv}, padding,
                       /*inBounds=*/ArrayRef<bool>{true})
                       .getResult();

      // Broadcast threshold to a vector and build the lane mask.
      Value vecThr = vector::BroadcastOp::create(
                         rewriter, loc, vecTy, shape->threshold)
                         .getResult();
      // Build the lane mask vector<L x i1> via cmpf.
      Value mask = arith::CmpFOp::create(rewriter, loc,
                                         shape->cmpOp.getPredicate(),
                                         vecC, vecThr).getResult();

      // vector.compressstore B[%k], %mask, %vecV — the lane-selecting write.
      //
      // Workaround for an MLIR upstream codegen bug: the ``MaybeAlign`` overload
      // of ``CompressStoreOp::build`` swaps mask and valueToStore before
      // delegating to the ``IntegerAttr`` overload (see
      // ``mlir/.../VectorOps.cpp.inc`` autogenerated from VectorOps.td).
      // Calling the ``IntegerAttr`` overload directly avoids the swap.
      SmallVector<Value> idxs{k};
      vector::CompressStoreOp::create(
          rewriter, loc, shape->dataStore.getMemRef(), idxs,
          mask, vecV, /*alignment=*/IntegerAttr{});

      // Advance %k by popcount(mask).
      //   bitcast vector<Lxi1> to vector<1xiL> → extract → math.ctpop → cast
      auto packedIntTy = rewriter.getIntegerType(L);
      auto packedVecTy = VectorType::get({1}, packedIntTy);
      Value packedVec = vector::BitCastOp::create(
                            rewriter, loc, packedVecTy, mask).getResult();
      Value packed = vector::ExtractOp::create(
                         rewriter, loc, packedVec, ArrayRef<int64_t>{0})
                         .getResult();
      Value popcnt = math::CtPopOp::create(rewriter, loc, packed).getResult();
      Value popcntIdx = arith::IndexCastOp::create(
                            rewriter, loc, rewriter.getIndexType(), popcnt)
                            .getResult();
      Value kNew = arith::AddIOp::create(rewriter, loc, k, popcntIdx).getResult();

      scf::YieldOp::create(rewriter, loc, ValueRange{kNew});
    }

    // Tail loop: clone the original scalar body for (N mod L) iterations,
    // starting from stripUb with the iter_arg from the vec loop's result.
    auto tailLoop = scf::ForOp::create(
        rewriter, loc, stripUb, ub, forOp.getStep(),
        ValueRange{vecLoop.getResult(0)});
    tailLoop->setAttr(kCompactDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      // Clone every op of the original body (except its terminator) with
      // a value mapping IV(orig) → IV(tail), iter_arg(orig) → iter_arg(tail).
      IRMapping mapping;
      mapping.map(forOp.getInductionVar(), tailLoop.getInductionVar());
      mapping.map(forOp.getRegionIterArgs().front(),
                  tailLoop.getRegionIterArgs().front());
      for (Operation &op : forOp.getBody()->getOperations()) {
        if (isa<scf::YieldOp>(op)) {
          // Re-yield the cloned counterInc result.
          Value mappedCounter = mapping.lookup(shape->counterInc.getResult());
          // Need to reflect what the scf.if would yield — clone the structure
          // of the original yield; the iter-arg yield reads from the if's
          // result, which has been mapped through the cloned scf.if.
          auto origYield = cast<scf::YieldOp>(&op);
          SmallVector<Value> mappedOperands;
          for (Value v : origYield.getOperands())
            mappedOperands.push_back(mapping.lookupOrDefault(v));
          scf::YieldOp::create(rewriter, op.getLoc(), mappedOperands);
          (void)mappedCounter;
          break;
        }
        rewriter.clone(op, mapping);
      }
    }

    rewriter.replaceOp(forOp, ValueRange{tailLoop.getResult(0)});
    return success();
  }
};

struct LegoVectorizeCompactPass
    : public mlir::lego::impl::LegoVectorizeCompactPassBase<
          LegoVectorizeCompactPass> {
  using LegoVectorizeCompactPassBase::LegoVectorizeCompactPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeCompactPattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeCompactPass() {
  return std::make_unique<LegoVectorizeCompactPass>();
}
std::unique_ptr<Pass> createLegoVectorizeCompactPass(llvm::StringRef target) {
  LegoVectorizeCompactPassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeCompactPass>(opts);
}
} // namespace lego
} // namespace mlir
