//===- LegoVectorizeRLE.cpp - Vectorize edge-detect / RLE loops ----------===//
//
// Recognises the canonical *run-length encoding* / *edge-detection*
// pattern that ``clang -O3`` scalarises:
//
//     k = 0; prev = sentinel
//     for i in 0..N:
//         if A[i] != prev:
//             out[k] = A[i]
//             prev   = A[i]
//             k      = k + 1
//
// This is a compaction whose predicate depends on the *previous* element
// (the iter_arg ``prev``).  clang refuses to vectorise because of the
// loop-carried predicate; AVX-512 has all the building blocks
// (``vpcmpneqps`` + ``vpcompressps``) — they're just not driven.
//
// LEGO recognises the shape and rewrites to a strip-mined vector loop
// using **upstream MLIR vector dialect ops only**:
//
//     scf.for %i = 0 to stripUb step L
//                  iter_args(%k, %prev_vec) -> (index, vector<L x T>) {
//       %v       = vector.transfer_read A[%i]   : vector<L x T>
//       // Lane 0 of v_shift comes from %prev_vec (carry from prev chunk
//       // or initial sentinel); lanes [1..L-1] come from v[0..L-2].
//       %v_shift = vector.shuffle %prev_vec, %v [L-1, L+0, L+1, ..., 2L-2]
//       %mask    = arith.cmpf one, %v, %v_shift : vector<L x i1>
//       vector.compressstore out[%k], %mask, %v
//       %popcnt  = math.ctpop ( vector.bitcast %mask -> iL ) -> iL
//       %k_new   = %k + popcnt
//       // Carry: a vector whose last lane is v[L-1] (next chunk's "prev").
//       scf.yield %k_new, %v
//     }
//     // Tail: scalar body for (N mod L) iterations seeded from %k, last lane.
//
// Uses ``vector.shuffle`` to thread the previous element across chunks
// without a scalar carry chain.  Same ``vector.compressstore`` machinery
// as ``lego-vectorize-compact`` — just with a different predicate.
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZERLEPASS
#include "Lego/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

static int64_t getLanesForType(llvm::StringRef target, int64_t elemBytes) {
  if (target == "avx512") return elemBytes == 4 ? 16 : 8;
  if (target == "avx2")   return elemBytes == 4 ? 8  : 4;
  if (target == "neon" || target == "sve") return elemBytes == 4 ? 4 : 2;
  return 16 / std::max<int64_t>(elemBytes, 1);
}

/// Decoded shape of an edge-detect / RLE loop.  Two canonicalisation
/// forms are accepted:
///
///   Form A — original 2-result scf.if (cpu_dsl emits this; the
///   side-effecting store inside the then branch keeps canonicalize
///   from folding it into a select):
///
///       scf.if(cmp) -> (index, T) {
///         memref.store v, out[k]; k1 = k+1; yield k1, v
///       } else { yield k, prev }
///
///   Form B — split form (canonicalize-friendly hand-written IR):
///
///       %newPrev = arith.select cmp, v, prev
///       scf.if(cmp) -> (index) {
///         memref.store v, out[k]; k1 = k+1; yield k1
///       } else { yield k }
///       yield (if_k, newPrev)
struct RLEShape {
  memref::LoadOp     dataLoad;     // outer load A[i] used by the cmpf
  memref::StoreOp    outStore;     // store A[i] to out[k]
  arith::CmpFOp      cmpOp;        // arith.cmpf one (or une), A[i], prev
  arith::SelectOp    prevSelect;   // Form B only — null in Form A
  scf::IfOp          ifOp;         // scf.if(p) -> (index, T)  or  -> (index)
  arith::AddIOp      counterInc;   // k + 1 inside the then-branch
  Value              initK;        // initial counter value
  Value              initPrev;     // initial prev value
};

static constexpr llvm::StringRef kRLEDoneAttr = "lego.rle_done";

static std::optional<RLEShape> matchRLEPattern(scf::ForOp forOp) {
  if (forOp->hasAttr(kRLEDoneAttr)) return std::nullopt;

  // Two iter_args: (index, T) — k and prev.
  if (forOp.getNumRegionIterArgs() != 2) return std::nullopt;
  Value iaK    = forOp.getRegionIterArgs()[0];
  Value iaPrev = forOp.getRegionIterArgs()[1];
  if (!iaK.getType().isIndex())            return std::nullopt;
  if (!isa<FloatType>(iaPrev.getType()))   return std::nullopt;

  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1)
    return std::nullopt;

  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();

  arith::CmpFOp   cmpOp;
  arith::SelectOp prevSelect;       // Form B only
  scf::IfOp       ifOp;
  for (Operation &op : body->getOperations()) {
    if (auto c = dyn_cast<arith::CmpFOp>(&op)) {
      if (cmpOp) return std::nullopt;
      cmpOp = c;
    } else if (auto s = dyn_cast<arith::SelectOp>(&op)) {
      if (prevSelect) return std::nullopt;
      prevSelect = s;
    } else if (auto i = dyn_cast<scf::IfOp>(&op)) {
      if (ifOp) return std::nullopt;
      ifOp = i;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op)) {
      // Allowed.
    } else {
      return std::nullopt;
    }
  }
  if (!cmpOp || !ifOp) return std::nullopt;
  bool isFormB = (prevSelect != nullptr);

  // cmpf must be ONE (ordered not-equal) or UNE — value != prev.
  arith::CmpFPredicate pred = cmpOp.getPredicate();
  if (pred != arith::CmpFPredicate::ONE &&
      pred != arith::CmpFPredicate::UNE)
    return std::nullopt;

  // The cmpf compares A[i] against the prev iter_arg.
  auto dataLoad = cmpOp.getLhs().getDefiningOp<memref::LoadOp>();
  if (!dataLoad) return std::nullopt;
  if (dataLoad.getIndices().size() != 1) return std::nullopt;
  if (dataLoad.getIndices().front() != iv) return std::nullopt;
  if (cmpOp.getRhs() != iaPrev) return std::nullopt;

  // The if condition is the cmp.  Form A returns (index, T); Form B
  // returns (index) only.
  if (ifOp.getCondition() != cmpOp.getResult()) return std::nullopt;
  if (isFormB) {
    if (ifOp.getNumResults() != 1) return std::nullopt;
    if (!ifOp.getResult(0).getType().isIndex()) return std::nullopt;

    if (prevSelect.getCondition() != cmpOp.getResult()) return std::nullopt;
    if (prevSelect.getTrueValue()  != cmpOp.getLhs())   return std::nullopt;
    if (prevSelect.getFalseValue() != iaPrev)           return std::nullopt;
    if (prevSelect.getResult().getType() != iaPrev.getType())
      return std::nullopt;
  } else {
    if (ifOp.getNumResults() != 2) return std::nullopt;
    if (!ifOp.getResult(0).getType().isIndex()) return std::nullopt;
    if (ifOp.getResult(1).getType() != iaPrev.getType())
      return std::nullopt;
  }

  // for-yield must yield (if_k, prev_update).
  auto fy = cast<scf::YieldOp>(body->getTerminator());
  if (fy.getNumOperands() != 2) return std::nullopt;
  if (fy.getOperand(0) != ifOp.getResult(0)) return std::nullopt;
  Value yieldPrev = fy.getOperand(1);
  if (isFormB) {
    if (yieldPrev != prevSelect.getResult()) return std::nullopt;
  } else {
    if (yieldPrev != ifOp.getResult(1)) return std::nullopt;
  }

  // Else-branch: yield k (and prev for Form A) unchanged.
  if (ifOp.getElseRegion().empty()) return std::nullopt;
  Block *elseBlk = &ifOp.getElseRegion().front();
  auto eY = cast<scf::YieldOp>(elseBlk->getTerminator());
  size_t expectedElseOps = isFormB ? 1 : 2;
  if (eY.getNumOperands() != expectedElseOps) return std::nullopt;
  if (eY.getOperand(0) != iaK) return std::nullopt;
  if (!isFormB && eY.getOperand(1) != iaPrev) return std::nullopt;

  // Then-branch: optional load + store + addi + yield (k+1[, A[i]]).
  Block *thenBlk = &ifOp.getThenRegion().front();
  memref::StoreOp outStore;
  arith::AddIOp counterInc;
  for (Operation &op : thenBlk->getOperations()) {
    if (auto s = dyn_cast<memref::StoreOp>(&op)) {
      if (outStore) return std::nullopt;
      outStore = s;
    } else if (auto a = dyn_cast<arith::AddIOp>(&op)) {
      if (counterInc) return std::nullopt;
      counterInc = a;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op)) {
      // Allowed.
    } else {
      return std::nullopt;
    }
  }
  if (!outStore || !counterInc) return std::nullopt;

  // store must write to out[k].
  if (outStore.getIndices().size() != 1) return std::nullopt;
  if (outStore.getIndices().front() != iaK) return std::nullopt;

  // counter increment: k + 1.
  Value lhs = counterInc.getLhs();
  Value rhs = counterInc.getRhs();
  Value other;
  if (lhs == iaK)      other = rhs;
  else if (rhs == iaK) other = lhs;
  else return std::nullopt;
  APInt incVal;
  if (!matchPattern(other, m_ConstantInt(&incVal)) || incVal != 1)
    return std::nullopt;

  // then-branch yield: (counterInc) for Form B; (counterInc, A[i]) for Form A.
  auto tY = cast<scf::YieldOp>(thenBlk->getTerminator());
  size_t expectedThenOps = isFormB ? 1 : 2;
  if (tY.getNumOperands() != expectedThenOps) return std::nullopt;
  if (tY.getOperand(0) != counterInc.getResult()) return std::nullopt;
  if (!isFormB) {
    Value newPrev = tY.getOperand(1);
    if (newPrev != cmpOp.getLhs()) {
      // Allow a duplicate inner load on iv.
      auto dup = newPrev.getDefiningOp<memref::LoadOp>();
      if (!dup) return std::nullopt;
      if (dup.getMemRef() != dataLoad.getMemRef()) return std::nullopt;
      if (dup.getIndices().size() != 1) return std::nullopt;
      if (dup.getIndices().front() != iv) return std::nullopt;
    }
  }

  RLEShape shape;
  shape.dataLoad   = dataLoad;
  shape.outStore   = outStore;
  shape.cmpOp      = cmpOp;
  shape.prevSelect = prevSelect;
  shape.ifOp       = ifOp;
  shape.counterInc = counterInc;
  shape.initK      = forOp.getInits()[0];
  shape.initPrev   = forOp.getInits()[1];
  return shape;
}

static void cloneScalarBody(OpBuilder &b, Location loc, scf::ForOp origLoop,
                            Value newIv, ValueRange newIterArgs) {
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  for (auto z : llvm::zip(origLoop.getRegionIterArgs(), newIterArgs))
    mapping.map(std::get<0>(z), std::get<1>(z));
  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (auto y = dyn_cast<scf::YieldOp>(op)) {
      SmallVector<Value> mappedOperands;
      for (Value v : y.getOperands())
        mappedOperands.push_back(mapping.lookupOrDefault(v));
      scf::YieldOp::create(b, op.getLoc(), mappedOperands);
      break;
    }
    b.clone(op, mapping);
  }
  (void)loc;
}

struct VectorizeRLEPattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeRLEPattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchRLEPattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    auto fpTy = cast<FloatType>(shape->initPrev.getType());
    int64_t elemBytes = fpTy.getWidth() / 8;
    int64_t L = getLanesForType(target, elemBytes);
    if (!llvm::isPowerOf2_64(L) || L < 2) return failure();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value lenV   = arith::SubIOp::create(rewriter, loc, ub, lb);
    Value cL     = arith::ConstantIndexOp::create(rewriter, loc, L);
    Value chunks = arith::DivUIOp::create(rewriter, loc, lenV, cL);
    Value stripBody = arith::MulIOp::create(rewriter, loc, chunks, cL);
    Value stripUb = arith::AddIOp::create(rewriter, loc, lb, stripBody);

    auto vecTy = VectorType::get({L}, fpTy);
    Value pad = arith::ConstantOp::create(rewriter, loc, fpTy,
                                          rewriter.getZeroAttr(fpTy));

    // Initial "prev_vec": a vector whose last lane is initPrev (others
    // unused — only lane L-1 is read by the next chunk's shuffle).
    Value vecInitPrev = vector::BroadcastOp::create(rewriter, loc, vecTy,
                                                    shape->initPrev);

    auto vecLoop = scf::ForOp::create(rewriter, loc, lb, stripUb, cL,
                                      ValueRange{shape->initK, vecInitPrev});
    vecLoop->setAttr(kRLEDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();
      Value k        = vecLoop.getRegionIterArgs()[0];
      Value prevVec  = vecLoop.getRegionIterArgs()[1];

      // 1. Load chunk.
      Value v = vector::TransferReadOp::create(
                    rewriter, loc, vecTy, shape->dataLoad.getMemRef(),
                    ValueRange{iv}, pad,
                    /*inBounds=*/ArrayRef<bool>{true}).getResult();

      // 2. Build v_shift: lane 0 from prev_vec[L-1], lanes [1..L-1] from
      //    v[0..L-2].  Shuffle indices for vector.shuffle(prev_vec, v):
      //    [L-1, L+0, L+1, ..., L+(L-2)].
      SmallVector<int64_t> shiftMask(L);
      shiftMask[0] = L - 1;
      for (int64_t j = 1; j < L; ++j) shiftMask[j] = L + (j - 1);
      Value vShift = vector::ShuffleOp::create(rewriter, loc, prevVec, v,
                                               shiftMask);

      // 3. Build mask (v != v_shift).
      Value mask = arith::CmpFOp::create(rewriter, loc,
                                         shape->cmpOp.getPredicate(),
                                         v, vShift).getResult();

      // 4. vector.compressstore out[k], mask, v.
      vector::CompressStoreOp::create(
          rewriter, loc, shape->outStore.getMemRef(), ValueRange{k},
          mask, v, /*alignment=*/IntegerAttr{});

      // 5. k += popcount(mask).
      auto packedIntTy = rewriter.getIntegerType(L);
      auto packedVecTy = VectorType::get({1}, packedIntTy);
      Value packedVec = vector::BitCastOp::create(rewriter, loc,
                                                  packedVecTy, mask)
                            .getResult();
      Value packed = vector::ExtractOp::create(rewriter, loc, packedVec,
                                               ArrayRef<int64_t>{0})
                         .getResult();
      Value popcnt = math::CtPopOp::create(rewriter, loc, packed).getResult();
      Value popcntIdx = arith::IndexCastOp::create(rewriter, loc,
                                                   rewriter.getIndexType(),
                                                   popcnt).getResult();
      Value kNew = arith::AddIOp::create(rewriter, loc, k, popcntIdx)
                       .getResult();

      // 6. Carry: yield v as the next chunk's prev_vec (only lane L-1 will
      //    be read).
      scf::YieldOp::create(rewriter, loc, ValueRange{kNew, v});
    }

    // After the vector loop: extract the scalar prev for the tail.
    rewriter.setInsertionPointAfter(vecLoop);
    Value finalK    = vecLoop.getResult(0);
    Value finalPrev = vector::ExtractOp::create(
                          rewriter, loc, vecLoop.getResult(1),
                          ArrayRef<int64_t>{L - 1})
                          .getResult();

    // Tail loop: scalar body for (N mod L) iterations.
    auto tailLoop = scf::ForOp::create(rewriter, loc, stripUb, ub,
                                       forOp.getStep(),
                                       ValueRange{finalK, finalPrev});
    tailLoop->setAttr(kRLEDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      cloneScalarBody(rewriter, loc, forOp,
                      tailLoop.getInductionVar(),
                      tailLoop.getRegionIterArgs());
    }

    rewriter.replaceOp(forOp,
                       ValueRange{tailLoop.getResult(0), tailLoop.getResult(1)});
    return success();
  }
};

struct LegoVectorizeRLEPass
    : public mlir::lego::impl::LegoVectorizeRLEPassBase<
          LegoVectorizeRLEPass> {
  using LegoVectorizeRLEPassBase::LegoVectorizeRLEPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeRLEPattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeRLEPass() {
  return std::make_unique<LegoVectorizeRLEPass>();
}
std::unique_ptr<Pass> createLegoVectorizeRLEPass(llvm::StringRef target) {
  LegoVectorizeRLEPassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeRLEPass>(opts);
}
} // namespace lego
} // namespace mlir
