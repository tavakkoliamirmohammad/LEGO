//===- LegoVectorizeArgmin.cpp - Vectorize argmin/argmax loops -----------===//
//
// Recognises the canonical argmin / argmax pattern that ``clang -O3``
// scalarises:
//
//     m  = +inf
//     mi = 0
//     for i in 0..N:
//         if A[i] < m:
//             m  = A[i]
//             mi = i
//
// clang refuses to vectorise because of the paired reduction (running
// min-value AND running min-index updated together) and the
// data-dependent branch.  AVX-512 has all the building blocks
// (``vminps``, ``vcmpps``, ``vpblendmd``, ``vpermt2d`` etc.) — they're
// just not driven by clang's heuristics.
//
// LEGO recognises the shape and rewrites to a strip-mined vector loop
// that maintains *L* parallel running winners (one per lane), then
// reduces to a single (min, idx) pair after the strip.  Uses
// **upstream MLIR vector dialect ops only**:
//
//     acc_min = vector<L x f32>   broadcast(+inf)
//     acc_idx = vector<L x index> broadcast(0)
//     iota    = [0, 1, ..., L-1]   // constant
//     for i = 0 to stripUb step L:
//       v_a   = vector.transfer_read A[i]            : vector<L x f32>
//       v_idx = iota + broadcast(i)                  : vector<L x index>
//       mask  = arith.cmpf olt, v_a, acc_min         : vector<L x i1>
//       acc_min = arith.minimumf v_a, acc_min
//       acc_idx = arith.select  mask, v_idx, acc_idx
//     end
//     final_min = vector.reduction <minimumf>, acc_min      // scalar
//     eq_mask   = arith.cmpf oeq, acc_min, broadcast(final_min)
//     lane_idx  = arith.select eq_mask, acc_idx, broadcast(N)
//     final_idx = vector.reduction <minui>, lane_idx        // scalar
//     // Tail: scalar body for (N mod L), seeded from (final_min, final_idx).
//
// Predicates supported (mapped to the appropriate min/max reductions):
//   ``olt`` → argmin (vector.reduction <minimumf>)
//   ``ogt`` → argmax (vector.reduction <maximumf>)
//
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZEARGMINPASS
#include "Lego/Passes.h"
#include "LegoSpecializedVectorize.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

using mlir::lego::specialised::computeStripBounds;
using mlir::lego::specialised::getDefaultLanesForType;

/// Decoded shape of an argmin/argmax loop.  Canonicalisation typically
/// breaks the paired update into:
///   * ``arith.select`` for the index iter_arg
///   * ``scf.if -> (T)`` for the value iter_arg (containing a re-load)
/// We accept this canonical form.
struct ArgminShape {
  memref::LoadOp       dataLoad;      // A[i] — outer load used by cmpf
  arith::CmpFOp        cmpOp;         // arith.cmpf <pred>, A[i], cur_min
  arith::SelectOp      idxSelect;     // arith.select cmp, %iv, cur_mi
  scf::IfOp            valIfOp;       // scf.if cmp -> (T) { reload; yield }
                                      //              else { yield cur_min }
  Value                initVal;       // initial min value
  Value                initIdx;       // initial min index
  arith::CmpFPredicate pred;          // olt → argmin, ogt → argmax
};

static constexpr llvm::StringRef kArgminDoneAttr = "lego.argmin_done";

/// Returns the matched shape if ``forOp`` is an argmin/argmax loop.
static std::optional<ArgminShape> matchArgminPattern(scf::ForOp forOp) {
  if (forOp->hasAttr(kArgminDoneAttr)) return std::nullopt;

  // Two iter_args: (T, index) where T is a float type.
  if (forOp.getNumRegionIterArgs() != 2) return std::nullopt;
  Value iaVal = forOp.getRegionIterArgs()[0];
  Value iaIdx = forOp.getRegionIterArgs()[1];
  if (!isa<FloatType>(iaVal.getType())) return std::nullopt;
  if (!iaIdx.getType().isIndex()) return std::nullopt;

  // Step must be 1.
  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1)
    return std::nullopt;

  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();

  arith::CmpFOp   cmpOp;
  arith::SelectOp idxSelect;
  scf::IfOp       valIfOp;
  // Loop body must contain (in any order):
  //   - memref.load(s) of A[i]
  //   - one arith.cmpf
  //   - one arith.select for the index update
  //   - one scf.if returning (T) for the value update
  //   - scf.yield
  for (Operation &op : body->getOperations()) {
    if (auto c = dyn_cast<arith::CmpFOp>(&op)) {
      if (cmpOp) return std::nullopt;
      cmpOp = c;
    } else if (auto s = dyn_cast<arith::SelectOp>(&op)) {
      if (idxSelect) return std::nullopt;
      idxSelect = s;
    } else if (auto i = dyn_cast<scf::IfOp>(&op)) {
      if (valIfOp) return std::nullopt;
      valIfOp = i;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op)) {
      // Allowed.
    } else {
      return std::nullopt;
    }
  }
  if (!cmpOp || !idxSelect || !valIfOp) return std::nullopt;

  // Predicate must be olt or ogt.
  arith::CmpFPredicate pred = cmpOp.getPredicate();
  if (pred != arith::CmpFPredicate::OLT && pred != arith::CmpFPredicate::OGT)
    return std::nullopt;

  // The cmpf compares A[i] (lhs) against the current iter_arg value (rhs).
  auto dataLoad = cmpOp.getLhs().getDefiningOp<memref::LoadOp>();
  if (!dataLoad) return std::nullopt;
  if (cmpOp.getRhs() != iaVal) return std::nullopt;
  if (dataLoad.getIndices().size() != 1) return std::nullopt;
  if (dataLoad.getIndices().front() != iv) return std::nullopt;

  // The arith.select must be: select(cmp, iv, iaIdx) → produce new index.
  if (idxSelect.getCondition() != cmpOp.getResult()) return std::nullopt;
  if (idxSelect.getTrueValue()  != iv)               return std::nullopt;
  if (idxSelect.getFalseValue() != iaIdx)            return std::nullopt;
  if (!idxSelect.getResult().getType().isIndex())    return std::nullopt;

  // The scf.if uses cmp as condition and returns one T (the value).
  if (valIfOp.getCondition() != cmpOp.getResult())   return std::nullopt;
  if (valIfOp.getNumResults() != 1)                  return std::nullopt;
  if (valIfOp.getResult(0).getType() != iaVal.getType())
    return std::nullopt;

  // Else branch yields cur_min (iaVal).
  if (valIfOp.getElseRegion().empty()) return std::nullopt;
  Block *elseBlk = &valIfOp.getElseRegion().front();
  auto eYield = cast<scf::YieldOp>(elseBlk->getTerminator());
  if (eYield.getNumOperands() != 1) return std::nullopt;
  if (eYield.getOperand(0) != iaVal) return std::nullopt;

  // Then branch yields A[i] (either via re-load or the outer load directly).
  Block *thenBlk = &valIfOp.getThenRegion().front();
  auto tYield = cast<scf::YieldOp>(thenBlk->getTerminator());
  if (tYield.getNumOperands() != 1) return std::nullopt;
  Value newVal = tYield.getOperand(0);
  if (newVal != cmpOp.getLhs()) {
    auto dup = newVal.getDefiningOp<memref::LoadOp>();
    if (!dup) return std::nullopt;
    if (dup.getMemRef() != dataLoad.getMemRef()) return std::nullopt;
    if (dup.getIndices().size() != 1) return std::nullopt;
    if (dup.getIndices().front() != iv) return std::nullopt;
  }

  // The forOp must yield (val_if_result, idx_select_result).
  auto fyield = cast<scf::YieldOp>(body->getTerminator());
  if (fyield.getNumOperands() != 2) return std::nullopt;
  if (fyield.getOperand(0) != valIfOp.getResult(0)) return std::nullopt;
  if (fyield.getOperand(1) != idxSelect.getResult()) return std::nullopt;

  ArgminShape shape;
  shape.dataLoad  = dataLoad;
  shape.cmpOp     = cmpOp;
  shape.idxSelect = idxSelect;
  shape.valIfOp   = valIfOp;
  shape.initVal   = forOp.getInits()[0];
  shape.initIdx   = forOp.getInits()[1];
  shape.pred      = pred;
  return shape;
}

/// Clone the original scalar body for the tail loop with a value mapping.
static void cloneScalarBody(OpBuilder &b, Location loc, scf::ForOp origLoop,
                            Value newIv,
                            ValueRange newIterArgs,
                            scf::YieldOp &outYield) {
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  for (auto z : llvm::zip(origLoop.getRegionIterArgs(), newIterArgs))
    mapping.map(std::get<0>(z), std::get<1>(z));
  for (Operation &op : origLoop.getBody()->getOperations()) {
    if (auto y = dyn_cast<scf::YieldOp>(op)) {
      // Re-yield the cloned (val, idx) pair.
      SmallVector<Value> mappedOperands;
      for (Value v : y.getOperands())
        mappedOperands.push_back(mapping.lookupOrDefault(v));
      outYield = scf::YieldOp::create(b, op.getLoc(), mappedOperands);
      break;
    }
    b.clone(op, mapping);
  }
}

struct VectorizeArgminPattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeArgminPattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchArgminPattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    auto fpTy = cast<FloatType>(shape->initVal.getType());
    int64_t elemBytes = fpTy.getWidth() / 8;
    int64_t L = getDefaultLanesForType(target, elemBytes);
    if (L < 2) return failure();

    bool isArgmax = (shape->pred == arith::CmpFPredicate::OGT);

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    auto sb = computeStripBounds(rewriter, loc, lb, ub, L);
    Value cL = sb.cL;
    Value stripUb = sb.stripUb;

    auto valVecTy = VectorType::get({L}, fpTy);
    auto idxVecTy = VectorType::get({L}, rewriter.getIndexType());
    auto i1VecTy  = VectorType::get({L}, rewriter.getI1Type());

    // Per-lane initial accumulators: broadcast init scalars.
    Value vecInitVal = vector::BroadcastOp::create(rewriter, loc, valVecTy,
                                                   shape->initVal);
    Value vecInitIdx = vector::BroadcastOp::create(rewriter, loc, idxVecTy,
                                                   shape->initIdx);

    // iota constant: vector<L x index> = [0, 1, ..., L-1].  Use
    // vector.step → vector<Lxindex> directly.
    SmallVector<int64_t> iotaInts(L);
    for (int64_t j = 0; j < L; ++j) iotaInts[j] = j;
    Value iota = vector::StepOp::create(rewriter, loc, idxVecTy);

    // Vector loop with iter_args (acc_min, acc_idx) of vector types.
    auto vecLoop = scf::ForOp::create(
        rewriter, loc, lb, stripUb, cL,
        ValueRange{vecInitVal, vecInitIdx});
    vecLoop->setAttr(kArgminDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();
      Value accVal = vecLoop.getRegionIterArgs()[0];
      Value accIdx = vecLoop.getRegionIterArgs()[1];

      Value pad = arith::ConstantOp::create(rewriter, loc, fpTy,
                                            rewriter.getZeroAttr(fpTy));
      Value vA = vector::TransferReadOp::create(
                     rewriter, loc, valVecTy, shape->dataLoad.getMemRef(),
                     ValueRange{iv}, pad,
                     /*inBounds=*/ArrayRef<bool>{true}).getResult();

      // v_idx = iota + broadcast(iv) : vector<Lxindex>.
      Value bcastIv = vector::BroadcastOp::create(rewriter, loc, idxVecTy, iv);
      Value vIdx = arith::AddIOp::create(rewriter, loc, iota, bcastIv);

      // mask = vA < accVal (or vA > accVal for argmax).
      Value mask = arith::CmpFOp::create(rewriter, loc, shape->pred,
                                         vA, accVal);
      // newAccVal = min/max(vA, accVal).
      Value newAccVal;
      if (isArgmax) {
        newAccVal = arith::MaximumFOp::create(rewriter, loc, vA, accVal)
                        .getResult();
      } else {
        newAccVal = arith::MinimumFOp::create(rewriter, loc, vA, accVal)
                        .getResult();
      }
      // newAccIdx = select(mask, vIdx, accIdx).
      Value newAccIdx = arith::SelectOp::create(rewriter, loc, mask,
                                                vIdx, accIdx);

      scf::YieldOp::create(rewriter, loc,
                           ValueRange{newAccVal, newAccIdx});
      (void)i1VecTy;
    }

    // Reduce vector accumulator to scalar (val, idx).
    rewriter.setInsertionPointAfter(vecLoop);
    Value vAccVal = vecLoop.getResult(0);
    Value vAccIdx = vecLoop.getResult(1);
    vector::CombiningKind redKind =
        isArgmax ? vector::CombiningKind::MAXIMUMF
                 : vector::CombiningKind::MINIMUMF;
    Value scalarMin = vector::ReductionOp::create(
                          rewriter, loc, redKind, vAccVal,
                          arith::FastMathFlags::none).getResult();
    Value bcastScalarMin = vector::BroadcastOp::create(rewriter, loc,
                                                       valVecTy, scalarMin);
    Value eqMask = arith::CmpFOp::create(rewriter, loc,
                                         arith::CmpFPredicate::OEQ,
                                         vAccVal, bcastScalarMin);
    // For lanes that don't hold the min, use ub (or any large value); we'll
    // pick the smallest lane index that holds the min.
    Value bcastBigIdx = vector::BroadcastOp::create(rewriter, loc, idxVecTy,
                                                    ub);
    Value laneOrBig = arith::SelectOp::create(rewriter, loc, eqMask,
                                              vAccIdx, bcastBigIdx);
    Value scalarIdx = vector::ReductionOp::create(
                          rewriter, loc, vector::CombiningKind::MINUI,
                          laneOrBig, arith::FastMathFlags::none).getResult();

    // Tail loop: clone original scalar body for (N mod L) iterations,
    // seeded from (scalarMin, scalarIdx).
    auto tailLoop = scf::ForOp::create(
        rewriter, loc, stripUb, ub, forOp.getStep(),
        ValueRange{scalarMin, scalarIdx});
    tailLoop->setAttr(kArgminDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      scf::YieldOp dummy;
      cloneScalarBody(rewriter, loc, forOp,
                      tailLoop.getInductionVar(),
                      tailLoop.getRegionIterArgs(), dummy);
    }

    rewriter.replaceOp(forOp,
                       ValueRange{tailLoop.getResult(0), tailLoop.getResult(1)});
    return success();
  }
};

struct LegoVectorizeArgminPass
    : public mlir::lego::impl::LegoVectorizeArgminPassBase<
          LegoVectorizeArgminPass> {
  using LegoVectorizeArgminPassBase::LegoVectorizeArgminPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeArgminPattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeArgminPass() {
  return std::make_unique<LegoVectorizeArgminPass>();
}
std::unique_ptr<Pass> createLegoVectorizeArgminPass(llvm::StringRef target) {
  LegoVectorizeArgminPassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeArgminPass>(opts);
}
} // namespace lego
} // namespace mlir
