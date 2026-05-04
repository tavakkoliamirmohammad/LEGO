//===- LegoVectorizeFilteredReduce.cpp - Predicated reduction vectorise --===//
//
// Recognises the canonical *filtered* / *predicated reduction* pattern
// that ``clang -O3`` scalarises:
//
//     acc = identity
//     for i in 0..N:
//         if cond[i] <pred> threshold:
//             acc = combine(acc, A[i])
//
// where ``combine`` is one of ``arith.addf`` / ``mulf`` / ``maximumf`` /
// ``minimumf``.  clang refuses to vectorise because of the data-dependent
// branch; AVX-512 has the exact instruction needed (``vmaskmovps`` /
// ``vblendmps`` to mask the contribution before reduction).
//
// LEGO recognises the shape and rewrites to a strip-mined vector loop
// with a vector accumulator + per-chunk masked select to identity:
//
//     acc_v = broadcast(identity) : vector<L x T>
//     for i = 0 to stripUb step L iter_args(acc_v):
//       v_a    = vector.transfer_read A[i]      : vector<L x T>
//       v_c    = vector.transfer_read cond[i]   : vector<L x T>
//       vthr   = broadcast(threshold)
//       mask   = arith.cmpf <pred>, v_c, vthr   : vector<L x i1>
//       v_id   = broadcast(identity)
//       contrib= arith.select mask, v_a, v_id   // suppress non-passing lanes
//       acc_v  = combine(acc_v, contrib)
//     end
//     final  = vector.reduction <kind>, acc_v   // collapse to scalar
//     final  = combine(initAcc, final)          // include initial acc
//     // Tail: scalar body for (N mod L) iterations seeded from final.
//
// Uses upstream MLIR vector ops only — no architecture-specific intrinsics.
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_LEGOVECTORIZEFILTEREDREDUCEPASS
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
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {

using mlir::lego::specialised::computeStripBounds;
using mlir::lego::specialised::getDefaultLanesForType;

enum class CombineKind { AddF, MulF, MaximumF, MinimumF };

struct FilteredReduceShape {
  memref::LoadOp   condLoad;     // cond[i]
  memref::LoadOp   dataLoad;     // A[i]
  arith::CmpFOp    cmpOp;        // arith.cmpf <pred>, cond[i], threshold
  scf::IfOp        ifOp;         // scf.if(p) -> (T) { combine; yield } else { yield acc }
  Operation       *combineOp;    // arith.addf / mulf / maximumf / minimumf
  Value            initAcc;      // initial accumulator (iter_arg init)
  Value            threshold;    // RHS of cmpf — must be loop-invariant
  CombineKind      kind;
};

static constexpr llvm::StringRef kFilteredReduceDoneAttr =
    "lego.filtered_reduce_done";

static std::optional<CombineKind> classifyCombine(Operation *op) {
  if (isa<arith::AddFOp>(op))     return CombineKind::AddF;
  if (isa<arith::MulFOp>(op))     return CombineKind::MulF;
  if (isa<arith::MaximumFOp>(op)) return CombineKind::MaximumF;
  if (isa<arith::MinimumFOp>(op)) return CombineKind::MinimumF;
  return std::nullopt;
}

static std::optional<FilteredReduceShape>
matchFilteredReducePattern(scf::ForOp forOp) {
  if (forOp->hasAttr(kFilteredReduceDoneAttr)) return std::nullopt;

  if (forOp.getNumRegionIterArgs() != 1) return std::nullopt;
  Value iaAcc = forOp.getRegionIterArgs()[0];
  if (!isa<FloatType>(iaAcc.getType())) return std::nullopt;

  APInt stepVal;
  if (!matchPattern(forOp.getStep(), m_ConstantInt(&stepVal)) ||
      stepVal != 1)
    return std::nullopt;

  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();

  arith::CmpFOp cmpOp;
  scf::IfOp     ifOp;
  for (Operation &op : body->getOperations()) {
    if (auto c = dyn_cast<arith::CmpFOp>(&op)) {
      if (cmpOp) return std::nullopt;
      cmpOp = c;
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

  // The if has one float result that's the new acc.
  if (ifOp.getCondition() != cmpOp.getResult()) return std::nullopt;
  if (ifOp.getNumResults() != 1) return std::nullopt;
  if (ifOp.getResult(0).getType() != iaAcc.getType()) return std::nullopt;

  // for-yield must yield the if's result.
  auto fy = cast<scf::YieldOp>(body->getTerminator());
  if (fy.getNumOperands() != 1) return std::nullopt;
  if (fy.getOperand(0) != ifOp.getResult(0)) return std::nullopt;

  // Else branch yields acc unchanged.
  if (ifOp.getElseRegion().empty()) return std::nullopt;
  Block *elseBlk = &ifOp.getElseRegion().front();
  auto eY = cast<scf::YieldOp>(elseBlk->getTerminator());
  if (eY.getNumOperands() != 1) return std::nullopt;
  if (eY.getOperand(0) != iaAcc) return std::nullopt;

  // Then branch: optional memref.load + combine_op + yield.
  Block *thenBlk = &ifOp.getThenRegion().front();
  Operation *combineOp = nullptr;
  std::optional<CombineKind> kind;
  for (Operation &op : thenBlk->getOperations()) {
    if (auto k = classifyCombine(&op)) {
      if (combineOp) return std::nullopt;
      combineOp = &op;
      kind = k;
    } else if (isa<memref::LoadOp, scf::YieldOp, arith::ConstantOp>(&op)) {
      // Allowed.
    } else {
      return std::nullopt;
    }
  }
  if (!combineOp || !kind) return std::nullopt;

  auto tY = cast<scf::YieldOp>(thenBlk->getTerminator());
  if (tY.getNumOperands() != 1) return std::nullopt;
  if (tY.getOperand(0) != combineOp->getResult(0)) return std::nullopt;

  // The combine op must mix iaAcc with a memref.load on iv.
  Value lhs = combineOp->getOperand(0);
  Value rhs = combineOp->getOperand(1);
  memref::LoadOp dataLoad;
  if (lhs == iaAcc)      dataLoad = rhs.getDefiningOp<memref::LoadOp>();
  else if (rhs == iaAcc) dataLoad = lhs.getDefiningOp<memref::LoadOp>();
  else return std::nullopt;
  if (!dataLoad) return std::nullopt;
  if (dataLoad.getIndices().size() != 1) return std::nullopt;
  if (dataLoad.getIndices().front() != iv) return std::nullopt;

  // The cmpf's lhs must be a memref.load of cond[i]; the rhs must be
  // loop-invariant (constant or defined outside the loop).
  auto condLoad = cmpOp.getLhs().getDefiningOp<memref::LoadOp>();
  if (!condLoad) return std::nullopt;
  if (condLoad.getIndices().size() != 1) return std::nullopt;
  if (condLoad.getIndices().front() != iv) return std::nullopt;
  Value threshold = cmpOp.getRhs();
  if (auto *defOp = threshold.getDefiningOp())
    if (forOp->isAncestor(defOp)) return std::nullopt;

  FilteredReduceShape shape;
  shape.condLoad  = condLoad;
  shape.dataLoad  = dataLoad;
  shape.cmpOp     = cmpOp;
  shape.ifOp      = ifOp;
  shape.combineOp = combineOp;
  shape.initAcc   = forOp.getInits().front();
  shape.threshold = threshold;
  shape.kind      = *kind;
  return shape;
}

static Value emitCombine(OpBuilder &b, Location loc, CombineKind kind,
                         Value lhs, Value rhs) {
  switch (kind) {
    case CombineKind::AddF:
      return arith::AddFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MulF:
      return arith::MulFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MaximumF:
      return arith::MaximumFOp::create(b, loc, lhs, rhs).getResult();
    case CombineKind::MinimumF:
      return arith::MinimumFOp::create(b, loc, lhs, rhs).getResult();
  }
  llvm_unreachable("unknown CombineKind");
}

static APFloat identityFor(CombineKind kind, FloatType fpTy) {
  const llvm::fltSemantics &sem = fpTy.getFloatSemantics();
  switch (kind) {
    case CombineKind::AddF:     return APFloat::getZero(sem);
    case CombineKind::MulF:     return APFloat(sem, 1);
    case CombineKind::MaximumF: return APFloat::getInf(sem, /*Negative=*/true);
    case CombineKind::MinimumF: return APFloat::getInf(sem, /*Negative=*/false);
  }
  llvm_unreachable("unknown CombineKind");
}

static vector::CombiningKind toVectorCombiningKind(CombineKind kind) {
  switch (kind) {
    case CombineKind::AddF:     return vector::CombiningKind::ADD;
    case CombineKind::MulF:     return vector::CombiningKind::MUL;
    case CombineKind::MaximumF: return vector::CombiningKind::MAXIMUMF;
    case CombineKind::MinimumF: return vector::CombiningKind::MINIMUMF;
  }
  llvm_unreachable("unknown CombineKind");
}

static void cloneScalarBody(OpBuilder &b, Location loc, scf::ForOp origLoop,
                            Value newIv, Value newAcc) {
  IRMapping mapping;
  mapping.map(origLoop.getInductionVar(), newIv);
  mapping.map(origLoop.getRegionIterArgs().front(), newAcc);
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

struct VectorizeFilteredReducePattern : public OpRewritePattern<scf::ForOp> {
  llvm::StringRef target;
  VectorizeFilteredReducePattern(MLIRContext *ctx, llvm::StringRef target)
      : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2), target(target) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto shape = matchFilteredReducePattern(forOp);
    if (!shape) return failure();

    Location loc = forOp.getLoc();
    auto fpTy = cast<FloatType>(shape->initAcc.getType());
    int64_t elemBytes = fpTy.getWidth() / 8;
    int64_t L = getDefaultLanesForType(target, elemBytes);
    if (L < 2) return failure();

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    auto sb = computeStripBounds(rewriter, loc, lb, ub, L);
    Value cL = sb.cL;
    Value stripUb = sb.stripUb;

    auto vecTy = VectorType::get({L}, fpTy);
    Value pad = arith::ConstantOp::create(rewriter, loc, fpTy,
                                          rewriter.getZeroAttr(fpTy));

    // Identity vector — used to suppress non-passing lanes via select.
    APFloat ident = identityFor(shape->kind, fpTy);
    auto identAttr = DenseElementsAttr::get(vecTy, FloatAttr::get(fpTy, ident));
    Value vecIdent = arith::ConstantOp::create(rewriter, loc, vecTy, identAttr);

    // Vector loop iter_arg starts at the identity (acc_v = identity), so the
    // *partial* sum is the contribution of the strip-mined region only —
    // we combine the original initAcc separately at the end.
    auto vecLoop = scf::ForOp::create(rewriter, loc, lb, stripUb, cL,
                                      ValueRange{vecIdent});
    vecLoop->setAttr(kFilteredReduceDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(vecLoop.getBody());
      Value iv = vecLoop.getInductionVar();
      Value vAcc = vecLoop.getRegionIterArgs().front();

      Value vA = vector::TransferReadOp::create(
                     rewriter, loc, vecTy, shape->dataLoad.getMemRef(),
                     ValueRange{iv}, pad,
                     /*inBounds=*/ArrayRef<bool>{true}).getResult();
      Value vC = vector::TransferReadOp::create(
                     rewriter, loc, vecTy, shape->condLoad.getMemRef(),
                     ValueRange{iv}, pad,
                     /*inBounds=*/ArrayRef<bool>{true}).getResult();

      Value vThr = vector::BroadcastOp::create(rewriter, loc, vecTy,
                                               shape->threshold);
      Value mask = arith::CmpFOp::create(rewriter, loc,
                                         shape->cmpOp.getPredicate(),
                                         vC, vThr).getResult();
      // Suppress non-passing lanes: replace them with the identity element so
      // the combine op leaves acc unchanged for those lanes.
      Value contrib = arith::SelectOp::create(rewriter, loc, mask, vA, vecIdent)
                          .getResult();
      Value newAcc = emitCombine(rewriter, loc, shape->kind, vAcc, contrib);

      scf::YieldOp::create(rewriter, loc, ValueRange{newAcc});
    }

    // Reduce vector accumulator to scalar.
    rewriter.setInsertionPointAfter(vecLoop);
    Value scalarPartial = vector::ReductionOp::create(
        rewriter, loc, toVectorCombiningKind(shape->kind),
        vecLoop.getResult(0), arith::FastMathFlags::none).getResult();
    // Combine with original initAcc to restore the full reduction.
    Value scalarAfterStrip =
        emitCombine(rewriter, loc, shape->kind, shape->initAcc, scalarPartial);

    // Tail loop: scalar body for (N mod L) iterations.
    auto tailLoop = scf::ForOp::create(rewriter, loc, stripUb, ub,
                                       forOp.getStep(),
                                       ValueRange{scalarAfterStrip});
    tailLoop->setAttr(kFilteredReduceDoneAttr, rewriter.getUnitAttr());
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      cloneScalarBody(rewriter, loc, forOp,
                      tailLoop.getInductionVar(),
                      tailLoop.getRegionIterArgs().front());
    }

    rewriter.replaceOp(forOp, ValueRange{tailLoop.getResult(0)});
    return success();
  }
};

struct LegoVectorizeFilteredReducePass
    : public mlir::lego::impl::LegoVectorizeFilteredReducePassBase<
          LegoVectorizeFilteredReducePass> {
  using LegoVectorizeFilteredReducePassBase::LegoVectorizeFilteredReducePassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizeFilteredReducePattern>(&getContext(), this->target);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVectorizeFilteredReducePass() {
  return std::make_unique<LegoVectorizeFilteredReducePass>();
}
std::unique_ptr<Pass>
createLegoVectorizeFilteredReducePass(llvm::StringRef target) {
  LegoVectorizeFilteredReducePassOptions opts;
  opts.target = target.str();
  return std::make_unique<LegoVectorizeFilteredReducePass>(opts);
}
} // namespace lego
} // namespace mlir
