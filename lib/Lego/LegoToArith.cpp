#define GEN_PASS_DEF_LEGOTOARITHPASS
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Lego/LegoUtils.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

// ============================================================================
// Helpers
// ============================================================================

Value getConstantIndex(OpBuilder &b, Location loc, int64_t val) {
  return arith::ConstantIndexOp::create(b, loc, val);
}

// Forward declarations
Value applyLayout(OpBuilder &b, Location loc, Value layout, ValueRange indices);
SmallVector<Value> applyInverseLayout(OpBuilder &b, Location loc, Value layout,
                                      Value flatIndex);

// ============================================================================
// flatten_index / unflatten_index  (Python L93-136)
//
// flatten_index((i0,...,iN), (d0,...,dN)) = sum_k( ik * prod(d_{k+1}..d_N) )
// unflatten_index(flat, (d0,...,dN)) => (i0,...,iN)
// ============================================================================

Value flattenIndex(OpBuilder &b, Location loc, ValueRange indices,
                   ValueRange dims) {
  Value flat = getConstantIndex(b, loc, 0);
  Value multiplier = getConstantIndex(b, loc, 1);

  // Iterate backwards: flat = i_N + d_N * (...)
  for (int k = dims.size() - 1; k >= 0; --k) {
    Value idx = indices[k];
    Value term = arith::MulIOp::create(b, loc, idx, multiplier);
    flat = arith::AddIOp::create(b, loc, flat, term);

    if (k > 0) {
      Value dim = dims[k];
      multiplier = arith::MulIOp::create(b, loc, multiplier, dim);
    }
  }
  return flat;
}

SmallVector<Value> unflattenIndex(OpBuilder &b, Location loc, Value flatIndex,
                                  ValueRange dims) {
  SmallVector<Value> indices;
  int rank = dims.size();
  Value current = flatIndex;

  for (int k = 0; k < rank; ++k) {
    Value strideVal = getConstantIndex(b, loc, 1);
    for (int j = k + 1; j < rank; ++j) {
      strideVal = arith::MulIOp::create(b, loc, strideVal, dims[j]);
    }
    Value idx = arith::DivUIOp::create(b, loc, current, strideVal);
    indices.push_back(idx);
    current = arith::RemUIOp::create(b, loc, current, strideVal);
  }
  return indices;
}
// ============================================================================
// RegP apply/inv
// ============================================================================

Value applyRegP(OpBuilder &b, Location loc, RegPOp op, ValueRange indices) {
  auto perm = extractI64Array(op.getPerm());
  auto dims = op.getDims();

  auto permIndices = sigmaValues(indices, perm);
  auto permDims = sigmaValues(dims, perm);

  return flattenIndex(b, loc, permIndices, permDims);
}

SmallVector<Value> applyInverseRegP(OpBuilder &b, Location loc, RegPOp op,
                                    Value flatIndex) {
  auto perm = extractI64Array(op.getPerm());
  auto dims = op.getDims();

  auto permDims = sigmaValues(dims, perm);
  auto permIndices = unflattenIndex(b, loc, flatIndex, permDims);
  auto invPerm = inversePermutation(perm);
  return sigmaValues(permIndices, invPerm);
}


// ============================================================================
// GenP apply/inv
// ============================================================================

Value applyGenP(OpBuilder &b, Location loc, GenPOp op, ValueRange indices) {
  Block &block = op.getBody().front();
  if (block.getNumArguments() != indices.size())
    return nullptr;
  
  IRMapping mapping;
  for (unsigned i = 0; i < indices.size(); ++i)
    mapping.map(block.getArgument(i), indices[i]);

  for (auto &opInst : block.without_terminator())
    b.clone(opInst, mapping);

  Operation *term = block.getTerminator();
  if (term->getNumOperands() == 1)
    return mapping.lookup(term->getOperand(0));

  return nullptr;
}

SmallVector<Value> applyInverseGenP(OpBuilder &b, Location loc, GenPOp op,
                                    Value flatIndex) {
  Region &invBody = op.getInvBody();
  if (invBody.empty())
    return {};

  Block &block = invBody.front();
  if (block.getNumArguments() != 1)
    return {};

  IRMapping mapping;
  mapping.map(block.getArgument(0), flatIndex);

  for (auto &opInst : block.without_terminator())
    b.clone(opInst, mapping);

  Operation *term = block.getTerminator();
  SmallVector<Value> results;
  for (Value operand : term->getOperands())
    results.push_back(mapping.lookup(operand));
  return results;
}

// ============================================================================
// OrderBy apply/inv
// ============================================================================

Value applyOrderBy(OpBuilder &b, Location loc, OrderByOp op,
                   ValueRange indices) {
  Value flatIndex = getConstantIndex(b, loc, 0);
  int offset = 0;

  for (Value perm : op.getPerms()) {
    SmallVector<Value> pDims = getLayoutInputDims(perm);
    if (pDims.empty())
      return nullptr;

    int count = pDims.size();
    if (offset + count > (int)indices.size())
      return nullptr;

    ValueRange slice = indices.slice(offset, count);
    offset += count;

    Value innerFlat = applyLayout(b, loc, perm, slice);
    if (!innerFlat)
      return nullptr;

    Value sizeVal = getConstantIndex(b, loc, 1);
    for (auto d : pDims)
      sizeVal = arith::MulIOp::create(b, loc, sizeVal, d);

    Value flatMul = arith::MulIOp::create(b, loc, flatIndex, sizeVal);
    flatIndex = arith::AddIOp::create(b, loc, flatMul, innerFlat);
  }
  return flatIndex;
}

SmallVector<Value> applyInverseOrderBy(OpBuilder &b, Location loc,
                                       OrderByOp op, Value flatIndex) {
  SmallVector<Value> allIndices;

  SmallVector<Value> permsVec(op.getPerms().begin(), op.getPerms().end());

  for (auto it = permsVec.rbegin(); it != permsVec.rend(); ++it) {
    Value perm = *it;
    SmallVector<Value> pDims = getLayoutInputDims(perm);
    Value sizeVal = getConstantIndex(b, loc, 1);
    for (auto d : pDims)
      sizeVal = arith::MulIOp::create(b, loc, sizeVal, d);

    Value innerFlat = arith::RemUIOp::create(b, loc, flatIndex, sizeVal);
    flatIndex = arith::DivUIOp::create(b, loc, flatIndex, sizeVal);

    SmallVector<Value> innerIndices =
        applyInverseLayout(b, loc, perm, innerFlat);
    if (innerIndices.empty())
      return {};

    allIndices.insert(allIndices.begin(), innerIndices.begin(),
                      innerIndices.end());
  }
  return allIndices;
}

// ============================================================================
// GroupBy apply/inv
// ============================================================================

Value applyGroupBy(OpBuilder &b, Location loc, GroupByOp op,
                   ValueRange indices) {
  auto groupDims = op.getGroupDims();
  Value current = flattenIndex(b, loc, indices, groupDims);

  // Iterate objects in REVERSE (matching Python)
  SmallVector<Value> objectsVec(op.getObjects().begin(),
                                op.getObjects().end());

  for (auto it = objectsVec.rbegin(); it != objectsVec.rend(); ++it) {
    Value obj = *it;
    SmallVector<Value> objDims = getLayoutInputDims(obj);
    if (objDims.empty())
      return nullptr;
    SmallVector<Value> objIndices = unflattenIndex(b, loc, current, objDims);
    current = applyLayout(b, loc, obj, objIndices);
    if (!current)
      return nullptr;
  }
  return current;
}



SmallVector<Value> applyInverseGroupBy(OpBuilder &b, Location loc,
                                       GroupByOp op, Value flatIndex) {
  auto groupDims = op.getGroupDims();
  Value current = flatIndex;

  // Iterate objects FORWARD (matching Python)
  for (Value obj : op.getObjects()) {
    SmallVector<Value> objDims = getLayoutInputDims(obj);
    if (objDims.empty())
      return {};
    SmallVector<Value> idxFromObj = applyInverseLayout(b, loc, obj, current);
    if (idxFromObj.empty())
      return {};
    current = flattenIndex(b, loc, idxFromObj, objDims);
  }
  return unflattenIndex(b, loc, current, groupDims);
}



// ============================================================================
// Dispatcher — applyLayout / applyInverseLayout
// ============================================================================

Value applyLayout(OpBuilder &b, Location loc, Value layout,
                  ValueRange indices) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp)
    return nullptr;

  if (auto regPOp = dyn_cast<RegPOp>(defOp))
    return applyRegP(b, loc, regPOp, indices);

  if (auto genPOp = dyn_cast<GenPOp>(defOp))
    return applyGenP(b, loc, genPOp, indices);

  if (auto orderByOp = dyn_cast<OrderByOp>(defOp))
    return applyOrderBy(b, loc, orderByOp, indices);

  if (auto groupByOp = dyn_cast<GroupByOp>(defOp))
    return applyGroupBy(b, loc, groupByOp, indices);

  return nullptr;
}

SmallVector<Value> applyInverseLayout(OpBuilder &b, Location loc, Value layout,
                                      Value flatIndex) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp)
    return {};

  if (auto regPOp = dyn_cast<RegPOp>(defOp))
    return applyInverseRegP(b, loc, regPOp, flatIndex);

  if (auto genPOp = dyn_cast<GenPOp>(defOp))
    return applyInverseGenP(b, loc, genPOp, flatIndex);

  if (auto orderByOp = dyn_cast<OrderByOp>(defOp))
    return applyInverseOrderBy(b, loc, orderByOp, flatIndex);

  if (auto groupByOp = dyn_cast<GroupByOp>(defOp))
    return applyInverseGroupBy(b, loc, groupByOp, flatIndex);

  return {};
}

// ============================================================================
// Rewrite Patterns
// ============================================================================

struct ApplyOpLowering : public OpRewritePattern<ApplyOp> {
  using OpRewritePattern<ApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyOp op,
                                PatternRewriter &rewriter) const override {
    Value res =
        applyLayout(rewriter, op.getLoc(), op.getLayout(), op.getIndices());
    if (!res)
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ApplyInverseOpLowering : public OpRewritePattern<ApplyInverseOp> {
  using OpRewritePattern<ApplyInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyInverseOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> res = applyInverseLayout(
        rewriter, op.getLoc(), op.getLayout(), op.getFlatIndex());
    if (res.empty())
      return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct LoadOpLowering : public OpRewritePattern<LoadOp> {
  using OpRewritePattern<LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value view = op.getView();
    auto castOp = view.getDefiningOp<CastViewOp>();
    if (!castOp) {
      return op.emitOpError(
          "expected view to be defined by lego.cast_view for now");
    }

    Value memref = castOp.getMemref();
    Value layout = castOp.getLayout();
    ValueRange indices = op.getIndices();

    Value flatIndex = applyLayout(rewriter, op.getLoc(), layout, indices);
    if (!flatIndex)
      return failure();

    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, memref,
                                                 ValueRange{flatIndex});
    return success();
  }
};

struct StoreOpLowering : public OpRewritePattern<StoreOp> {
  using OpRewritePattern<StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(StoreOp op,
                                PatternRewriter &rewriter) const override {
    Value view = op.getView();
    auto castOp = view.getDefiningOp<CastViewOp>();
    if (!castOp)
      return failure();

    Value memref = castOp.getMemref();
    Value layout = castOp.getLayout();
    ValueRange indices = op.getIndices();
    Value value = op.getValue();

    Value flatIndex = applyLayout(rewriter, op.getLoc(), layout, indices);
    if (!flatIndex)
      return failure();

    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, memref,
                                                 ValueRange{flatIndex});
    return success();
  }
};

struct CastViewOpLowering : public OpRewritePattern<CastViewOp> {
  using OpRewritePattern<CastViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastViewOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// ============================================================================
// Pass
// ============================================================================

struct LegoToArithPassImpl
    : public mlir::lego::impl::LegoToArithPassBase<LegoToArithPassImpl> {
  using mlir::lego::impl::LegoToArithPassBase<
      LegoToArithPassImpl>::LegoToArithPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ApplyOpLowering, ApplyInverseOpLowering, LoadOpLowering,
                 StoreOpLowering, CastViewOpLowering>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoToArithPass() {
  return std::make_unique<LegoToArithPassImpl>();
}
} // namespace lego
} // namespace mlir
