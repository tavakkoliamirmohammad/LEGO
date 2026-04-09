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
#include "mlir/IR/Matchers.h"
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
  int rank = dims.size();

  // Pre-compute strides right-to-left so the multiplication tree matches
  // flattenIndex exactly.  After CSE the stride SSA values are shared,
  // enabling SimplifyDivId / Extended-SimplifyDivId to fire.
  SmallVector<Value> strides(rank);
  strides[rank - 1] = getConstantIndex(b, loc, 1);
  for (int k = rank - 2; k >= 0; --k)
    strides[k] = arith::MulIOp::create(b, loc, strides[k + 1], dims[k + 1]);

  SmallVector<Value> indices;
  Value current = flatIndex;

  for (int k = 0; k < rank; ++k) {
    // When a dim is constant 1, the index is always 0 and the remainder
    // is unchanged.  This avoids emitting redundant div/rem ops that
    // would be hard to simplify away for symbolic expressions.
    APInt dimVal;
    if (matchPattern(dims[k], m_ConstantInt(&dimVal)) && dimVal == 1) {
      indices.push_back(getConstantIndex(b, loc, 0));
      continue;  // current stays the same
    }

    Value idx = arith::DivUIOp::create(b, loc, current, strides[k]);
    indices.push_back(idx);
    current = arith::RemUIOp::create(b, loc, current, strides[k]);
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

    Value innerFlat;
    bool isFirstPerm = (std::next(it) == permsVec.rend());
    if (isFirstPerm) {
      // For the first (outermost) perm, flatIndex is already in
      // [0, sizeVal) because all subsequent perm sizes have been
      // divided out.  Skipping the remui avoids a redundant modulo
      // that is hard to simplify away for symbolic dimensions.
      innerFlat = flatIndex;
    } else {
      innerFlat = arith::RemUIOp::create(b, loc, flatIndex, sizeVal);
      flatIndex = arith::DivUIOp::create(b, loc, flatIndex, sizeVal);
    }

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
// TileBy apply/inv — direct lowering without GroupBy intermediaries
//
// Operates per-dimension using flattenIndex / unflattenIndex, then
// delegates to the inner layout's apply / inv.
// ============================================================================

/// Collect the k-th dim from each inner perm → [P0.dim[k], P1.dim[k], ...].
static SmallVector<Value> getPerPermDimsAtK(OrderByOp ob, int k) {
  SmallVector<Value> dims;
  for (Value perm : ob.getPerms())
    dims.push_back(getLayoutInputDims(perm)[k]);
  return dims;
}

/// Collect the k-th tile dim from each level → [T[0][k], T[1][k], ...].
static SmallVector<Value> getTileDimsAtK(ValueRange tileDims, int d,
                                          int q_tile, int k) {
  SmallVector<Value> dims;
  for (int l = 0; l < q_tile; l++)
    dims.push_back(tileDims[l * d + k]);
  return dims;
}

/// Check if TileBy is identity (tile dims match inner perm dims exactly).
static bool isTileByIdentity(TileByOp op, OrderByOp innerOB,
                              int d, int q_tile) {
  if (!innerOB || q_tile != (int)innerOB.getPerms().size())
    return false;
  auto tileDims = op.getTileDims();
  for (int h = 0; h < q_tile; h++) {
    auto permDims = getLayoutInputDims(innerOB.getPerms()[h]);
    for (int k = 0; k < d; k++)
      if (tileDims[h * d + k] != permDims[k])
        return false;
  }
  return true;
}

Value applyTileBy(OpBuilder &b, Location loc, TileByOp op,
                  ValueRange indices) {
  auto tileShape = extractI64Array(op.getTileShape());
  int64_t d = tileShape[0];
  int64_t q_tile = tileShape.size();
  auto tileDims = op.getTileDims();
  Value inner = op.getInput();
  auto innerOB = inner.getDefiningOp<OrderByOp>();
  int q_inner = innerOB ? (int)innerOB.getPerms().size() : 1;

  if (isTileByIdentity(op, innerOB, d, q_tile))
    return applyLayout(b, loc, inner, indices);

  // Step 1: Per dimension k, combine tile levels via flattenIndex.
  SmallVector<Value> combined(d);
  for (int k = 0; k < d; k++) {
    SmallVector<Value> levelsAtK, dimsAtK = getTileDimsAtK(tileDims, d, q_tile, k);
    for (int l = 0; l < q_tile; l++)
      levelsAtK.push_back(indices[l * d + k]);
    combined[k] = flattenIndex(b, loc, levelsAtK, dimsAtK);
  }

  // Step 2: If q_inner > 1, decompose each combined[k] into per-perm
  // sub-indices via unflattenIndex, then reorder to perm-grouped.
  SmallVector<Value> innerIndices;
  if (!innerOB || q_inner == 1) {
    innerIndices.assign(combined.begin(), combined.end());
  } else {
    SmallVector<SmallVector<Value>> perPerm(q_inner);
    for (int k = 0; k < d; k++) {
      auto sub = unflattenIndex(b, loc, combined[k], getPerPermDimsAtK(innerOB, k));
      for (int h = 0; h < q_inner; h++)
        perPerm[h].push_back(sub[h]);
    }
    for (int h = 0; h < q_inner; h++)
      innerIndices.append(perPerm[h].begin(), perPerm[h].end());
  }

  // Step 3: Apply inner layout.
  return applyLayout(b, loc, inner, innerIndices);
}

SmallVector<Value> applyInverseTileBy(OpBuilder &b, Location loc, TileByOp op,
                                      Value flatIndex) {
  auto tileShape = extractI64Array(op.getTileShape());
  int64_t d = tileShape[0];
  int64_t q_tile = tileShape.size();
  auto tileDims = op.getTileDims();
  Value inner = op.getInput();
  auto innerOB = inner.getDefiningOp<OrderByOp>();
  int q_inner = innerOB ? (int)innerOB.getPerms().size() : 1;

  if (isTileByIdentity(op, innerOB, d, q_tile))
    return applyInverseLayout(b, loc, inner, flatIndex);

  // Step 1: Apply inner layout inverse.
  SmallVector<Value> innerIndices = applyInverseLayout(b, loc, inner, flatIndex);
  if (innerIndices.empty())
    return {};

  // Step 2: If q_inner > 1, recombine per-perm sub-indices per dimension
  // via flattenIndex.
  SmallVector<Value> combined(d);
  if (!innerOB || q_inner == 1) {
    for (int k = 0; k < d; k++)
      combined[k] = innerIndices[k];
  } else {
    for (int k = 0; k < d; k++) {
      SmallVector<Value> subAtK, dimsAtK = getPerPermDimsAtK(innerOB, k);
      for (int h = 0; h < q_inner; h++)
        subAtK.push_back(innerIndices[h * d + k]);
      combined[k] = flattenIndex(b, loc, subAtK, dimsAtK);
    }
  }

  // Step 3: Decompose each combined[k] into tile-level indices
  // via unflattenIndex.
  SmallVector<Value> result(d * q_tile);
  for (int k = 0; k < d; k++) {
    auto tileAtK = unflattenIndex(b, loc, combined[k],
                                  getTileDimsAtK(tileDims, d, q_tile, k));
    for (int l = 0; l < q_tile; l++)
      result[l * d + k] = tileAtK[l];
  }

  return result;
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

  if (auto tileByOp = dyn_cast<TileByOp>(defOp))
    return applyTileBy(b, loc, tileByOp, indices);

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

  if (auto tileByOp = dyn_cast<TileByOp>(defOp))
    return applyInverseTileBy(b, loc, tileByOp, flatIndex);

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
// View argument resolution
//
// When a function takes !lego.view arguments, LoadOp/StoreOp lowering
// can't find the CastViewOp because the view is a block argument.
// This pre-pass resolves this by expanding function signatures to
// include the underlying memref and layout, then inserting CastViewOp
// at the function entry.
// ============================================================================

/// Find all call sites for a given function in the module.
static SmallVector<func::CallOp> findCallSites(ModuleOp module,
                                                func::FuncOp func) {
  SmallVector<func::CallOp> calls;
  StringRef funcName = func.getSymName();
  module.walk([&](func::CallOp call) {
    if (call.getCallee() == funcName)
      calls.push_back(call);
  });
  return calls;
}

/// Collect the transitive set of ops that define `root`, stopping at
/// block arguments.  Returns ops in topological order.
static SmallVector<Operation *> collectDefChain(Value root) {
  SmallVector<Operation *> result;
  SmallVector<Value> worklist = {root};
  DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    if (!def || visited.count(def))
      continue;
    visited.insert(def);
    result.push_back(def);
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  // Reverse for topological order (operands before users).
  std::reverse(result.begin(), result.end());
  return result;
}

/// Resolve view function arguments by expanding signatures.
/// For each function arg of type !lego.view, if all callers pass a
/// CastViewOp result, expand the signature to include a memref arg,
/// clone the layout-defining ops into the callee, and insert a
/// CastViewOp at the function entry.
static void resolveViewArguments(ModuleOp module) {
  SmallVector<func::FuncOp> funcsToProcess;
  module.walk([&](func::FuncOp func) {
    for (Type argTy : func.getArgumentTypes()) {
      if (isa<ViewType>(argTy)) {
        funcsToProcess.push_back(func);
        break;
      }
    }
  });

  for (func::FuncOp func : funcsToProcess) {
    if (func.isDeclaration())
      continue;

    auto calls = findCallSites(module, func);
    if (calls.empty())
      continue;

    // Collect view argument positions (skip already-resolved ones).
    SmallVector<unsigned> viewArgIndices;
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      if (!isa<ViewType>(func.getFunctionType().getInput(i)))
        continue;
      BlockArgument arg = func.getArgument(i);
      // Skip if already resolved: arg has no remaining uses (CastViewOp
      // was lowered) or its sole use is an inserted CastViewOp.
      if (arg.use_empty())
        continue;
      bool alreadyResolved = false;
      if (arg.hasOneUse()) {
        if (isa<CastViewOp>(*arg.user_begin()))
          alreadyResolved = true;
      }
      if (!alreadyResolved)
        viewArgIndices.push_back(i);
    }

    // For each view arg, check that ALL callers pass a CastViewOp result.
    struct ViewArgInfo {
      unsigned argIdx;
      Type memrefType;
      CastViewOp firstCastOp; // Used to clone layout ops.
    };
    SmallVector<ViewArgInfo> resolvableArgs;

    for (unsigned viewIdx : viewArgIndices) {
      bool allCallersOk = true;
      Type memrefType;
      CastViewOp firstCast;

      for (auto call : calls) {
        Value operand = call.getOperand(viewIdx);
        auto castOp = operand.getDefiningOp<CastViewOp>();
        if (!castOp) {
          allCallersOk = false;
          break;
        }
        if (!memrefType) {
          memrefType = castOp.getMemref().getType();
          firstCast = castOp;
        }
      }

      if (allCallersOk && memrefType)
        resolvableArgs.push_back({viewIdx, memrefType, firstCast});
    }

    if (resolvableArgs.empty())
      continue;

    // Build new function type: original args + memref per view arg.
    SmallVector<Type> newArgTypes(func.getArgumentTypes());
    SmallVector<unsigned> memrefArgPositions;

    for (auto &info : resolvableArgs) {
      memrefArgPositions.push_back(newArgTypes.size());
      newArgTypes.push_back(info.memrefType);
    }

    auto newFuncType = FunctionType::get(func.getContext(), newArgTypes,
                                         func.getResultTypes());

    // Add new block arguments.
    Block &entryBlock = func.getBody().front();
    for (auto &info : resolvableArgs)
      entryBlock.addArgument(info.memrefType, func.getLoc());

    func.setFunctionType(newFuncType);

    // At the function entry, clone layout-defining ops from the first
    // call site and create a CastViewOp for each resolved view arg.
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&entryBlock);

    for (unsigned i = 0; i < resolvableArgs.size(); ++i) {
      unsigned viewArgIdx = resolvableArgs[i].argIdx;
      unsigned memrefPos = memrefArgPositions[i];
      CastViewOp srcCast = resolvableArgs[i].firstCastOp;

      Value memrefArg = entryBlock.getArgument(memrefPos);
      Value viewArg = entryBlock.getArgument(viewArgIdx);

      // Clone the layout def chain into the callee.
      Value srcLayout = srcCast.getLayout();
      auto defChain = collectDefChain(srcLayout);

      IRMapping mapping;
      // Map block args in the source that are arith constants — they'll
      // be cloned.  Block args in the source that are function params
      // need corresponding params in the callee (handled by the existing
      // arg passing).
      for (Operation *op : defChain) {
        Operation *cloned = builder.clone(*op, mapping);
        for (unsigned r = 0; r < op->getNumResults(); ++r)
          mapping.map(op->getResult(r), cloned->getResult(r));
      }

      Value clonedLayout = mapping.lookupOrDefault(srcLayout);
      auto newCastView = CastViewOp::create(
          builder, func.getLoc(), viewArg.getType(), memrefArg, clonedLayout);
      viewArg.replaceAllUsesWith(newCastView.getResult());
    }

    // Update all call sites to pass extra memref arguments.
    for (auto call : calls) {
      SmallVector<Value> newOperands(call.getOperands());
      for (auto &info : resolvableArgs) {
        auto castOp = call.getOperand(info.argIdx).getDefiningOp<CastViewOp>();
        newOperands.push_back(castOp.getMemref());
      }
      OpBuilder callBuilder(call);
      auto newCall = func::CallOp::create(callBuilder, call.getLoc(),
                                          call.getCallee(),
                                          call.getResultTypes(), newOperands);
      call.replaceAllUsesWith(newCall.getResults());
      call.erase();
    }
  }
}

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

    // Resolve view function arguments before pattern-based lowering.
    resolveViewArguments(module);

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
