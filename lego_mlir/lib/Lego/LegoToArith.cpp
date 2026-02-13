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

using namespace mlir;
using namespace mlir::lego;

namespace {

// ============================================================================
// Helpers
// ============================================================================

Value getConstantIndex(OpBuilder &b, Location loc, int64_t val) {
  return b.create<arith::ConstantIndexOp>(loc, val);
}

// Forward declarations
Value applyLayout(OpBuilder &b, Location loc, Value layout, ValueRange indices);
SmallVector<Value> applyInverseLayout(OpBuilder &b, Location loc, Value layout,
                                      Value flatIndex);

// Extract I64ArrayAttr into SmallVector<int64_t>
SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  for (auto a : attr)
    result.push_back(cast<IntegerAttr>(a).getInt());
  return result;
}

// ============================================================================
// flatten_index / unflatten_index  (Python L93-136)
//
// flatten_index((i0,...,iN), (d0,...,dN)) = sum_k( ik * prod(d_{k+1}..d_N) )
// unflatten_index(flat, (d0,...,dN)) => (i0,...,iN)
// ============================================================================

Value flattenIndex(OpBuilder &b, Location loc, ValueRange indices,
                   ArrayRef<int64_t> dims) {
  Value flat = getConstantIndex(b, loc, 0);
  int rank = indices.size();

  for (int k = 0; k < rank; ++k) {
    int64_t stride = 1;
    for (int j = k + 1; j < rank; ++j)
      stride *= dims[j];
    Value strideVal = getConstantIndex(b, loc, stride);
    Value term = b.create<arith::MulIOp>(loc, indices[k], strideVal);
    flat = b.create<arith::AddIOp>(loc, flat, term);
  }
  return flat;
}

SmallVector<Value> unflattenIndex(OpBuilder &b, Location loc, Value flatIndex,
                                  ArrayRef<int64_t> dims) {
  SmallVector<Value> indices;
  int rank = dims.size();
  Value current = flatIndex;

  for (int k = 0; k < rank; ++k) {
    int64_t stride = 1;
    for (int j = k + 1; j < rank; ++j)
      stride *= dims[j];
    Value strideVal = getConstantIndex(b, loc, stride);
    Value idx = b.create<arith::DivUIOp>(loc, current, strideVal);
    indices.push_back(idx);
    current = b.create<arith::RemUIOp>(loc, current, strideVal);
  }
  return indices;
}

// ============================================================================
// sigma / inverse_permutation  (Python L181-194)
// ============================================================================

// sigma(values, perm) => new_arr[i] = values[perm[i]]
template <typename T>
SmallVector<T> sigma(ArrayRef<T> values, ArrayRef<int64_t> perm) {
  SmallVector<T> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    result[i] = values[perm[i]];
  return result;
}

SmallVector<Value> sigmaValues(ValueRange values, ArrayRef<int64_t> perm) {
  SmallVector<Value> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    result[i] = values[perm[i]];
  return result;
}

SmallVector<int64_t> inversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inv(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    inv[perm[i]] = i;
  return inv;
}

// get_sigma_perm(d, q) => [k + d*h for h in range(q) for k in range(d)]
// Wait, Python is: [[k + d*h for h in range(q)] for k in range(d)]
// flattened via sum(..., [])
SmallVector<int64_t> getSigmaPerm(int d, int q) {
  SmallVector<int64_t> result;
  for (int k = 0; k < d; ++k)
    for (int h = 0; h < q; ++h)
      result.push_back(k + d * h);
  return result;
}

// ============================================================================
// getLayoutInputDims — recursively determine input dimensions for a layout
// ============================================================================

SmallVector<int64_t> getLayoutInputDims(Value layout) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp)
    return {};

  if (auto regPOp = dyn_cast<RegPOp>(defOp))
    return extractI64Array(regPOp.getDims());

  if (auto rowOp = dyn_cast<RowOp>(defOp))
    return extractI64Array(rowOp.getDims());

  if (auto colOp = dyn_cast<ColOp>(defOp))
    return extractI64Array(colOp.getDims());

  if (auto genPOp = dyn_cast<GenPOp>(defOp))
    return extractI64Array(genPOp.getDims());

  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
    SmallVector<int64_t> allDims;
    for (Value perm : orderByOp.getPerms()) {
      auto pDims = getLayoutInputDims(perm);
      allDims.append(pDims.begin(), pDims.end());
    }
    return allDims;
  }

  if (auto groupByOp = dyn_cast<GroupByOp>(defOp))
    return extractI64Array(groupByOp.getGroupDims());

  if (auto tileByOp = dyn_cast<TileByOp>(defOp)) {
    // TileBy input dims = the tile_sizes count (one dim per tile)
    return extractI64Array(tileByOp.getTileSizes());
  }

  return {};
}

// ============================================================================
// RegP  (Python L197-228)
//
// apply(idx) = flatten(sigma(idx, perm), sigma(dims, perm))
// inv(flat)  = sigma(unflatten(flat, sigma(dims, perm)), inverse_perm)
// ============================================================================

Value applyRegP(OpBuilder &b, Location loc, RegPOp op, ValueRange indices) {
  auto dims = extractI64Array(op.getDims());
  auto perm = extractI64Array(op.getPerm());

  auto permutedIndices = sigmaValues(indices, perm);
  auto permutedDims = sigma<int64_t>(dims, perm);

  return flattenIndex(b, loc, permutedIndices, permutedDims);
}

SmallVector<Value> applyInverseRegP(OpBuilder &b, Location loc, RegPOp op,
                                    Value flatIndex) {
  auto dims = extractI64Array(op.getDims());
  auto perm = extractI64Array(op.getPerm());

  auto permutedDims = sigma<int64_t>(dims, perm);
  auto permutedIndices = unflattenIndex(b, loc, flatIndex, permutedDims);

  auto invPerm = inversePermutation(perm);
  return sigmaValues(permutedIndices, invPerm);
}

// ============================================================================
// Row  (Python L231-234)
//
// Row(*dims) = RegP(dims, identity_perm)
// apply(idx) = flatten(idx, dims)     [identity perm is no-op]
// inv(flat)  = unflatten(flat, dims)
// ============================================================================

Value applyRow(OpBuilder &b, Location loc, RowOp op, ValueRange indices) {
  auto dims = extractI64Array(op.getDims());
  // Identity perm: flatten(idx, dims)
  return flattenIndex(b, loc, indices, dims);
}

SmallVector<Value> applyInverseRow(OpBuilder &b, Location loc, RowOp op,
                                   Value flatIndex) {
  auto dims = extractI64Array(op.getDims());
  return unflattenIndex(b, loc, flatIndex, dims);
}

// ============================================================================
// Col  (Python L237-240)
//
// Col(*dims) = RegP(dims, reversed_identity)
// apply(idx) = flatten(sigma(idx, rev), sigma(dims, rev))
// inv(flat)  = sigma(unflatten(flat, sigma(dims, rev)), inv(rev))
// ============================================================================

Value applyCol(OpBuilder &b, Location loc, ColOp op, ValueRange indices) {
  auto dims = extractI64Array(op.getDims());
  int d = dims.size();
  // Reversed identity: [d-1, d-2, ..., 0]
  SmallVector<int64_t> revPerm;
  for (int i = d - 1; i >= 0; --i)
    revPerm.push_back(i);

  auto permutedIndices = sigmaValues(indices, revPerm);
  auto permutedDims = sigma<int64_t>(dims, revPerm);

  return flattenIndex(b, loc, permutedIndices, permutedDims);
}

SmallVector<Value> applyInverseCol(OpBuilder &b, Location loc, ColOp op,
                                   Value flatIndex) {
  auto dims = extractI64Array(op.getDims());
  int d = dims.size();
  SmallVector<int64_t> revPerm;
  for (int i = d - 1; i >= 0; --i)
    revPerm.push_back(i);

  auto permutedDims = sigma<int64_t>(dims, revPerm);
  auto permutedIndices = unflattenIndex(b, loc, flatIndex, permutedDims);

  auto invPerm = inversePermutation(revPerm);
  return sigmaValues(permutedIndices, invPerm);
}

// ============================================================================
// GenP  (Python L150-178)
//
// apply: inlines the body region
// inv:   NOT YET IMPLEMENTED (requires a second region)
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

// ============================================================================
// OrderBy  (Python L243-362)
//
// apply(idx):
//   flat = 0; offset = 0
//   for perm in self.perms:
//     dim_count = len(perm.dims())
//     total = product(perm.dims())
//     icur = idx[offset : offset + dim_count]
//     iflat_cur = perm.apply(icur)
//     flat = flat * total + iflat_cur
//     offset += dim_count
//   return flat
//
// inv(flat):
//   indices = []
//   for perm in reversed(self.perms):
//     total = product(perm.dims())
//     iflat_cur = flat % total
//     flat = flat // total
//     indices = perm.inv(iflat_cur) + indices   [prepend]
//   return indices
// ============================================================================

Value applyOrderBy(OpBuilder &b, Location loc, OrderByOp op,
                   ValueRange indices) {
  Value flatIndex = getConstantIndex(b, loc, 0);
  int offset = 0;

  for (Value perm : op.getPerms()) {
    SmallVector<int64_t> pDims = getLayoutInputDims(perm);
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

    int64_t size = 1;
    for (auto d : pDims)
      size *= d;
    Value sizeVal = getConstantIndex(b, loc, size);

    Value flatMul = b.create<arith::MulIOp>(loc, flatIndex, sizeVal);
    flatIndex = b.create<arith::AddIOp>(loc, flatMul, innerFlat);
  }
  return flatIndex;
}

SmallVector<Value> applyInverseOrderBy(OpBuilder &b, Location loc,
                                       OrderByOp op, Value flatIndex) {
  SmallVector<Value> allIndices;

  SmallVector<Value> permsVec(op.getPerms().begin(), op.getPerms().end());

  for (auto it = permsVec.rbegin(); it != permsVec.rend(); ++it) {
    Value perm = *it;
    SmallVector<int64_t> pDims = getLayoutInputDims(perm);
    int64_t size = 1;
    for (auto d : pDims)
      size *= d;
    Value sizeVal = getConstantIndex(b, loc, size);

    Value innerFlat = b.create<arith::RemUIOp>(loc, flatIndex, sizeVal);
    flatIndex = b.create<arith::DivUIOp>(loc, flatIndex, sizeVal);

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
// GroupBy  (Python L365-437)
//
// apply(*idx):
//   current = flatten_index(idx, self._dims)
//   for obj in reversed(self.objects):
//     obj_dims = obj.dims()
//     idx_for_obj = unflatten_index(current, obj_dims)
//     current = obj.apply(idx_for_obj)
//   return current
//
// inv(flat_idx):
//   current = flat_idx
//   for obj in self.objects:
//     obj_dims = obj.dims()
//     idx_from_obj = obj.inv(current)
//     current = flatten_index(idx_from_obj, obj_dims)
//   original_idx = unflatten_index(current, self._dims)
//   return original_idx
// ============================================================================

Value applyGroupBy(OpBuilder &b, Location loc, GroupByOp op,
                   ValueRange indices) {
  auto groupDims = extractI64Array(op.getGroupDims());
  Value current = flattenIndex(b, loc, indices, groupDims);

  // Iterate objects in REVERSE (matching Python)
  SmallVector<Value> objectsVec(op.getObjects().begin(),
                                op.getObjects().end());

  for (auto it = objectsVec.rbegin(); it != objectsVec.rend(); ++it) {
    Value obj = *it;
    SmallVector<int64_t> objDims = getLayoutInputDims(obj);
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
  auto groupDims = extractI64Array(op.getGroupDims());
  Value current = flatIndex;

  // Iterate objects FORWARD (matching Python)
  for (Value obj : op.getObjects()) {
    SmallVector<int64_t> objDims = getLayoutInputDims(obj);
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
// TileBy  (Python L270-286 — method on OrderBy)
//
// For now, TileBy splits each dim into (idx/size, idx%size) and passes
// to inner layout. The full Python TileBy creates a GroupBy+OrderBy chain
// which requires the OrderBy chain context.
// ============================================================================

Value applyTileBy(OpBuilder &b, Location loc, TileByOp op,
                  ValueRange indices) {
  auto sizes = extractI64Array(op.getTileSizes());

  SmallVector<Value> tiledIndices;
  for (size_t i = 0; i < indices.size(); ++i) {
    Value idx = indices[i];
    int64_t s = sizes[i];
    Value sizeVal = getConstantIndex(b, loc, s);
    Value q = b.create<arith::DivUIOp>(loc, idx, sizeVal);
    Value r = b.create<arith::RemUIOp>(loc, idx, sizeVal);
    tiledIndices.push_back(q);
    tiledIndices.push_back(r);
  }
  return applyLayout(b, loc, op.getInput(), tiledIndices);
}

SmallVector<Value> applyInverseTileBy(OpBuilder &b, Location loc, TileByOp op,
                                      Value flatIndex) {
  SmallVector<Value> innerIndices =
      applyInverseLayout(b, loc, op.getInput(), flatIndex);
  if (innerIndices.empty())
    return {};

  auto sizes = extractI64Array(op.getTileSizes());

  SmallVector<Value> originalIndices;
  for (size_t i = 0; i < sizes.size(); ++i) {
    Value q = innerIndices[2 * i];
    Value r = innerIndices[2 * i + 1];
    int64_t s = sizes[i];
    Value sizeVal = getConstantIndex(b, loc, s);
    Value qs = b.create<arith::MulIOp>(loc, q, sizeVal);
    Value idx = b.create<arith::AddIOp>(loc, qs, r);
    originalIndices.push_back(idx);
  }
  return originalIndices;
}

// ============================================================================
// Dispatcher — applyLayout / applyInverseLayout
// ============================================================================

Value applyLayout(OpBuilder &b, Location loc, Value layout,
                  ValueRange indices) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp)
    return nullptr;

  if (auto rowOp = dyn_cast<RowOp>(defOp))
    return applyRow(b, loc, rowOp, indices);

  if (auto colOp = dyn_cast<ColOp>(defOp))
    return applyCol(b, loc, colOp, indices);

  if (auto regPOp = dyn_cast<RegPOp>(defOp))
    return applyRegP(b, loc, regPOp, indices);

  if (auto genPOp = dyn_cast<GenPOp>(defOp))
    return applyGenP(b, loc, genPOp, indices);

  if (auto tileByOp = dyn_cast<TileByOp>(defOp))
    return applyTileBy(b, loc, tileByOp, indices);

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

  if (auto rowOp = dyn_cast<RowOp>(defOp))
    return applyInverseRow(b, loc, rowOp, flatIndex);

  if (auto colOp = dyn_cast<ColOp>(defOp))
    return applyInverseCol(b, loc, colOp, flatIndex);

  if (auto regPOp = dyn_cast<RegPOp>(defOp))
    return applyInverseRegP(b, loc, regPOp, flatIndex);

  if (auto tileByOp = dyn_cast<TileByOp>(defOp))
    return applyInverseTileBy(b, loc, tileByOp, flatIndex);

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
    patterns.add<ApplyOpLowering, ApplyInverseOpLowering>(context);

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
