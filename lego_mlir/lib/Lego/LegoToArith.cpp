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

// Helper to get or create a constant index
Value getConstantIndex(OpBuilder &b, Location loc, int64_t val) {
  return b.create<arith::ConstantIndexOp>(loc, val);
}

// Forward declarations
Value applyLayout(OpBuilder &b, Location loc, Value layout, ValueRange indices);
SmallVector<Value> applyInverseLayout(OpBuilder &b, Location loc, Value layout, Value flatIndex);

// Flatten indices: index = i0 * (d1*...*dn) + i1 * (d2*...*dn) + ... + in
Value flattenIndex(OpBuilder &b, Location loc, ValueRange indices, ArrayRef<int64_t> dims) {
  Value flat = getConstantIndex(b, loc, 0);
  int rank = indices.size();
  
  for (int k = 0; k < rank; ++k) {
    int64_t stride = 1;
    for (int j = k + 1; j < rank; ++j) {
      stride *= dims[j];
    }
    Value strideVal = getConstantIndex(b, loc, stride);
    Value term = b.create<arith::MulIOp>(loc, indices[k], strideVal);
    flat = b.create<arith::AddIOp>(loc, flat, term);
  }
  return flat;
}

// Unflatten index: inverse of flatten
SmallVector<Value> unflattenIndex(OpBuilder &b, Location loc, Value flatIndex, ArrayRef<int64_t> dims) {
  SmallVector<Value> indices;
  int rank = dims.size();
  Value currentQuery = flatIndex;

  for (int k = 0; k < rank; ++k) {
    int64_t stride = 1;
    for (int j = k + 1; j < rank; ++j) {
      stride *= dims[j];
    }
    Value strideVal = getConstantIndex(b, loc, stride);
    
    // index[k] = currentQuery / stride
    Value idx = b.create<arith::DivUIOp>(loc, currentQuery, strideVal);
    indices.push_back(idx);
    
    // currentQuery = currentQuery % stride
    currentQuery = b.create<arith::RemUIOp>(loc, currentQuery, strideVal);
  }
  return indices;
}


// --- Logic for RegPOp ---

Value applyRegP(OpBuilder &b, Location loc, RegPOp op, ValueRange indices) {
  ArrayAttr permAttr = op.getPerm();
  ArrayAttr dimsAttr = op.getDims();
  
  SmallVector<int64_t> dims;
  for (auto attr : dimsAttr) {
    dims.push_back(cast<IntegerAttr>(attr).getInt());
  }

  SmallVector<int64_t> perm;
  for (auto attr : permAttr) {
    perm.push_back(cast<IntegerAttr>(attr).getInt());
  }

  // Permute indices
  SmallVector<Value> permutedIndices;
  permutedIndices.resize(indices.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    permutedIndices[i] = indices[perm[i]];
  }
  
  // Permute dimensions
  SmallVector<int64_t> permutedDims;
  permutedDims.resize(dims.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    permutedDims[i] = dims[perm[i]];
  }

  return flattenIndex(b, loc, permutedIndices, permutedDims);
}

SmallVector<Value> applyInverseRegP(OpBuilder &b, Location loc, RegPOp op, Value flatIndex) {
  ArrayAttr permAttr = op.getPerm();
  ArrayAttr dimsAttr = op.getDims();
  
  SmallVector<int64_t> dims;
  for (auto attr : dimsAttr) {
    dims.push_back(cast<IntegerAttr>(attr).getInt());
  }

  SmallVector<int64_t> perm;
  for (auto attr : permAttr) {
    perm.push_back(cast<IntegerAttr>(attr).getInt());
  }
  
  // Permute dimensions
  SmallVector<int64_t> permutedDims;
  permutedDims.resize(dims.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    permutedDims[i] = dims[perm[i]];
  }

  // Unflatten using permuted dims
  SmallVector<Value> permutedIndices = unflattenIndex(b, loc, flatIndex, permutedDims);
  
  // Inverse permute indices
  SmallVector<Value> originalIndices;
  originalIndices.resize(permutedIndices.size());
  
  // if y = perm(x), then y[i] = x[perm[i]]
  // so x[perm[i]] = y[i]
  for (size_t i = 0; i < perm.size(); ++i) {
    originalIndices[perm[i]] = permutedIndices[i];
  }
  
  return originalIndices;
}

// --- Logic for GenPOp ---

Value applyGenP(OpBuilder &b, Location loc, GenPOp op, ValueRange indices) {
  // GenP has a region that takes indices and returns transformed indices (or flat index?)
  // The design says "yields the transformed indices (or flat index)".
  // Let's assume it yields transformed indices which we then flatten, OR it yields the flat index directly.
  // Given ApplyOp takes a layout and expects a flat index, GenP usually acts as a layout block.
  // If GenP represents a full layout, it might compute flat index.
  // But usually GenPOp maps multi-dim to flat.
  
  // For now, let's assume the region yields a SINGLE value which is the FLAT index.
  // Or it yields multi-dim indices which we assume are for the standard layout?
  // User logic: "The region ... yields the transformed indices".
  
  // We inline the region.
  Block &block = op.getBody().front();
  // Check if arg count matches
  if (block.getNumArguments() != indices.size()) {
      // Mismatch
      return nullptr;
  }
  
  // Map arguments
  IRMapping mapping;
  for (unsigned i = 0; i < indices.size(); ++i) {
      mapping.map(block.getArgument(i), indices[i]);
  }
  
  // Clone ops
  for (auto &opInst : block.without_terminator()) {
      b.clone(opInst, mapping);
  }
  
  // Get the yielded value
  Operation *term = block.getTerminator(); // assumed to be yield?
  // We don't have a specific yield op defined, let's assume standard scf.yield or similar
  // OR the region just ends.
  // The ODS didn't specify a terminator.
  // Let's assume the last op is the result or we look for a yield.
  
  if (term->getNumOperands() == 1) {
      return mapping.lookup(term->getOperand(0));
  }
  
  return nullptr; // Unsupported terminator
}


// --- Helper to get input dims for a layout ---

SmallVector<int64_t> getLayoutInputDims(Value layout) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp) return {};

  if (auto regPOp = dyn_cast<RegPOp>(defOp)) {
    SmallVector<int64_t> dims;
    for (auto attr : regPOp.getDims()) {
      dims.push_back(cast<IntegerAttr>(attr).getInt());
    }
    return dims;
  }
  
  if (auto rowOp = dyn_cast<RowOp>(defOp)) {
    return {rowOp.getN(), rowOp.getM()};
  }
  
  if (auto genPOp = dyn_cast<GenPOp>(defOp)) {
    SmallVector<int64_t> dims;
    for (auto attr : genPOp.getDims()) {
      dims.push_back(cast<IntegerAttr>(attr).getInt());
    }
    return dims;
  }
  
  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
    SmallVector<int64_t> allDims;
    for (Value perm : orderByOp.getPerms()) {
      auto pDims = getLayoutInputDims(perm);
      allDims.append(pDims.begin(), pDims.end());
    }
    return allDims;
  }
  
  // TileByOp? 
  // If TileBy wraps input, its input dimensions?
  // TileBy consumes indices and passes them to input.
  // So TileBy input dims match input layout input dims?
  // No, TileBy input dims are "untiled".
  // Input layout input dims are "tiled".
  // Dimensions count doubles?
  // If TileByOp is present, we can try to infer.
  // But strictly, TileBy consumes N indices.
  // For now, let's assume we don't nest TileBy inside OrderBy without being explicit.
  
  return {};
}

// --- Logic for TileByOp ---

Value applyTileBy(OpBuilder &b, Location loc, TileByOp op, ValueRange indices) {
  ArrayAttr sizesAttr = op.getTileSizes();
  SmallVector<int64_t> sizes;
  for (auto attr : sizesAttr) {
    sizes.push_back(cast<IntegerAttr>(attr).getInt());
  }

  // Check rank match
  if (indices.size() != sizes.size()) {
    // Basic check: should match? 
    // Or tile applies to first N dims? 
    // Python says: for each dim d_i, splits into d_i/s_i, d_i%s_i.
    // So input indices count == sizes count.
    // Result indices count == 2 * input indices count.
  }

  SmallVector<Value> tiledIndices;
  for (size_t i = 0; i < indices.size(); ++i) {
    Value idx = indices[i];
    int64_t s = sizes[i];
    Value sizeVal = getConstantIndex(b, loc, s);
    
    // q = idx / s
    Value q = b.create<arith::DivUIOp>(loc, idx, sizeVal);
    // r = idx % s
    Value r = b.create<arith::RemUIOp>(loc, idx, sizeVal);
    
    tiledIndices.push_back(q);
    tiledIndices.push_back(r);
  }
  
  // Recursively apply the inner layout
  return applyLayout(b, loc, op.getInput(), tiledIndices);
}

SmallVector<Value> applyInverseTileBy(OpBuilder &b, Location loc, TileByOp op, Value flatIndex) {
  // Get inner layout indices
  SmallVector<Value> innerIndices = applyInverseLayout(b, loc, op.getInput(), flatIndex);
  if (innerIndices.empty()) return {};

  ArrayAttr sizesAttr = op.getTileSizes();
  SmallVector<int64_t> sizes;
  for (auto attr : sizesAttr) {
    sizes.push_back(cast<IntegerAttr>(attr).getInt());
  }
  
  // innerIndices are [q0, r0, q1, r1, ...]
  // We want [q0 * s0 + r0, q1 * s1 + r1, ...]
  
  SmallVector<Value> originalIndices;
  for (size_t i = 0; i < sizes.size(); ++i) {
    Value q = innerIndices[2*i];
    Value r = innerIndices[2*i + 1];
    int64_t s = sizes[i];
    Value sizeVal = getConstantIndex(b, loc, s);
    
    // idx = q * s + r
    Value qs = b.create<arith::MulIOp>(loc, q, sizeVal);
    Value idx = b.create<arith::AddIOp>(loc, qs, r);
    originalIndices.push_back(idx);
  }
  
  return originalIndices;
}

// --- Logic for OrderByOp ---

Value applyOrderBy(OpBuilder &b, Location loc, OrderByOp op, ValueRange indices) {
  // OrderBy(perm1, perm2, ...)
  // It applies a sequence of blocks.
  // Each block consumes a subset of indices.
  // Accumulates into a single flat index.
  // formula: flat = flat * size_k + next_flat
  
  Value flatIndex = getConstantIndex(b, loc, 0);
  int offset = 0;
  
  for (Value perm : op.getPerms()) {
    // Get dims for this perm to know how many indices to consume
    SmallVector<int64_t> pDims = getLayoutInputDims(perm);
    if (pDims.empty()) {
      // If we can't determine dims, we can't slice types.
      // But maybe we can pass the remaining indices?
      // No, we need strict slicing for correctness.
      // Error out or assume consumption of remainder?
      return nullptr;
    }
    
    int count = pDims.size();
    if (offset + count > indices.size()) return nullptr;
    
    ValueRange slice = indices.slice(offset, count);
    offset += count;
    
    // Apply nested layout
    Value innerFlat = applyLayout(b, loc, perm, slice);
    if (!innerFlat) return nullptr;
    
    // Calculate total size of this perm block
    // size = product(pDims)
    // Wait, is it product(pDims)? 
    // Yes, if it's a dense packing.
    // OrderBy assumes dense packing of blocks.
    int64_t size = 1;
    for (auto d : pDims) size *= d;
    Value sizeVal = getConstantIndex(b, loc, size);
    
    // Update flat index
    // flat = flat * size + inner
    Value flatMul = b.create<arith::MulIOp>(loc, flatIndex, sizeVal);
    flatIndex = b.create<arith::AddIOp>(loc, flatMul, innerFlat);
  }
  
  return flatIndex;
}

SmallVector<Value> applyInverseOrderBy(OpBuilder &b, Location loc, OrderByOp op, Value flatIndex) {
  // OrderBy inverse
  // Iterate backwards.
  // current_block_flat = flat % size
  // flat = flat / size
  // current_indices = block.inv(current_block_flat)
  // prepend indices
  
  SmallVector<Value> allIndices;
  
  // We need to iterate backwards.
  // getPerms returns a range.
  auto perms = op.getPerms();
  
  // OperandRange doesn't support rbegin directly, copy to vector
  SmallVector<Value> permsVec(perms.begin(), perms.end());
  
  for (auto it = permsVec.rbegin(); it != permsVec.rend(); ++it) {
    Value perm = *it;
    SmallVector<int64_t> pDims = getLayoutInputDims(perm);
    int64_t size = 1;
    for (auto d : pDims) size *= d;
    Value sizeVal = getConstantIndex(b, loc, size);
    
    Value innerFlat = b.create<arith::RemUIOp>(loc, flatIndex, sizeVal);
    flatIndex = b.create<arith::DivUIOp>(loc, flatIndex, sizeVal);
    
    SmallVector<Value> innerIndices = applyInverseLayout(b, loc, perm, innerFlat);
    if (innerIndices.empty()) return {};
    
    // Prepend
    allIndices.insert(allIndices.begin(), innerIndices.begin(), innerIndices.end());
  }
  
  return allIndices;
}

// --- Dispatcher ---

Value applyLayout(OpBuilder &b, Location loc, Value layout, ValueRange indices) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp) return nullptr;

  if (auto rowOp = dyn_cast<RowOp>(defOp)) {
    // RowOp(n, m) -> i*m + j
    int64_t m_val = rowOp.getM();
    Value m = getConstantIndex(b, loc, m_val);
    Value i = indices[0];
    Value j = indices[1];
    Value i_m = b.create<arith::MulIOp>(loc, i, m);
    return b.create<arith::AddIOp>(loc, i_m, j);
  }
  
  if (auto regPOp = dyn_cast<RegPOp>(defOp)) {
    return applyRegP(b, loc, regPOp, indices);
  }
  
  if (auto genPOp = dyn_cast<GenPOp>(defOp)) {
    return applyGenP(b, loc, genPOp, indices);
  }
  
  if (auto tileByOp = dyn_cast<TileByOp>(defOp)) {
    return applyTileBy(b, loc, tileByOp, indices);
  }
  
  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
     return applyOrderBy(b, loc, orderByOp, indices);
  }
  
  return nullptr;
}

SmallVector<Value> applyInverseLayout(OpBuilder &b, Location loc, Value layout, Value flatIndex) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp) return {};

  if (auto rowOp = dyn_cast<RowOp>(defOp)) {
    int64_t m_val = rowOp.getM();
    Value m = getConstantIndex(b, loc, m_val);
    Value i = b.create<arith::DivUIOp>(loc, flatIndex, m);
    Value j = b.create<arith::RemUIOp>(loc, flatIndex, m);
    return {i, j};
  }
  
  if (auto regPOp = dyn_cast<RegPOp>(defOp)) {
    return applyInverseRegP(b, loc, regPOp, flatIndex);
  }

  if (auto tileByOp = dyn_cast<TileByOp>(defOp)) {
    return applyInverseTileBy(b, loc, tileByOp, flatIndex);
  }
  
  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
    return applyInverseOrderBy(b, loc, orderByOp, flatIndex);
  }
  
  return {};
}

struct ApplyOpLowering : public OpRewritePattern<ApplyOp> {
  using OpRewritePattern<ApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyOp op, PatternRewriter &rewriter) const override {
    Value res = applyLayout(rewriter, op.getLoc(), op.getLayout(), op.getIndices());
    if (!res) return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ApplyInverseOpLowering : public OpRewritePattern<ApplyInverseOp> {
  using OpRewritePattern<ApplyInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ApplyInverseOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> res = applyInverseLayout(rewriter, op.getLoc(), op.getLayout(), op.getFlatIndex());
    if (res.empty()) return failure();
    rewriter.replaceOp(op, res);
    return success();
  }
};

// End anonymous namespace for helper functions
  
struct LegoToArithPassImpl : public mlir::lego::impl::LegoToArithPassBase<LegoToArithPassImpl> {
  using mlir::lego::impl::LegoToArithPassBase<LegoToArithPassImpl>::LegoToArithPassBase;
  
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
