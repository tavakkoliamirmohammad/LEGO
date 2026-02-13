#define GEN_PASS_DEF_LEGOTOARITHPASS
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

// get_sigma_perm(d, q) matches Python:
// [[k + d*h for h in range(q)] for k in range(d)] flattened
// Result: 0, d, 2d, ..., 1, d+1, 2d+1, ...
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

struct TileDimsInfo {
  SmallVector<int64_t> flatDims;
  int64_t d = 0;
  int64_t q = 0;
  bool valid = false;
};

TileDimsInfo extractNestedTileDims(ArrayAttr tileDimsAttr) {
  TileDimsInfo info;
  if (!tileDimsAttr) return info;
  info.q = tileDimsAttr.size();
  if (info.q == 0) return info; // Empty outer list

  for (Attribute innerAttr : tileDimsAttr) {
    auto innerArray = dyn_cast<ArrayAttr>(innerAttr);
    if (!innerArray) return {}; // Invalid structure: expected nested list

    if (info.d == 0) {
      info.d = innerArray.size();
      if (info.d == 0) return {}; // Empty inner list not allowed
    } else if ((int64_t)innerArray.size() != info.d) {
      return {}; // Dimensional mismatch: all groups must have size d
    }

    for (Attribute valAttr : innerArray) {
      if (auto intAttr = dyn_cast<IntegerAttr>(valAttr)) {
        info.flatDims.push_back(intAttr.getInt());
      } else {
        return {}; // Invalid element: expected integer
      }
    }
  }
  info.valid = true;
  return info;
}

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
    // TileBy input dims = the flattened tile_dims
    auto info = extractNestedTileDims(tileByOp.getTileDims());
    return info.flatDims;
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
// apply: inlines the body (apply) region
// inv:   inlines the inv_body (inverse) region
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

// ============================================================================
// TileBy (Python OrderBy.TileBy)
//
// Expands to GroupBy([tile_dims], objects...) where objects are:
//   for each perm in input_orderby:
//     perm
//     RegP(perm_dims, inverse(sigma(d_perm, q_perm)))
//   RegP(tile_dims, sigma(d_tile, q_tile))
//
// apply: flatten indices -> apply GroupBy logic (reverse objects)
// inv:   apply GroupBy inv logic (forward objects) -> unflatten
// ============================================================================

// Helper to get {d, q} for a layout op.
// Most primitive ops (Row, Col, RegP, GenP) have q=1, d=rank.
std::pair<int, int> getLayoutDQ(Value layout) {
  // For now, assume all blocks in the OrderBy chain are simple blocks with q=1
  SmallVector<int64_t> dims = getLayoutInputDims(layout);
  return {static_cast<int>(dims.size()), 1};
}



// Helper for TileBy Apply
Value applyTileBy(OpBuilder &b, Location loc, TileByOp op,
                  ValueRange indices) {
  auto info = extractNestedTileDims(op.getTileDims());
  if (!info.valid) return {};

  auto tileDims = info.flatDims;
  int64_t d_tile = info.d;
  int64_t q_tile = info.q;

  // 1. Flatten input indices
  Value currentFlat = flattenIndex(b, loc, indices, tileDims);

  // 2. Identify the chain of objects from input OrderBy
  Operation *defOp = op.getInput().getDefiningOp();
  SmallVector<Value> chain;
  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
    auto range = orderByOp.getPerms(); // Variadic<Lego_LayoutType>
    chain.append(range.begin(), range.end());
  } else {
    chain.push_back(op.getInput());
  }

  // 3. Construct virtual GroupBy objects list:
  //    [o1, reshuffle1, o2, reshuffle2, ..., final_reshuffle]
  //    But apply() iterates in REVERSE.

  // 3a. Apply final_reshuffle (RegP with sigma_dq)
  {
    auto sigma = getSigmaPerm(d_tile, q_tile); // [0, d, 2d...]
    // unflatten currentFlat using tileDims
    auto indices = unflattenIndex(b, loc, currentFlat, tileDims);
    // permute indices
    auto permutedIndices = sigmaValues(indices, sigma);
    // permute dims
    auto permutedDims = ::sigma<int64_t>(tileDims, sigma);
    // flatten
    currentFlat = flattenIndex(b, loc, permutedIndices, permutedDims);
  }

  // 3b. Iterate chain in reverse
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    Value obj = *it;
    auto [d_obj, q_obj] = getLayoutDQ(obj);
    auto objDims = getLayoutInputDims(obj); // dims of this block

    // Apply reshuffle_obj (RegP with inverse(sigma(d_obj, q_obj)))
    auto sigma_o = getSigmaPerm(d_obj, q_obj);
    auto sigma_o_inv = inversePermutation(sigma_o);
    auto reshuffleInputDims = ::sigma<int64_t>(objDims, sigma_o);

    // Apply Reshuffle
    {
       auto indices = unflattenIndex(b, loc, currentFlat, reshuffleInputDims);
       auto permIndices = sigmaValues(indices, sigma_o_inv); // matches RegP apply
       currentFlat = flattenIndex(b, loc, permIndices, objDims);
    }

    // Apply object `o`
    {
      auto indices = unflattenIndex(b, loc, currentFlat, objDims);
      currentFlat = applyLayout(b, loc, obj, indices);
    }
  }

  return currentFlat;
}

SmallVector<Value> applyInverseTileBy(OpBuilder &b, Location loc, TileByOp op,
                                      Value flatIndex) {
  auto info = extractNestedTileDims(op.getTileDims());
  if (!info.valid) return {};

  auto tileDims = info.flatDims;
  int64_t d_tile = info.d;
  int64_t q_tile = info.q;

  Value currentFlat = flatIndex;

  // 1. Identify chain
  Operation *defOp = op.getInput().getDefiningOp();
  SmallVector<Value> chain;
  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
    auto range = orderByOp.getPerms();
    chain.append(range.begin(), range.end());
  } else {
    chain.push_back(op.getInput());
  }

  // 2. Iterate chain FORWARD (GroupBy.inv logic)
  // List: [o1, reshuffle1, o2, reshuffle2, ..., final_reshuffle]
  for (Value obj : chain) {
    auto [d_obj, q_obj] = getLayoutDQ(obj);
    auto objDims = getLayoutInputDims(obj);

    // Apply inv(o)
    {
      // GroupBy.inv(flat): obj.inv(flat) -> flatten into obj.dims
      auto indices = applyInverseLayout(b, loc, obj, currentFlat); 
      // If o is GenP/etc, indices are output of inv().
      // We must flatten them using objDims.
      currentFlat = flattenIndex(b, loc, indices, objDims);
    }

    // Apply inv(reshuffle)
    // Reshuffle: RegP(sigma(o_dims, sigma_o), sigma_o_inv)
    // inverse is RegP(o_dims, sigma_o)
    {
      auto sigma_o = getSigmaPerm(d_obj, q_obj);
      // RegP.inv(flat) unflat(flat, perm(dims)) -> perm_inv(idx)
      // Here "perm" of the reshuffle op is sigma_o_inv.
      // So "perm(dims)" is sigma(inputDims, sigma_o_inv) = o_dims (since input was permuted).
      // Wait, let's just use the conceptual transform.
      // Reshuffle was: input -> permute by sigma_o_inv. output -> permute dims back.
      // Inverse is: input -> permute by sigma_o. output -> permute dims.
      
      // We start with currentFlat which corresponds to `o_dims` (from previous step).
      auto indices = unflattenIndex(b, loc, currentFlat, objDims);
      // Permute by sigma_o
      auto permIndices = sigmaValues(indices, sigma_o);
      // Output dims: sigma(o_dims, sigma_o)
      auto outDims = ::sigma<int64_t>(objDims, sigma_o);
      currentFlat = flattenIndex(b, loc, permIndices, outDims);
    }
  }

  // 3. Apply inv(final_reshuffle)
  // Final reshuffle: RegP(tileDims, sigma_dq)
  // Inverse: RegP(permutedDims, inverse(sigma_dq))
  // Logically: unflatten using permutedDims -> permute by inv(sigma_dq) -> flatten using tileDims
  {
    auto sigma = getSigmaPerm(d_tile, q_tile);
    auto sigma_inv = inversePermutation(sigma);
    auto permutedDims = ::sigma<int64_t>(tileDims, sigma);

    auto indices = unflattenIndex(b, loc, currentFlat, permutedDims);
    auto permIndices = sigmaValues(indices, sigma_inv);
    currentFlat = flattenIndex(b, loc, permIndices, tileDims);
  }

  // 4. Finally unflatten to tileDims (GroupBy.inv returns N-D indices)
  return unflattenIndex(b, loc, currentFlat, tileDims);
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

  if (auto genPOp = dyn_cast<GenPOp>(defOp))
    return applyInverseGenP(b, loc, genPOp, flatIndex);

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
