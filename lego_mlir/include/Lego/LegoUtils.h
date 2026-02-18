#ifndef LEGO_UTILS_H
#define LEGO_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "Lego/LegoOps.h"

using namespace mlir;
using namespace mlir::lego;

namespace mlir {
namespace lego {

// Extract I64ArrayAttr into SmallVector<int64_t>
inline SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  if (!attr) return result;
  for (auto elt : attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(elt)) {
      result.push_back(intAttr.getInt());
    }
  }
  return result;
}

// sigma(values, perm) => new_arr[i] = values[perm[i]]
template <typename T>
SmallVector<T> sigma(ArrayRef<T> values, ArrayRef<int64_t> perm) {
  SmallVector<T> result;
  result.reserve(perm.size());
  for (int64_t idx : perm) {
    if (idx >= 0 && idx < (int64_t)values.size()) {
      result.push_back(values[idx]);
    }
  }
  return result;
}

inline SmallVector<Value> sigmaValues(ValueRange values, ArrayRef<int64_t> perm) {
  SmallVector<Value> result;
  result.reserve(perm.size());
  for (int64_t idx : perm) {
    if (idx >= 0 && idx < (int64_t)values.size()) {
      result.push_back(values[idx]);
    }
  }
  return result;
}

inline SmallVector<int64_t> inversePermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inverse(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inverse[perm[i]] = i;
  }
  return inverse;
}

// get_sigma_perm(d, q) matches Python:
// [[k + d*h for h in range(q)] for k in range(d)] flattened
// Result: 0, d, 2d, ..., 1, d+1, 2d+1, ...
inline SmallVector<int64_t> getSigmaPerm(int d, int q) {
  SmallVector<int64_t> perm;
  perm.reserve(d * q);
  for (int k = 0; k < d; ++k) {
    for (int h = 0; h < q; ++h) {
      perm.push_back(k + d * h);
    }
  }
  return perm;
}

struct TileDimsInfo {
  SmallVector<int64_t> flatDims;
  int64_t d = 0;
  int64_t q = 0;
  bool valid = false;
};

inline TileDimsInfo extractNestedTileDims(ArrayAttr tileDimsAttr) {
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

    auto dims = extractI64Array(innerArray);
    info.flatDims.append(dims.begin(), dims.end());
  }
  info.valid = true;
  return info;
}

inline SmallVector<int64_t> getLayoutInputDims(Value layout) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp) return {};

  if (auto rowOp = dyn_cast<RowOp>(defOp))
    return extractI64Array(rowOp.getDims());

  if (auto colOp = dyn_cast<ColOp>(defOp))
    return extractI64Array(colOp.getDims());

  if (auto regPOp = dyn_cast<RegPOp>(defOp))
    return extractI64Array(regPOp.getDims());

  if (auto genPOp = dyn_cast<GenPOp>(defOp))
    return extractI64Array(genPOp.getDims());

  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
    // Concatanation of dims of the perms
    SmallVector<int64_t> dims;
    for (Value v : orderByOp.getPerms()) {
      auto subDims = getLayoutInputDims(v);
      dims.append(subDims.begin(), subDims.end());
    }
    return dims;
  }

  if (auto groupByOp = dyn_cast<GroupByOp>(defOp)) {
    return extractI64Array(groupByOp.getGroupDims());
  }

  if (auto tileByOp = dyn_cast<TileByOp>(defOp)) {
    return getLayoutInputDims(tileByOp.getInput());
  }

  return {};
}

// Helper to get {d, q} for a layout op.
// Most primitive ops (Row, Col, RegP, GenP) have q=1, d=rank.
inline std::pair<int, int> getLayoutDQ(Value layout) {
  Operation *defOp = layout.getDefiningOp();
  if (!defOp) return {0, 0};

  if (auto rowOp = dyn_cast<RowOp>(defOp)) {
    int d = rowOp.getDims().size();
    return {d, 1};
  }
  if (auto colOp = dyn_cast<ColOp>(defOp)) {
    int d = colOp.getDims().size();
    return {d, 1};
  }
  if (auto regPOp = dyn_cast<RegPOp>(defOp)) {
    int d = regPOp.getDims().size();
    return {d, 1};
  }
  if (auto genPOp = dyn_cast<GenPOp>(defOp)) {
    int d = genPOp.getDims().size();
    return {d, 1};
  }
  if (auto orderByOp = dyn_cast<OrderByOp>(defOp)) {
      ValueRange perms = orderByOp.getPerms();
      int q = perms.size();
      if (q == 0) return {0, 0};
      int d = getLayoutInputDims(perms[0]).size();
      return {d, q};
  }
  return {0, 0};
}

} // namespace lego
} // namespace mlir

#endif // LEGO_UTILS_H
