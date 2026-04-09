#ifndef LEGO_LAYOUT_LOWERING_H
#define LEGO_LAYOUT_LOWERING_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace lego {

/// Flatten a multi-dimensional index into a 1D flat index using row-major
/// order.  Emits arith ops: flat = sum_k(idx_k * prod(dims[k+1..])).
Value flattenIndex(OpBuilder &b, Location loc, ValueRange indices,
                   ValueRange dims);

/// Decompose a flat index into multi-dimensional indices given dimension
/// sizes.  Emits arith div/rem ops.
SmallVector<Value> unflattenIndex(OpBuilder &b, Location loc, Value flatIndex,
                                  ValueRange dims);

/// Apply a LEGO layout to a set of indices, producing a flat index.
/// Dispatches to layout-specific lowering for RegP, GenP, OrderBy,
/// GroupBy, and TileBy.  Returns nullptr on unsupported layout.
Value applyLayout(OpBuilder &b, Location loc, Value layout,
                  ValueRange indices);

/// Apply the inverse of a LEGO layout to a flat index, producing
/// multi-dimensional indices.  Returns empty vector on failure.
SmallVector<Value> applyInverseLayout(OpBuilder &b, Location loc,
                                      Value layout, Value flatIndex);

} // namespace lego
} // namespace mlir

#endif // LEGO_LAYOUT_LOWERING_H
