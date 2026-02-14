#include "Lego/LegoOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

using namespace mlir;
using namespace mlir::lego;
using namespace mlir::transform;

#define GET_OP_CLASSES
#include "Lego/LegoOps.cpp.inc"
#include "Lego/LegoUtils.h"
#include <numeric>

// ============================================================================
// RegPOp Verification
// ============================================================================

LogicalResult RegPOp::verify() {
  auto perm = extractI64Array(getPerm());
  auto dims = extractI64Array(getDims());

  if (perm.size() != dims.size()) {
    return emitOpError("Permutation rank " + std::to_string(perm.size()) +
                       " does not match dimensions rank " +
                       std::to_string(dims.size()));
  }

  for (int64_t d : dims) {
      if (d <= 0) return emitOpError("Dimension " + std::to_string(d) + " must be strictly positive");
  }

  // Verify perm is a valid permutation of 0..size-1
  SmallVector<int64_t> sortedPerm = perm;
  std::sort(sortedPerm.begin(), sortedPerm.end());
  for (size_t i = 0; i < sortedPerm.size(); ++i) {
    if (sortedPerm[i] != (int64_t)i) {
      return emitOpError("Invalid permutation: not a permutation of 0.." +
                         std::to_string(sortedPerm.size() - 1));
    }
  }

  return success();
}

// ============================================================================
// TileByOp Verification
// ============================================================================

LogicalResult TileByOp::verify() {
  auto info = extractNestedTileDims(getTileDims());
  if (!info.valid) {
    return emitOpError("Invalid tile dimensions structure. Expected nested list [[...], ...]");
  }

  for (auto d : info.flatDims) {
      if (d <= 0) return emitOpError("Tile dimension " + std::to_string(d) + " must be strictly positive");
  }

  int64_t d = info.d;
  int64_t q = info.q; // Unused for check, but part of structure

  // Get input (d, q) from OrderBy or other layout
  auto [inputD, inputQ] = getLayoutDQ(getInput());
  
  if (inputD != 0 || inputQ != 0) {
      if (d != inputD) {
          return emitOpError("Inner tile dimension " + std::to_string(d) + 
                             " does not match input layout dimension " + std::to_string(inputD));
      }

      // Verify global product of dimensions (volume preservation)
      int64_t tileProduct = 1;
      for (auto attr : getTileDims()) {
          auto tileGroup = extractI64Array(cast<ArrayAttr>(attr));
          for (auto x : tileGroup) tileProduct *= x;
      }

      int64_t inputProduct = 1;
      auto inputDims = getLayoutInputDims(getInput());
      for (auto x : inputDims) inputProduct *= x;

      if (tileProduct != inputProduct) {
           return emitOpError("Total product of tile dims (" + std::to_string(tileProduct) + 
                              ") does not match total product of input dims (" + 
                              std::to_string(inputProduct) + ")");
      }
  }

  return success();
}

// ============================================================================
// RowOp Verification
// ============================================================================

LogicalResult RowOp::verify() {
    auto dims = extractI64Array(getDims());
    for (int64_t d : dims) {
        if (d <= 0) return emitOpError("Dimension " + std::to_string(d) + " must be strictly positive");
    }
    return success();
}

// ============================================================================
// ColOp Verification
// ============================================================================

LogicalResult ColOp::verify() {
    auto dims = extractI64Array(getDims());
    for (int64_t d : dims) {
        if (d <= 0) return emitOpError("Dimension " + std::to_string(d) + " must be strictly positive");
    }
    return success();
}

DiagnosedSilenceableFailure ApplyLayoutTransformOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results,
    transform::TransformState &state) {
  
  // Get the target payload operations
  auto targets = state.getPayloadOps(getTarget());
  
  // TODO: Retrieve the layout object from getLayout()
  // Since layout is an SSA value, we need to inspect what defined it.
  
  // For now, we just pass the targets through to the result.
  results.set(getOperation()->getResult(0), targets);
  
  return DiagnosedSilenceableFailure::success();
}

void ApplyLayoutTransformOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  transform::consumesHandle(getOperation()->getOpOperands().take_front(1), effects);
  transform::onlyReadsHandle(getOperation()->getOpOperands().drop_front(1), effects);
  transform::producesHandle(getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}
