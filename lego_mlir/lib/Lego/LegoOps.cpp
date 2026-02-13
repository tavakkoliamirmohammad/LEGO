#include "Lego/LegoOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

using namespace mlir;
using namespace mlir::lego;
using namespace mlir::transform;

#define GET_OP_CLASSES
#include "Lego/LegoOps.cpp.inc"

DiagnosedSilenceableFailure ApplyLayoutOp::apply(
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

void ApplyLayoutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  transform::consumesHandle(getOperation()->getOpOperands().take_front(1), effects);
  transform::onlyReadsHandle(getOperation()->getOpOperands().drop_front(1), effects);
  transform::producesHandle(getOperation()->getResults(), effects);
  transform::modifiesPayload(effects);
}
