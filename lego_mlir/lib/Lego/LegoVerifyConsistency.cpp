#define GEN_PASS_DEF_LEGOVERIFYCONSISTENCYPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

// Simple symbolic evaluator to check if two regions define an inverse mapping.
// For now, focuses on linear combinations and rank-1 identities.
struct LegoVerifyConsistencyPassImpl
    : public mlir::lego::impl::LegoVerifyConsistencyPassBase<LegoVerifyConsistencyPassImpl> {
  using mlir::lego::impl::LegoVerifyConsistencyPassBase<
      LegoVerifyConsistencyPassImpl>::LegoVerifyConsistencyPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    module.walk([&](GenPOp op) {
      if (op.getInvBody().empty()) return;

      auto dimsVals = op.getDims();
      int rank = dimsVals.size();
      
      Block &applyBlock = op.getBody().front();
      Block &invBlock = op.getInvBody().front();

      // 1. Basic rank check
      if (applyBlock.getNumArguments() != (unsigned)rank) {
          op.emitError("apply region argument count mismatch: expected ") << rank << ", got " << applyBlock.getNumArguments();
          return;
      }
      if (invBlock.getTerminator()->getNumOperands() != (unsigned)rank) {
          op.emitError("inv region return count mismatch: expected ") << rank << ", got " << invBlock.getTerminator()->getNumOperands();
          return;
      }

      // 2. Linear consistency check (heuristic)
      // If we can prove inv(apply(args)) != args, emit warning/error.
      
      // We can try to evaluate for a few points if they are constant-folded.
      // But more simply, if apply is "yield arg0 + arg1" and inv is "yield 0, 0",
      // it's obviously wrong.
      
      if (invBlock.getNumArguments() == 1) {
          auto term = invBlock.getTerminator();
          bool allConstZero = true;
          for (Value operand : term->getOperands()) {
              APInt val;
              if (!matchPattern(operand, m_ConstantInt(&val)) || val.getSExtValue() != 0) {
                  allConstZero = false;
                  break;
              }
          }
          
          if (allConstZero && rank > 0) {
              // Check if apply is non-zero
              auto applyTerm = applyBlock.getTerminator();
              APInt val;
              if (!matchPattern(applyTerm->getOperand(0), m_ConstantInt(&val)) || val.getSExtValue() != 0) {
                   op.emitWarning("inv region yields constant zero, but apply region is non-constant or non-zero. Potential inconsistency.");
              }
          }
      }
    });
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyConsistencyPass() {
  return std::make_unique<LegoVerifyConsistencyPassImpl>();
}
} // namespace lego
} // namespace mlir
