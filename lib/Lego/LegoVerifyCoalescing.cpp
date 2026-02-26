#define GEN_PASS_DEF_LEGOVERIFYCOALESCINGPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "Lego/SMTUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/AsmState.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

struct LegoVerifyCoalescingPassImpl
    : public mlir::lego::impl::LegoVerifyCoalescingPassBase<
          LegoVerifyCoalescingPassImpl> {
  using mlir::lego::impl::LegoVerifyCoalescingPassBase<
      LegoVerifyCoalescingPassImpl>::LegoVerifyCoalescingPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    getContext().getOrLoadDialect<smt::SMTDialect>();
    ModuleOp module = getOperation();

    // Find all lego.apply operations
    SmallVector<ApplyOp> applyOps;
    module.walk([&](ApplyOp apply) { applyOps.push_back(apply); });

    if (applyOps.empty()) return;

    AsmState state(module);
    unsigned nextId = 0;

    for (auto apply : applyOps) {
      // Check if this apply is used in a memory access pattern
      // For simplicity, we verify coalescing for all applies
      // In practice, you'd annotate which applies correspond to global memory
      if (failed(verifyCoalescing(apply, state, nextId))) {
        // Don't signal pass failure - just emit warnings
        // This is because coalescing is a performance property, not correctness
      }
    }
  }

private:
  // Verify that a warp of WARP_SIZE threads produces coalesced accesses
  LogicalResult verifyCoalescing(ApplyOp apply, AsmState &state, unsigned &nextId) {
    SMTSolverContext smtCtx(apply.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;
    SMTBuilder &builder = *smtCtx.builder;

    // Model: For a 1D thread layout, threadIdx maps to layout indices
    // We'll verify the simple case: threadIdx → (threadIdx, 0) for 2D layouts
    // Or threadIdx → threadIdx for 1D layouts

    size_t numIndices = apply.getIndices().size();

    // Warp size
    int WARP_SIZE = warpSize.getValue();

    // Create symbolic variables for the base thread ID
    std::string baseThreadVarName = "base_thread";
    Value baseThread = smt::DeclareFunOp::create(
        b, apply.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr(baseThreadVarName));

    // Get the layout and check it's a gen_p
    GenPOp genP = dyn_cast_or_null<GenPOp>(apply.getLayout().getDefiningOp());
    if (!genP || genP.getBody().empty()) {
      // Cannot verify non-gen_p layouts yet
      return success();
    }

    // Compute addresses for all 32 threads
    SmallVector<Value> addresses;
    SmallVector<std::string> addrVarNames;

    for (int t = 0; t < WARP_SIZE; ++t) {
      // We'll create a fresh builder context for each thread
      DenseMap<Value, Value> threadValMap;

      // Map the base thread parameter to (baseThread + t)
      Value tConst = smt::IntConstantOp::create(
          b, apply.getLoc(), b.getI64IntegerAttr(t));
      Value threadId = smt::IntAddOp::create(
          b, apply.getLoc(), ValueRange{baseThread, tConst});

      // Map function arguments to thread-specific values
      auto parentFunc = apply->getParentOfType<func::FuncOp>();
      if (parentFunc) {
        bool found = false;
        
        // Find the argument with the `lego.thread_id` attribute
        for (BlockArgument arg : parentFunc.getArguments()) {
          if (parentFunc.getArgAttr(arg.getArgNumber(), "lego.thread_id")) {
            threadValMap[arg] = threadId;
            found = true;
            break;
          }
        }
        
        if (!found) {
          apply.emitWarning("Could not find a block argument with 'lego.thread_id' attribute in parent function. Verification might not be accurate.");
        }
      }

      // Build index values for this thread using a helper
      SMTBuilder threadBuilder(b, state, nextId);
      threadBuilder.valMap = threadValMap;

      SmallVector<Value> concreteIndices;
      for (Value idx : apply.getIndices()) {
        Value smtIdx = threadBuilder.getOrCreate(idx);
        concreteIndices.push_back(smtIdx);
      }

      // Compute flat index using the layout's apply region
      SmallVector<Value> flatResults;
      threadBuilder.buildRegion(genP.getBody(), concreteIndices, flatResults);
      if (flatResults.size() != 1) {
        return success();
      }

      addresses.push_back(flatResults[0]);
      addrVarNames.push_back("addr_" + std::to_string(t));
    }

    // Verify property: addresses are sequential (unit stride)
    // For i in [0, WARP_SIZE-1]: addr[i+1] - addr[i] = 1
    // If NOT true, assert the negation and check SAT

    SmallVector<Value> nonSequential;
    for (int t = 0; t < WARP_SIZE - 1; ++t) {
      Value diff = smt::IntSubOp::create(
          b, apply.getLoc(), addresses[t + 1], addresses[t]);
      Value one = smt::IntConstantOp::create(
          b, apply.getLoc(), b.getI64IntegerAttr(1));
      Value notUnitStride = smt::DistinctOp::create(
          b, apply.getLoc(), ValueRange{diff, one});
      nonSequential.push_back(notUnitStride);
    }

    // Declare named variables for addresses (BEFORE exporting!)
    for (int t = 0; t < 8 && t < WARP_SIZE; ++t) {
      Value namedAddr = smt::DeclareFunOp::create(
          b, apply.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr("addr_" + std::to_string(t)));
      Value eq = smt::EqOp::create(b, apply.getLoc(), namedAddr, addresses[t]);
      smt::AssertOp::create(b, apply.getLoc(), eq);
    }

    Value anyNonSequential = smt::OrOp::create(b, apply.getLoc(), nonSequential);
    smt::AssertOp::create(b, apply.getLoc(), anyNonSequential);

    SmallVector<std::string> allVars;
    allVars.push_back("base_thread");
    for (int t = 0; t < WARP_SIZE; ++t) {
        allVars.push_back("addr_" + std::to_string(t));
    }

    SMTResult result = smtCtx.checkSatisfiability(allVars);

    if (result.isSat) {
      std::string warnMsg = "Layout may produce non-coalesced memory accesses (unit stride not guaranteed)";
      if (result.model.count(baseThreadVarName)) {
        warnMsg += "\n  Counter-example starting at thread: " +
                   std::to_string(result.model[baseThreadVarName]);
      }

      // Show sample of addresses to illustrate the problem
      warnMsg += "\n  Sample addresses (first 8 threads):";
      for (int t = 0; t < 8 && t < WARP_SIZE; ++t) {
        std::string addrVar = "addr_" + std::to_string(t);
        if (result.model.count(addrVar)) {
          warnMsg += "\n    Thread " + std::to_string(t) + ": " +
                     std::to_string(result.model[addrVar]);
        }
      }
      warnMsg += "\n  (For coalescing, addresses should be consecutive: 0, 1, 2, 3, ...)";

      apply.emitWarning(warnMsg);
      return failure();
    } else if (result.isUnsat) {
      // Coalescing property verified!
      return success();
    } else {
      apply.emitWarning("Coalescing check returned unknown");
      return success();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyCoalescingPass() {
  return std::make_unique<LegoVerifyCoalescingPassImpl>();
}
} // namespace lego
} // namespace mlir
