#define GEN_PASS_DEF_LEGOVERIFYCOALESCINGPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "Lego/SMTUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
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

    SmallVector<ApplyOp> applyOps;
    module.walk([&](ApplyOp apply) { applyOps.push_back(apply); });

    if (applyOps.empty()) return;

    AsmState state(module);
    unsigned nextId = 0;

    for (auto apply : applyOps) {
      if (failed(verifyCoalescing(apply, state, nextId))) {
        // Performance property — emit warnings, don't signal pass failure.
      }
    }
  }

private:
  LogicalResult verifyCoalescing(ApplyOp apply, AsmState &state, unsigned &nextId) {
    int WARP_SIZE = warpSize.getValue();
    SMTSolverContext smtCtx(apply.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;

    Value baseThread;
    SmallVector<Value> addresses;
    if (failed(computeWarpAddresses(apply, apply.getLayout(),
                                    apply.getIndices(), smtCtx, state,
                                    nextId, WARP_SIZE, baseThread, addresses)))
      return success();

    // Verify: addresses are sequential (unit stride).
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

    // Named variables for counter-example readability.
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
    for (int t = 0; t < WARP_SIZE; ++t)
        allVars.push_back("addr_" + std::to_string(t));

    SMTResult result = smtCtx.checkSatisfiability(allVars);

    if (result.isSat) {
      std::string warnMsg = "Layout may produce non-coalesced memory accesses (unit stride not guaranteed)";
      if (result.model.count("base_thread")) {
        warnMsg += "\n  Counter-example starting at thread: " +
                   std::to_string(result.model["base_thread"]);
      }
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
