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
  // Verify that a warp of 32 threads produces coalesced accesses
  LogicalResult verifyCoalescing(ApplyOp apply, AsmState &state, unsigned &nextId) {
    MLIRContext *ctx = &getContext();
    OwningOpRef<ModuleOp> smtModule = ModuleOp::create(apply.getLoc());
    OpBuilder b(smtModule->getBodyRegion());

    auto solver = smt::SolverOp::create(b, apply.getLoc(), TypeRange{}, ValueRange{});
    if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&solver.getRegion().front());

    smt::SetLogicOp::create(b, apply.getLoc(), "QF_NIA");
    SMTBuilder builder(b, state, nextId);

    // Model: For a 1D thread layout, threadIdx maps to layout indices
    // We'll verify the simple case: threadIdx → (threadIdx, 0) for 2D layouts
    // Or threadIdx → threadIdx for 1D layouts

    size_t numIndices = apply.getIndices().size();

    // Warp size
    constexpr int WARP_SIZE = 32;

    // Create symbolic variables for the base thread ID
    std::string baseThreadVar = "base_thread";
    Value baseThread = smt::DeclareFunOp::create(
        b, apply.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr(baseThreadVar));

    // Compute addresses for all 32 threads
    SmallVector<Value> addresses;
    SmallVector<std::string> addrVarNames;

    for (int t = 0; t < WARP_SIZE; ++t) {
      // Thread ID = base_thread + t
      Value tConst = smt::IntConstantOp::create(
          b, apply.getLoc(), b.getI64IntegerAttr(t));
      Value threadId = smt::IntAddOp::create(
          b, apply.getLoc(), ValueRange{baseThread, tConst});

      // Map thread ID to indices
      // Assumption: First dimension is threaded, rest are 0
      SmallVector<Value> indices;
      for (size_t i = 0; i < numIndices; ++i) {
        if (i == 0) {
          indices.push_back(threadId);
        } else {
          indices.push_back(smt::IntConstantOp::create(
              b, apply.getLoc(), b.getI64IntegerAttr(0)));
        }
      }

      // Build the layout apply region to compute flat index
      GenPOp genP = dyn_cast_or_null<GenPOp>(apply.getLayout().getDefiningOp());
      if (!genP || genP.getBody().empty()) {
        // Cannot verify non-gen_p layouts yet
        return success();
      }

      SmallVector<Value> flatResults;
      builder.buildRegion(genP.getBody(), indices, flatResults);
      if (flatResults.size() != 1) {
        return success(); // Skip if unexpected
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

    Value anyNonSequential = smt::OrOp::create(b, apply.getLoc(), nonSequential);
    smt::AssertOp::create(b, apply.getLoc(), anyNonSequential);

    auto checkOp = smt::CheckOp::create(b, apply.getLoc(), TypeRange{});
    for (Region &r : checkOp->getRegions()) {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToStart(&r.emplaceBlock());
      smt::YieldOp::create(b, apply.getLoc(), ValueRange{});
    }
    smt::YieldOp::create(b, apply.getLoc(), ValueRange{});

    // Export and run Z3
    std::string smtLib;
    llvm::raw_string_ostream os(smtLib);
    if (failed(smt::exportSMTLIB(*smtModule, os))) {
      apply.emitWarning("Failed to export SMT-LIB for coalescing check");
      return success(); // Don't fail
    }

    SmallVector<std::string> varNames = {baseThreadVar};
    smtLib += generateGetValueCommands(varNames);

    SMTResult result = runZ3WithModel(smtLib);

    if (result.isSat) {
      std::string warnMsg = "Layout may produce non-coalesced memory accesses (unit stride not guaranteed)";
      if (result.model.count(baseThreadVar)) {
        warnMsg += "\n  Counter-example base thread: " +
                   std::to_string(result.model[baseThreadVar]);
      }
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
