#define GEN_PASS_DEF_LEGOVERIFYBANKCONFLICTSPASS
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

struct LegoVerifyBankConflictsPassImpl
    : public mlir::lego::impl::LegoVerifyBankConflictsPassBase<
          LegoVerifyBankConflictsPassImpl> {
  using mlir::lego::impl::LegoVerifyBankConflictsPassBase<
      LegoVerifyBankConflictsPassImpl>::LegoVerifyBankConflictsPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    getContext().getOrLoadDialect<smt::SMTDialect>();
    ModuleOp module = getOperation();

    // Find all lego.apply operations that might access shared memory
    SmallVector<ApplyOp> applyOps;
    module.walk([&](ApplyOp apply) { applyOps.push_back(apply); });

    if (applyOps.empty()) return;

    AsmState state(module);
    unsigned nextId = 0;

    for (auto apply : applyOps) {
      // Check for bank conflicts
      // This is a performance check, so we emit warnings rather than errors
      if (failed(verifyBankConflictFree(apply, state, nextId))) {
        // Don't signal pass failure for performance properties
      }
    }
  }

private:
  // Verify that a warp of 32 threads has no bank conflicts
  // Bank conflicts occur when bank(addr_i) = bank(addr_j) for i ≠ j
  // where bank(addr) = (addr / 4) % 32 for 4-byte elements
  LogicalResult verifyBankConflictFree(ApplyOp apply, AsmState &state, unsigned &nextId) {
    MLIRContext *ctx = &getContext();
    OwningOpRef<ModuleOp> smtModule = ModuleOp::create(apply.getLoc());
    OpBuilder b(smtModule->getBodyRegion());

    auto solver = smt::SolverOp::create(b, apply.getLoc(), TypeRange{}, ValueRange{});
    if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&solver.getRegion().front());

    smt::SetLogicOp::create(b, apply.getLoc(), "QF_NIA");
    SMTBuilder builder(b, state, nextId);

    size_t numIndices = apply.getIndices().size();
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_BANKS = 32;
    constexpr int ELEM_SIZE = 4; // 4 bytes = 32 bits

    // Base thread variable
    std::string baseThreadVar = "base_thread";
    Value baseThread = smt::DeclareFunOp::create(
        b, apply.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr(baseThreadVar));

    // Compute addresses for all threads in the warp
    SmallVector<Value> addresses;
    for (int t = 0; t < WARP_SIZE; ++t) {
      Value tConst = smt::IntConstantOp::create(
          b, apply.getLoc(), b.getI64IntegerAttr(t));
      Value threadId = smt::IntAddOp::create(
          b, apply.getLoc(), ValueRange{baseThread, tConst});

      // Map thread to indices (assume first dim is threaded)
      SmallVector<Value> indices;
      for (size_t i = 0; i < numIndices; ++i) {
        if (i == 0) {
          indices.push_back(threadId);
        } else {
          indices.push_back(smt::IntConstantOp::create(
              b, apply.getLoc(), b.getI64IntegerAttr(0)));
        }
      }

      // Get the layout and compute flat index
      GenPOp genP = dyn_cast_or_null<GenPOp>(apply.getLayout().getDefiningOp());
      if (!genP || genP.getBody().empty()) {
        // Skip non-gen_p layouts
        return success();
      }

      SmallVector<Value> flatResults;
      builder.buildRegion(genP.getBody(), indices, flatResults);
      if (flatResults.size() != 1) {
        return success();
      }

      addresses.push_back(flatResults[0]);
    }

    // Check for bank conflicts
    // For each pair (i, j) where i < j, check if they access the same bank
    // Bank = (address / ELEM_SIZE) % NUM_BANKS

    Value elemSize = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(ELEM_SIZE));
    Value numBanks = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(NUM_BANKS));

    // We check if there EXISTS a conflict (SAT = conflict exists)
    SmallVector<Value> conflicts;

    for (int i = 0; i < WARP_SIZE; ++i) {
      for (int j = i + 1; j < WARP_SIZE; ++j) {
        // bank_i = (addr_i / ELEM_SIZE) % NUM_BANKS
        Value wordIdx_i = smt::IntDivOp::create(
            b, apply.getLoc(), addresses[i], elemSize);
        Value bank_i = smt::IntModOp::create(
            b, apply.getLoc(), wordIdx_i, numBanks);

        // bank_j = (addr_j / ELEM_SIZE) % NUM_BANKS
        Value wordIdx_j = smt::IntDivOp::create(
            b, apply.getLoc(), addresses[j], elemSize);
        Value bank_j = smt::IntModOp::create(
            b, apply.getLoc(), wordIdx_j, numBanks);

        // Check if same bank AND different addresses
        Value sameBank = smt::EqOp::create(
            b, apply.getLoc(), bank_i, bank_j);
        Value diffAddr = smt::DistinctOp::create(
            b, apply.getLoc(), ValueRange{addresses[i], addresses[j]});
        Value conflict = smt::AndOp::create(
            b, apply.getLoc(), ValueRange{sameBank, diffAddr});

        conflicts.push_back(conflict);
      }
    }

    // Assert that at least one conflict exists
    Value anyConflict = smt::OrOp::create(b, apply.getLoc(), conflicts);
    smt::AssertOp::create(b, apply.getLoc(), anyConflict);

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
      apply.emitWarning("Failed to export SMT-LIB for bank conflict check");
      return success();
    }

    SmallVector<std::string> varNames = {baseThreadVar};
    smtLib += generateGetValueCommands(varNames);

    SMTResult result = runZ3WithModel(smtLib);

    if (result.isSat) {
      std::string warnMsg = "Layout may cause shared memory bank conflicts";
      if (result.model.count(baseThreadVar)) {
        warnMsg += "\n  Counter-example base thread: " +
                   std::to_string(result.model[baseThreadVar]);
      }
      apply.emitWarning(warnMsg);
      return failure();
    } else if (result.isUnsat) {
      // Bank-conflict free!
      return success();
    } else {
      apply.emitWarning("Bank conflict check returned unknown");
      return success();
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyBankConflictsPass() {
  return std::make_unique<LegoVerifyBankConflictsPassImpl>();
}
} // namespace lego
} // namespace mlir
