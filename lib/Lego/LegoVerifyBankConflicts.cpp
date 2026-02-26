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
  // Verify that a warp of WARP_SIZE threads has no bank conflicts
  // Bank conflicts occur when bank(addr_i) = bank(addr_j) for i ≠ j
  // where bank(addr) = (addr * elementSize / 4) % numBanks
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
    int WARP_SIZE = warpSize.getValue();
    int NUM_BANKS = numBanks.getValue();
    int ELEMENT_SIZE = elementSize.getValue();
    std::string baseThreadVarName = baseThreadVar.getValue();

    // Base thread variable
    Value baseThread = smt::DeclareFunOp::create(
        b, apply.getLoc(),
        Type(b.getType<smt::IntType>()),
        b.getStringAttr(baseThreadVarName));

    // Get the layout and check it's a gen_p
    GenPOp genP = dyn_cast_or_null<GenPOp>(apply.getLayout().getDefiningOp());
    if (!genP || genP.getBody().empty()) {
      // Skip non-gen_p layouts
      return success();
    }

    // Build a function that computes the address for a given thread ID
    // The apply operation's indices are expressions in terms of the function arguments
    // We need to translate those index expressions into SMT

    // First, translate each index argument into SMT, treating BlockArguments as symbolic
    SmallVector<Value> symbolicIndices;
    for (Value idx : apply.getIndices()) {
      Value smtIdx = builder.getOrCreate(idx);
      symbolicIndices.push_back(smtIdx);
    }

    // Compute addresses for all threads in the warp
    SmallVector<Value> addresses;
    for (int t = 0; t < WARP_SIZE; ++t) {
      // For each thread, substitute thread ID into the index expressions
      // This requires building the index computation with thread = t

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
          // We can optionally return success() here to abort check early
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
    }

    // Check for bank conflicts
    // The layout generates element indices (not byte addresses)
    // byte_address = element_index * ELEMENT_SIZE
    // bank_word = byte_address / 4
    // bank = bank_word % NUM_BANKS
    // Conflicts occur when multiple threads access different addresses in the same bank

    Value numBanksConst = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(NUM_BANKS));
    Value elementSizeConst = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(ELEMENT_SIZE));
    Value fourConst = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(4));

    // We check if there EXISTS a conflict (SAT = conflict exists)
    SmallVector<Value> conflicts;

    for (int i = 0; i < WARP_SIZE; ++i) {
      for (int j = i + 1; j < WARP_SIZE; ++j) {
        // bank_i = (addr_i * ELEMENT_SIZE / 4) % NUM_BANKS
        Value byte_addr_i = smt::IntMulOp::create(
            b, apply.getLoc(), ValueRange{addresses[i], elementSizeConst});
        Value bank_word_i = smt::IntDivOp::create(
            b, apply.getLoc(), byte_addr_i, fourConst);
        Value bank_i = smt::IntModOp::create(
            b, apply.getLoc(), bank_word_i, numBanksConst);

        // bank_j = (addr_j * ELEMENT_SIZE / 4) % NUM_BANKS
        Value byte_addr_j = smt::IntMulOp::create(
            b, apply.getLoc(), ValueRange{addresses[j], elementSizeConst});
        Value bank_word_j = smt::IntDivOp::create(
            b, apply.getLoc(), byte_addr_j, fourConst);
        Value bank_j = smt::IntModOp::create(
            b, apply.getLoc(), bank_word_j, numBanksConst);

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

    // Declare named variables for addresses and banks (BEFORE exporting!)
    for (int t = 0; t < 8 && t < WARP_SIZE; ++t) {
      // Address
      Value namedAddr = smt::DeclareFunOp::create(
          b, apply.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr("addr_" + std::to_string(t)));
      Value eqAddr = smt::EqOp::create(b, apply.getLoc(), namedAddr, addresses[t]);
      smt::AssertOp::create(b, apply.getLoc(), eqAddr);

      // Bank number
      Value byte_addr_t = smt::IntMulOp::create(
          b, apply.getLoc(), ValueRange{addresses[t], elementSizeConst});
      Value bank_word_t = smt::IntDivOp::create(
          b, apply.getLoc(), byte_addr_t, fourConst);
      Value bank = smt::IntModOp::create(b, apply.getLoc(), bank_word_t, numBanksConst);
      Value namedBank = smt::DeclareFunOp::create(
          b, apply.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr("bank_" + std::to_string(t)));
      Value eqBank = smt::EqOp::create(b, apply.getLoc(), namedBank, bank);
      smt::AssertOp::create(b, apply.getLoc(), eqBank);
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

    SmallVector<std::string> varNames = {baseThreadVarName};

    // Add named variables for a sample of addresses and banks
    SmallVector<std::string> allVars = varNames;
    // Show first 8 addresses and their bank assignments
    for (int t = 0; t < 8 && t < WARP_SIZE; ++t) {
      allVars.push_back("addr_" + std::to_string(t));
      allVars.push_back("bank_" + std::to_string(t));
    }

    // Remove the (reset) command that clears solver state before get-value
    size_t resetPos = smtLib.rfind("(reset)");
    if (resetPos != std::string::npos) {
      smtLib.erase(resetPos, 8); // Remove "(reset)\n"
    }

    smtLib += generateGetValueCommands(allVars);

    SMTResult result = runZ3WithModel(smtLib);

    if (result.isSat) {
      std::string warnMsg = "Layout may cause shared memory bank conflicts";
      if (result.model.count(baseThreadVarName)) {
        warnMsg += "\n  Counter-example starting at thread: " +
                   std::to_string(result.model[baseThreadVarName]);
      }

      // Show sample of addresses and banks to illustrate conflicts
      warnMsg += "\n  Sample memory access pattern (first 8 threads):";
      for (int t = 0; t < 8 && t < WARP_SIZE; ++t) {
        std::string addrVar = "addr_" + std::to_string(t);
        std::string bankVar = "bank_" + std::to_string(t);
        if (result.model.count(addrVar) && result.model.count(bankVar)) {
          warnMsg += "\n    Thread " + std::to_string(t) +
                     ": addr=" + std::to_string(result.model[addrVar]) +
                     ", bank=" + std::to_string(result.model[bankVar]);
        }
      }
      warnMsg += "\n  (Conflict occurs when multiple threads access different addresses in the same bank)";

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
