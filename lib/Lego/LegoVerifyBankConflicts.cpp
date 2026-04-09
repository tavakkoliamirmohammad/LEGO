#define GEN_PASS_DEF_LEGOVERIFYBANKCONFLICTSPASS
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

    SmallVector<ApplyOp> applyOps;
    module.walk([&](ApplyOp apply) { applyOps.push_back(apply); });

    if (applyOps.empty()) return;

    AsmState state(module);
    unsigned nextId = 0;

    for (auto apply : applyOps) {
      if (failed(verifyBankConflictFree(apply, state, nextId))) {
        // Performance property — emit warnings, don't signal pass failure.
      }
    }
  }

private:
  LogicalResult verifyBankConflictFree(ApplyOp apply, AsmState &state, unsigned &nextId) {
    int WARP_SIZE = warpSize.getValue();
    int NUM_BANKS = numBanks.getValue();
    int ELEMENT_SIZE = elementSize.getValue();
    SMTSolverContext smtCtx(apply.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;

    Value baseThread;
    SmallVector<Value> addresses;
    if (failed(computeWarpAddresses(apply, apply.getLayout(),
                                    apply.getIndices(), smtCtx, state,
                                    nextId, WARP_SIZE, baseThread, addresses)))
      return success();

    // Bank conflict detection.
    // bank(addr) = (addr * ELEMENT_SIZE / 4) % NUM_BANKS
    Value numBanksConst = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(NUM_BANKS));
    Value elementSizeConst = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(ELEMENT_SIZE));
    Value fourConst = smt::IntConstantOp::create(
        b, apply.getLoc(), b.getI64IntegerAttr(4));

    SmallVector<Value> conflicts;
    for (int i = 0; i < WARP_SIZE; ++i) {
      for (int j = i + 1; j < WARP_SIZE; ++j) {
        Value byte_addr_i = smt::IntMulOp::create(
            b, apply.getLoc(), ValueRange{addresses[i], elementSizeConst});
        Value bank_word_i = smt::IntDivOp::create(
            b, apply.getLoc(), byte_addr_i, fourConst);
        Value bank_i = smt::IntModOp::create(
            b, apply.getLoc(), bank_word_i, numBanksConst);

        Value byte_addr_j = smt::IntMulOp::create(
            b, apply.getLoc(), ValueRange{addresses[j], elementSizeConst});
        Value bank_word_j = smt::IntDivOp::create(
            b, apply.getLoc(), byte_addr_j, fourConst);
        Value bank_j = smt::IntModOp::create(
            b, apply.getLoc(), bank_word_j, numBanksConst);

        Value sameBank = smt::EqOp::create(
            b, apply.getLoc(), bank_i, bank_j);
        Value diffAddr = smt::DistinctOp::create(
            b, apply.getLoc(), ValueRange{addresses[i], addresses[j]});
        Value conflict = smt::AndOp::create(
            b, apply.getLoc(), ValueRange{sameBank, diffAddr});
        conflicts.push_back(conflict);
      }
    }

    // Named variables for counter-example readability.
    for (int t = 0; t < 8 && t < WARP_SIZE; ++t) {
      Value namedAddr = smt::DeclareFunOp::create(
          b, apply.getLoc(),
          Type(b.getType<smt::IntType>()),
          b.getStringAttr("addr_" + std::to_string(t)));
      Value eqAddr = smt::EqOp::create(b, apply.getLoc(), namedAddr, addresses[t]);
      smt::AssertOp::create(b, apply.getLoc(), eqAddr);

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

    Value anyConflict = smt::OrOp::create(b, apply.getLoc(), conflicts);
    smt::AssertOp::create(b, apply.getLoc(), anyConflict);

    SmallVector<std::string> allVars;
    allVars.push_back("base_thread");
    for (int t = 0; t < WARP_SIZE; ++t) {
        allVars.push_back("addr_" + std::to_string(t));
        allVars.push_back("bank_" + std::to_string(t));
    }

    SMTResult result = smtCtx.checkSatisfiability(allVars);

    if (result.isSat) {
      std::string warnMsg = "Layout may cause shared memory bank conflicts";
      if (result.model.count("base_thread")) {
        warnMsg += "\n  Counter-example starting at thread: " +
                   std::to_string(result.model["base_thread"]);
      }
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
