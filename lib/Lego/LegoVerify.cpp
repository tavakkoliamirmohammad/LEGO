#define GEN_PASS_DEF_LEGOVERIFYPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
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

struct LegoVerifyPassImpl
    : public mlir::lego::impl::LegoVerifyPassBase<LegoVerifyPassImpl> {
  using mlir::lego::impl::LegoVerifyPassBase<
      LegoVerifyPassImpl>::LegoVerifyPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    getContext().getOrLoadDialect<smt::SMTDialect>();
    ModuleOp module = getOperation();

    SmallVector<CheckOp> checkOps;
    module.walk([&](CheckOp op) { checkOps.push_back(op); });

    if (checkOps.empty())
      return;

    // Merge check ops that reference the same flat index: combine their
    // property flags so we only build warp addresses once per operand.
    DenseMap<Value, SmallVector<CheckOp>> byOperand;
    for (auto checkOp : checkOps)
      byOperand[checkOp.getFlatIndex()].push_back(checkOp);

    SmallVector<CheckOp> mergedOps;
    for (auto &[flatIdx, ops] : byOperand) {
      if (ops.size() == 1) {
        mergedOps.push_back(ops[0]);
        continue;
      }
      // Merge flags into the first op, erase the rest.
      CheckOp primary = ops[0];
      for (unsigned i = 1; i < ops.size(); ++i) {
        if (ops[i].getCoalescing() && !primary.getCoalescing())
          primary.setCoalescing(true);
        if (ops[i].getBankConflictFree() && !primary.getBankConflictFree())
          primary.setBankConflictFree(true);
        ops[i].erase();
      }
      mergedOps.push_back(primary);
    }

    AsmState state(module);
    unsigned nextId = 0;

    for (auto checkOp : mergedOps) {
      if (checkOp.getCoalescing())
        verifyCoalescing(checkOp, state, nextId);
      if (checkOp.getBankConflictFree())
        verifyBankConflictFree(checkOp, state, nextId);
      checkOp.erase();
    }
  }

private:
  std::optional<BlockArgument> findThreadArg(Operation *op) {
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return std::nullopt;
    for (BlockArgument arg : parentFunc.getArguments()) {
      if (parentFunc.getArgAttr(arg.getArgNumber(), "lego.thread_id"))
        return arg;
    }
    return std::nullopt;
  }

  LogicalResult buildWarpAddresses(
      CheckOp checkOp, int warpSize, AsmState &state, unsigned &nextId,
      SMTSolverContext &smtCtx, Value &baseThread,
      SmallVectorImpl<Value> &addresses) {
    OpBuilder &b = *smtCtx.b;
    Location loc = checkOp.getLoc();

    auto threadArg = findThreadArg(checkOp);
    if (!threadArg) {
      checkOp.emitWarning("Could not find a block argument with "
                          "'lego.thread_id' attribute — skipping verification");
      return failure();
    }

    baseThread = smt::DeclareFunOp::create(
        b, loc, Type(b.getType<smt::IntType>()),
        b.getStringAttr("base_thread"));

    Value flatIndex = checkOp.getFlatIndex();

    for (int t = 0; t < warpSize; ++t) {
      Value tConst = smt::IntConstantOp::create(b, loc, b.getI64IntegerAttr(t));
      Value threadId = smt::IntAddOp::create(b, loc, ValueRange{baseThread, tConst});

      DenseMap<Value, Value> threadValMap;
      threadValMap[*threadArg] = threadId;

      SMTBuilder threadSMT(b, state, nextId);
      threadSMT.valMap = threadValMap;

      Value smtAddr = threadSMT.getOrCreate(flatIndex);
      addresses.push_back(smtAddr);
    }
    return success();
  }

  void verifyCoalescing(CheckOp checkOp, AsmState &state, unsigned &nextId) {
    int warpSize = checkOp.getWarpSize();
    SMTSolverContext smtCtx(checkOp.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;

    Value baseThread;
    SmallVector<Value> addresses;
    if (failed(buildWarpAddresses(checkOp, warpSize, state, nextId,
                                  smtCtx, baseThread, addresses)))
      return;

    Value zero = smt::IntConstantOp::create(b, checkOp.getLoc(), b.getI64IntegerAttr(0));
    Value baseGeZero = smt::IntCmpOp::create(
        b, checkOp.getLoc(), smt::IntPredicate::ge, baseThread, zero);
    smt::AssertOp::create(b, checkOp.getLoc(), baseGeZero);

    SmallVector<Value> nonSequential;
    for (int t = 0; t < warpSize - 1; ++t) {
      Value diff = smt::IntSubOp::create(
          b, checkOp.getLoc(), addresses[t + 1], addresses[t]);
      Value one = smt::IntConstantOp::create(
          b, checkOp.getLoc(), b.getI64IntegerAttr(1));
      Value notUnitStride = smt::DistinctOp::create(
          b, checkOp.getLoc(), ValueRange{diff, one});
      nonSequential.push_back(notUnitStride);
    }

    for (int t = 0; t < 8 && t < warpSize; ++t) {
      Value namedAddr = smt::DeclareFunOp::create(
          b, checkOp.getLoc(), Type(b.getType<smt::IntType>()),
          b.getStringAttr("addr_" + std::to_string(t)));
      Value eq = smt::EqOp::create(b, checkOp.getLoc(), namedAddr, addresses[t]);
      smt::AssertOp::create(b, checkOp.getLoc(), eq);
    }

    Value anyNonSequential = smt::OrOp::create(b, checkOp.getLoc(), nonSequential);
    smt::AssertOp::create(b, checkOp.getLoc(), anyNonSequential);

    SmallVector<std::string> allVars;
    allVars.push_back("base_thread");
    for (int t = 0; t < 8 && t < warpSize; ++t)
      allVars.push_back("addr_" + std::to_string(t));

    SMTResult result = smtCtx.checkSatisfiability(allVars, /*timeoutMs=*/120000);

    if (result.isSat) {
      std::string warnMsg = "Layout may produce non-coalesced memory accesses (unit stride not guaranteed)";
      if (result.model.count("base_thread")) {
        warnMsg += "\n  Counter-example starting at thread: " +
                   std::to_string(result.model["base_thread"]);
      }
      warnMsg += "\n  Sample addresses (first 8 threads):";
      for (int t = 0; t < 8 && t < warpSize; ++t) {
        std::string addrVar = "addr_" + std::to_string(t);
        if (result.model.count(addrVar)) {
          warnMsg += "\n    Thread " + std::to_string(t) + ": " +
                     std::to_string(result.model[addrVar]);
        }
      }
      warnMsg += "\n  (For coalescing, addresses should be consecutive: 0, 1, 2, 3, ...)";
      checkOp.emitWarning(warnMsg);
    } else if (result.isUnknown) {
      checkOp.emitWarning("Coalescing check returned unknown");
    }
  }

  void verifyBankConflictFree(CheckOp checkOp, AsmState &state, unsigned &nextId) {
    int warpSize = checkOp.getWarpSize();
    int numBanks = checkOp.getNumBanks();
    int elementSize = checkOp.getElementSize();
    SMTSolverContext smtCtx(checkOp.getLoc(), state, nextId);
    OpBuilder &b = *smtCtx.b;

    Value baseThread;
    SmallVector<Value> addresses;
    if (failed(buildWarpAddresses(checkOp, warpSize, state, nextId,
                                  smtCtx, baseThread, addresses)))
      return;

    Value zero = smt::IntConstantOp::create(b, checkOp.getLoc(), b.getI64IntegerAttr(0));
    Value baseGeZero = smt::IntCmpOp::create(
        b, checkOp.getLoc(), smt::IntPredicate::ge, baseThread, zero);
    smt::AssertOp::create(b, checkOp.getLoc(), baseGeZero);

    Value numBanksConst = smt::IntConstantOp::create(
        b, checkOp.getLoc(), b.getI64IntegerAttr(numBanks));

    SmallVector<Value> banks;
    bool simplified = (elementSize % 4 == 0);
    int wordScale = simplified ? elementSize / 4 : 0;
    Value scaleConst = (simplified && wordScale != 1)
        ? smt::IntConstantOp::create(b, checkOp.getLoc(), b.getI64IntegerAttr(wordScale))
        : Value();
    Value elementSizeConst = !simplified
        ? smt::IntConstantOp::create(b, checkOp.getLoc(), b.getI64IntegerAttr(elementSize))
        : Value();
    Value fourConst = !simplified
        ? smt::IntConstantOp::create(b, checkOp.getLoc(), b.getI64IntegerAttr(4))
        : Value();

    for (int t = 0; t < warpSize; ++t) {
      Value bankInput;
      if (simplified) {
        if (wordScale == 1) {
          bankInput = addresses[t];
        } else {
          bankInput = smt::IntMulOp::create(
              b, checkOp.getLoc(), ValueRange{addresses[t], scaleConst});
        }
      } else {
        Value byteAddr = smt::IntMulOp::create(
            b, checkOp.getLoc(), ValueRange{addresses[t], elementSizeConst});
        bankInput = smt::IntDivOp::create(b, checkOp.getLoc(), byteAddr, fourConst);
      }
      Value bank = smt::IntModOp::create(
          b, checkOp.getLoc(), bankInput, numBanksConst);
      banks.push_back(bank);
    }

    SmallVector<Value> conflicts;
    for (int i = 0; i < warpSize; ++i) {
      for (int j = i + 1; j < warpSize; ++j) {
        Value sameBank = smt::EqOp::create(
            b, checkOp.getLoc(), banks[i], banks[j]);
        Value diffAddr = smt::DistinctOp::create(
            b, checkOp.getLoc(), ValueRange{addresses[i], addresses[j]});
        Value conflict = smt::AndOp::create(
            b, checkOp.getLoc(), ValueRange{sameBank, diffAddr});
        conflicts.push_back(conflict);
      }
    }

    for (int t = 0; t < 8 && t < warpSize; ++t) {
      Value namedAddr = smt::DeclareFunOp::create(
          b, checkOp.getLoc(), Type(b.getType<smt::IntType>()),
          b.getStringAttr("addr_" + std::to_string(t)));
      Value eqAddr = smt::EqOp::create(b, checkOp.getLoc(), namedAddr, addresses[t]);
      smt::AssertOp::create(b, checkOp.getLoc(), eqAddr);

      Value namedBank = smt::DeclareFunOp::create(
          b, checkOp.getLoc(), Type(b.getType<smt::IntType>()),
          b.getStringAttr("bank_" + std::to_string(t)));
      Value eqBank = smt::EqOp::create(b, checkOp.getLoc(), namedBank, banks[t]);
      smt::AssertOp::create(b, checkOp.getLoc(), eqBank);
    }

    Value anyConflict = smt::OrOp::create(b, checkOp.getLoc(), conflicts);
    smt::AssertOp::create(b, checkOp.getLoc(), anyConflict);

    SmallVector<std::string> allVars;
    allVars.push_back("base_thread");
    for (int t = 0; t < 8 && t < warpSize; ++t) {
      allVars.push_back("addr_" + std::to_string(t));
      allVars.push_back("bank_" + std::to_string(t));
    }

    SMTResult result = smtCtx.checkSatisfiability(allVars, /*timeoutMs=*/120000);

    if (result.isSat) {
      std::string warnMsg = "Layout may cause shared memory bank conflicts";
      if (result.model.count("base_thread")) {
        warnMsg += "\n  Counter-example starting at thread: " +
                   std::to_string(result.model["base_thread"]);
      }
      warnMsg += "\n  Sample memory access pattern (first 8 threads):";
      for (int t = 0; t < 8 && t < warpSize; ++t) {
        std::string addrVar = "addr_" + std::to_string(t);
        std::string bankVar = "bank_" + std::to_string(t);
        if (result.model.count(addrVar) && result.model.count(bankVar)) {
          warnMsg += "\n    Thread " + std::to_string(t) +
                     ": addr=" + std::to_string(result.model[addrVar]) +
                     ", bank=" + std::to_string(result.model[bankVar]);
        }
      }
      warnMsg += "\n  (Conflict occurs when multiple threads access different addresses in the same bank)";
      checkOp.emitWarning(warnMsg);
    } else if (result.isUnknown) {
      checkOp.emitWarning("Bank conflict check returned unknown");
    }
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyPass() {
  return std::make_unique<LegoVerifyPassImpl>();
}
} // namespace lego
} // namespace mlir
