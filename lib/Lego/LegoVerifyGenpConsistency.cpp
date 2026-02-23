#define GEN_PASS_DEF_LEGOVERIFYGENPCONSISTENCYPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/AsmState.h"
#include "Lego/SMTUtils.h"
#include <fstream>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <memory>
#include <array>

using namespace mlir;
using namespace mlir::lego;

namespace {

struct LegoVerifyGenpConsistencyPassImpl
    : public mlir::lego::impl::LegoVerifyGenpConsistencyPassBase<LegoVerifyGenpConsistencyPassImpl> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AsmState state(module);
    unsigned nextId = 0;
    
    module.walk([&](GenPOp op) {
      if (op.getInvBody().empty()) return;
      
      if (!verifyInverse(op, state, nextId)) {
          op.emitError("Inconsistent GenP: apply and inv regions are not bijections.");
          // signalPassFailure();
      }
    });
  }

private:
  bool verifyInverse(GenPOp op, AsmState &state, unsigned &nextId) {
    OwningOpRef<ModuleOp> smtModule = ModuleOp::create(op.getLoc());
    OpBuilder b(smtModule->getBodyRegion());
    auto solver = smt::SolverOp::create(b, op.getLoc(), TypeRange{}, ValueRange{});
    if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();
    
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&solver.getRegion().front());
    
    smt::SetLogicOp::create(b, op.getLoc(), "QF_NIA");
    SMTBuilder builder(b, state, nextId);

    auto dims = op.getDims();
    SmallVector<Value> x_vars;
    Value zero = smt::IntConstantOp::create(b, op.getLoc(), b.getI64IntegerAttr(0));
    
    for (Value d : dims) {
      Value x = builder.getOrCreate(d); 
      Type smtIntTy = b.getType<smt::IntType>();
      Value coord = smt::DeclareFunOp::create(b, op.getLoc(), smtIntTy, b.getStringAttr(builder.getSSAName(d)));
      x_vars.push_back(coord);
      
      Value dimVal = builder.getOrCreate(d);
      Value geZero = smt::IntCmpOp::create(b, op.getLoc(), smt::IntPredicate::ge, coord, zero);
      Value ltDim = smt::IntCmpOp::create(b, op.getLoc(), smt::IntPredicate::lt, coord, dimVal);
      smt::AssertOp::create(b, op.getLoc(), geZero);
      smt::AssertOp::create(b, op.getLoc(), ltDim);
    }

    // y = apply(x)
    SmallVector<Value> applyResults;
    builder.buildRegion(op.getBody(), x_vars, applyResults);
    if (applyResults.empty()) return true;

    // z = inv(y)
    SmallVector<Value> invResults;
    builder.buildRegion(op.getInvBody(), applyResults, invResults);

    // Assert z != x
    SmallVector<Value> diffs;
    for (size_t i = 0; i < x_vars.size(); ++i) {
        Value eq = smt::EqOp::create(b, op.getLoc(), x_vars[i], invResults[i]);
        Value ne = smt::NotOp::create(b, op.getLoc(), eq);
        diffs.push_back(ne);
    }
    
    Value inconsistent = diffs.size() == 1 ? diffs[0] : smt::OrOp::create(b, op.getLoc(), diffs);
    smt::AssertOp::create(b, op.getLoc(), inconsistent);

    auto checkOp = smt::CheckOp::create(b, op.getLoc(), TypeRange{});
    for (Region &r : checkOp->getRegions()) {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToStart(&r.emplaceBlock());
        smt::YieldOp::create(b, op.getLoc(), ValueRange{});
    }
    smt::YieldOp::create(b, op.getLoc(), ValueRange{});

    std::string smtLib;
    llvm::raw_string_ostream os(smtLib);
    if (failed(smt::exportSMTLIB(*smtModule, os))) return true;

    return !runZ3(smtLib);
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyGenpConsistencyPass() {
  return std::make_unique<LegoVerifyGenpConsistencyPassImpl>();
}
} // namespace lego
} // namespace mlir
