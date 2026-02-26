#define GEN_PASS_DEF_LEGOEXTERNALSMTVERIFIERPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
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

struct LegoExternalSMTVerifierPassImpl
    : public mlir::lego::impl::LegoExternalSMTVerifierPassBase<
          LegoExternalSMTVerifierPassImpl> {
  using mlir::lego::impl::LegoExternalSMTVerifierPassBase<
      LegoExternalSMTVerifierPassImpl>::LegoExternalSMTVerifierPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    getContext().getOrLoadDialect<smt::SMTDialect>();
    ModuleOp module = getOperation();
    SmallVector<AssumeOp> assumes;
    SmallVector<AssertApplyBoundsOp> applies;
    SmallVector<AssertInvBoundsOp> invs;

    module.walk([&](Operation *op) {
      if (auto assume = dyn_cast<AssumeOp>(op)) assumes.push_back(assume);
      if (auto apply = dyn_cast<AssertApplyBoundsOp>(op)) applies.push_back(apply);
      if (auto inv = dyn_cast<AssertInvBoundsOp>(op)) invs.push_back(inv);
    });

    if (applies.empty() && invs.empty()) return;

    AsmState state(module);
    unsigned nextId = 0;
    
    for (auto apply : applies) {
      SMTSolverContext smtCtx(apply.getLoc(), state, nextId);
      OpBuilder &b = *smtCtx.b;
      SMTBuilder &builder = *smtCtx.builder;

      SmallVector<Value> dims = getLayoutInputDims(apply.getLayout());
      for (Value d : dims) builder.getOrCreate(d);
      for (Value idx : apply.getIndices()) builder.getOrCreate(idx);

      for (auto assume : assumes) {
         Value cond = builder.getOrCreate(assume.getCondition());
         smt::AssertOp::create(b, assume.getLoc(), cond);
      }

      SmallVector<Value> oobExprs;
      Value zero = smt::IntConstantOp::create(b, apply.getLoc(), b.getI64IntegerAttr(0));
      for (size_t i = 0; i < apply.getIndices().size(); ++i) {
          Value idx = builder.getOrCreate(apply.getIndices()[i]);
          Value dim = (i < dims.size()) ? builder.getOrCreate(dims[i]) : 
                      smt::IntConstantOp::create(b, apply.getLoc(), b.getI64IntegerAttr(1));
          
          oobExprs.push_back(smt::IntCmpOp::create(b, apply.getLoc(), smt::IntPredicate::lt, idx, zero));
          oobExprs.push_back(smt::IntCmpOp::create(b, apply.getLoc(), smt::IntPredicate::ge, idx, dim));
      }
      Value finalOOB = oobExprs.size() == 1 ? oobExprs[0] : smt::OrOp::create(b, apply.getLoc(), oobExprs);
      smt::AssertOp::create(b, apply.getLoc(), finalOOB);
      SMTResult result = smtCtx.checkSatisfiability({});
      if (result.isSat) {
          apply.emitError("Out-of-bounds access is possible (proven by Z3)");
      }
    }

    for (auto inv : invs) {
      SMTSolverContext smtCtx(inv.getLoc(), state, nextId);
      OpBuilder &b = *smtCtx.b;
      SMTBuilder &builder = *smtCtx.builder;

      SmallVector<Value> dims = getLayoutInputDims(inv.getLayout());
      for (Value d : dims) builder.getOrCreate(d);
      builder.getOrCreate(inv.getFlatIndex());

      for (auto assume : assumes) {
         Value cond = builder.getOrCreate(assume.getCondition());
         smt::AssertOp::create(b, assume.getLoc(), cond);
      }

      Value vol = smt::IntConstantOp::create(b, inv.getLoc(), b.getI64IntegerAttr(1));
      if (!dims.empty()) {
          SmallVector<Value> dimVals;
          for (Value d : dims) dimVals.push_back(builder.getOrCreate(d));
          vol = dimVals.size() == 1 ? dimVals[0] : smt::IntMulOp::create(b, inv.getLoc(), dimVals);
      }

      Value flatIdx = builder.getOrCreate(inv.getFlatIndex());
      Value zero = smt::IntConstantOp::create(b, inv.getLoc(), b.getI64IntegerAttr(0));
      Value ltZero = smt::IntCmpOp::create(b, inv.getLoc(), smt::IntPredicate::lt, flatIdx, zero);
      Value geVol = smt::IntCmpOp::create(b, inv.getLoc(), smt::IntPredicate::ge, flatIdx, vol);
      Value oob = smt::OrOp::create(b, inv.getLoc(), ValueRange{ltZero, geVol});
      
      smt::AssertOp::create(b, inv.getLoc(), oob);
      SMTResult result = smtCtx.checkSatisfiability({});
      if (result.isSat) {
          inv.emitError("Out-of-bounds flat index is possible (proven by Z3)");
      }
    }

    // Erase markers after verification
    for (auto apply : applies) apply.erase();
    for (auto inv : invs) inv.erase();
    for (auto assume : assumes) assume.erase();
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoExternalSMTVerifierPass() {
  return std::make_unique<LegoExternalSMTVerifierPassImpl>();
}
} // namespace lego
} // namespace mlir
