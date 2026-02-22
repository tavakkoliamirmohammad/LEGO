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

struct SMTBuilder {
  OpBuilder &builder;
  DenseMap<Value, Value> valMap;
  AsmState &state;
  unsigned nextId = 0;

  SMTBuilder(OpBuilder &b, AsmState &s) 
    : builder(b), state(s) {}

  std::string getSSAName(Value v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    v.printAsOperand(os, state);
    
    // Sanitize the name for SMT-LIB
    std::string sanitized;
    for (char c : s) {
      if (isalnum(c)) {
        sanitized += c;
      } else if (c == '%') {
        // Skip prefix
      } else {
        sanitized += '_';
      }
    }
    
    if (sanitized.empty() || sanitized.find("UNKNOWN") != std::string::npos) {
      return "v" + std::to_string(nextId++);
    }
    return sanitized;
  }

  Value getOrCreate(Value v) {
    if (valMap.count(v)) return valMap[v];
    
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      // Input argument -> symbolic variable
      Type smtTy = v.getType().isInteger(1) ? 
                   Type(builder.getType<smt::BoolType>()) : 
                   Type(builder.getType<smt::IntType>());
      Value var = smt::DeclareFunOp::create(builder, v.getLoc(), smtTy, builder.getStringAttr(getSSAName(v)));
      valMap[v] = var;
      return var;
    }

    for (Value operand : defOp->getOperands()) {
      getOrCreate(operand);
    }

    Location loc = v.getLoc();
    if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        if (v.getType().isInteger(1)) {
          valMap[v] = smt::BoolConstantOp::create(builder, loc, intAttr.getInt() != 0);
        } else {
          valMap[v] = smt::IntConstantOp::create(builder, loc, intAttr);
        }
      }
    } else if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
      valMap[v] = smt::IntAddOp::create(builder, loc, ValueRange{valMap[addOp.getLhs()], valMap[addOp.getRhs()]});
    } else if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
      valMap[v] = smt::IntSubOp::create(builder, loc, valMap[subOp.getLhs()], valMap[subOp.getRhs()]);
    } else if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
      valMap[v] = smt::IntMulOp::create(builder, loc, ValueRange{valMap[mulOp.getLhs()], valMap[mulOp.getRhs()]});
    } else if (auto divOp = dyn_cast<arith::DivUIOp>(defOp)) {
      valMap[v] = smt::IntDivOp::create(builder, loc, valMap[divOp.getLhs()], valMap[divOp.getRhs()]);
    } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(defOp)) {
      valMap[v] = smt::IntDivOp::create(builder, loc, valMap[divSIOp.getLhs()], valMap[divSIOp.getRhs()]);
    } else if (auto remOp = dyn_cast<arith::RemUIOp>(defOp)) {
      valMap[v] = smt::IntModOp::create(builder, loc, valMap[remOp.getLhs()], valMap[remOp.getRhs()]);
    } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(defOp)) {
      valMap[v] = smt::IntModOp::create(builder, loc, valMap[remSIOp.getLhs()], valMap[remSIOp.getRhs()]);
    } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
      auto lhs = valMap[cmpOp.getLhs()];
      auto rhs = valMap[cmpOp.getRhs()];
      switch (cmpOp.getPredicate()) {
        case arith::CmpIPredicate::eq: 
          valMap[v] = smt::EqOp::create(builder, loc, lhs, rhs); break;
        case arith::CmpIPredicate::ne: 
          valMap[v] = smt::DistinctOp::create(builder, loc, ValueRange{lhs, rhs}); break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult: 
          valMap[v] = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::lt, lhs, rhs); break;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule: 
          valMap[v] = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::le, lhs, rhs); break;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt: 
          valMap[v] = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::gt, lhs, rhs); break;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge: 
          valMap[v] = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge, lhs, rhs); break;
      }
    } else if (isa<arith::IndexCastOp, arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(defOp)) {
      valMap[v] = getOrCreate(defOp->getOperand(0));
    }

    if (!valMap.count(v)) {
      // Fallback: create a new symbolic variable
      Type smtTy = v.getType().isInteger(1) ? 
                   Type(builder.getType<smt::BoolType>()) : 
                   Type(builder.getType<smt::IntType>());
      valMap[v] = smt::DeclareFunOp::create(builder, loc, smtTy, builder.getStringAttr(getSSAName(v)));
    }
    return valMap[v];
  }
};

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
    
    MLIRContext *ctx = &getContext();
    for (auto apply : applies) {
      OwningOpRef<ModuleOp> smtModule = ModuleOp::create(apply.getLoc());
      OpBuilder b(smtModule->getBodyRegion());
      auto solver = smt::SolverOp::create(b, apply.getLoc(), TypeRange{}, ValueRange{});
      if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();
      
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&solver.getRegion().front());
      
      smt::SetLogicOp::create(b, apply.getLoc(), "QF_NIA");
      SMTBuilder builder(b, state);

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
      
      auto checkOp = smt::CheckOp::create(b, apply.getLoc(), TypeRange{});
      for (Region &r : checkOp->getRegions()) {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(&r.emplaceBlock());
          smt::YieldOp::create(b, apply.getLoc(), ValueRange{});
      }
      smt::YieldOp::create(b, apply.getLoc(), ValueRange{});

      
      std::string smtLib;
      llvm::raw_string_ostream os(smtLib);
      if (failed(smt::exportSMTLIB(*smtModule, os))) {
          apply.emitError("Failed to export SMT-LIB");
          signalPassFailure();
          continue;
      }


      if (verifyBounds(apply, smtLib)) {
          apply.emitError("Out-of-bounds access is possible (proven by Z3)");
          signalPassFailure();
      }
    }

    for (auto inv : invs) {
      OwningOpRef<ModuleOp> smtModule = ModuleOp::create(inv.getLoc());
      OpBuilder b(smtModule->getBodyRegion());
      auto solver = smt::SolverOp::create(b, inv.getLoc(), TypeRange{}, ValueRange{});
      if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();
      
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(&solver.getRegion().front());
      
      smt::SetLogicOp::create(b, inv.getLoc(), "QF_NIA");
      SMTBuilder builder(b, state);

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
      auto checkOp = smt::CheckOp::create(b, inv.getLoc(), TypeRange{});
      for (Region &r : checkOp->getRegions()) {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPointToStart(&r.emplaceBlock());
          smt::YieldOp::create(b, inv.getLoc(), ValueRange{});
      }
      smt::YieldOp::create(b, inv.getLoc(), ValueRange{});

      std::string smtLib;
      llvm::raw_string_ostream os(smtLib);
      if (failed(smt::exportSMTLIB(*smtModule, os))) {
          inv.emitError("Failed to export SMT-LIB");
          signalPassFailure();
          continue;
      }


      if (verifyBounds(inv, smtLib)) {
          inv.emitError("Out-of-bounds flat index is possible (proven by Z3)");
          signalPassFailure();
      }
    }

    // Erase markers after verification
    for (auto apply : applies) apply.erase();
    for (auto inv : invs) inv.erase();
    for (auto assume : assumes) assume.erase();
  }

private:
  bool verifyBounds(Operation *op, const std::string &smtLib) {
    int out_pipe[2]; // Parent write, child read
    int in_pipe[2];  // Child write, parent read
    if (pipe(out_pipe) == -1 || pipe(in_pipe) == -1) return false;

    pid_t pid = fork();
    if (pid == -1) return false;

    if (pid == 0) {
        // Child process
        dup2(out_pipe[0], STDIN_FILENO);
        dup2(in_pipe[1], STDOUT_FILENO);
        dup2(in_pipe[1], STDERR_FILENO); // Capture errors too

        close(out_pipe[0]);
        close(out_pipe[1]);
        close(in_pipe[0]);
        close(in_pipe[1]);

        execlp("z3", "z3", "-in", nullptr);
        _exit(1);
    }

    // Parent process
    close(out_pipe[0]);
    close(in_pipe[1]);

    // Send SMT-LIB to Z3. 
    // The SMT-LIB export usually ends with (check-sat)\n(reset).
    // We want to insert (get-model) between them.
    std::string modifiedLib = smtLib;
    size_t pos = modifiedLib.find("(check-sat)");
    if (pos != std::string::npos) {
        modifiedLib.insert(pos + 11, "\n(get-model)");
    }
    
    write(out_pipe[1], modifiedLib.c_str(), modifiedLib.size());
    close(out_pipe[1]); // EOF to Z3

    // Read response
    std::string result;
    char buffer[4096];
    ssize_t n;
    while ((n = read(in_pipe[0], buffer, sizeof(buffer))) > 0) {
        result.append(buffer, n);
    }
    close(in_pipe[0]);
    waitpid(pid, nullptr, 0);

    // Z3 output handling:
    // If we see "sat", the subsequent output is the model.
    if (result.find("sat") != std::string::npos && result.find("unsat") == std::string::npos) {
        llvm::errs() << "--- Z3 Counter-example ---\n" << result << "\n--------------------------\n";
        return true;
    }
    return false;
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
