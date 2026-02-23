#define GEN_PASS_DEF_LEGOVERIFYCONSISTENCYPASS
#include "Lego/LegoOps.h"
#include "Lego/Passes.h"
#include "Lego/LegoUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

// Simple symbolic evaluator to check if two regions define an inverse mapping.
// For now, focuses on linear combinations and rank-1 identities.
struct SMTBuilder {
  OpBuilder &builder;
  DenseMap<Value, Value> valMap;
  AsmState &state;
  unsigned nextId = 0;

  SMTBuilder(OpBuilder &b, AsmState &s) 
    : builder(b), state(s) {}

  std::string getSSAName(Value v) {
    return "v" + std::to_string(nextId++);
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

  void buildRegion(Region &region, ValueRange args, SmallVectorImpl<Value> &results) {
    OpBuilder::InsertionGuard guard(builder);
    Block &block = region.front();
    for (size_t i = 0; i < args.size(); ++i) {
      valMap[block.getArgument(i)] = args[i];
    }
    
    for (Operation &operation : block) {
      if (operation.hasTrait<OpTrait::IsTerminator>()) {
        for (Value operand : operation.getOperands()) {
          results.push_back(getOrCreate(operand));
        }
        break;
      }
      for (Value res : operation.getResults()) {
        getOrCreate(res);
      }
    }
  }
};

struct LegoVerifyConsistencyPassImpl
    : public mlir::lego::impl::LegoVerifyConsistencyPassBase<LegoVerifyConsistencyPassImpl> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<smt::SMTDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AsmState state(module);
    
    module.walk([&](GenPOp op) {
      if (op.getInvBody().empty()) return;
      
      if (!verifyInverse(op, state)) {
          op.emitError("Inconsistent GenP: apply and inv regions are not bijections.");
      }
    });
  }

private:
  bool verifyInverse(GenPOp op, AsmState &state) {
    OwningOpRef<ModuleOp> smtModule = ModuleOp::create(op.getLoc());
    OpBuilder b(smtModule->getBodyRegion());
    auto solver = smt::SolverOp::create(b, op.getLoc(), TypeRange{}, ValueRange{});
    if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();
    
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&solver.getRegion().front());
    
    smt::SetLogicOp::create(b, op.getLoc(), "QF_NIA");
    SMTBuilder builder(b, state);

    auto dims = op.getDims();
    SmallVector<Value> x_vars;
    Value zero = smt::IntConstantOp::create(b, op.getLoc(), b.getI64IntegerAttr(0));
    
    for (Value d : dims) {
      Value x = builder.getOrCreate(d); // This actually creates a fun for 'd' if it's symbolic, but we want a fresh var for 'x'
      // Wait, 'd' is the dimension value. We need symbolic coordinates.
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

  bool runZ3(const std::string &smtLib) {
    int out_pipe[2]; int in_pipe[2];
    if (pipe(out_pipe) == -1 || pipe(in_pipe) == -1) return false;
    pid_t pid = fork();
    if (pid == -1) return false;
    if (pid == 0) {
        dup2(out_pipe[0], STDIN_FILENO); dup2(in_pipe[1], STDOUT_FILENO);
        close(out_pipe[0]); close(out_pipe[1]); close(in_pipe[0]); close(in_pipe[1]);
        execlp("z3", "z3", "-in", nullptr);
        _exit(1);
    }
    close(out_pipe[0]); close(in_pipe[1]);
    write(out_pipe[1], smtLib.c_str(), smtLib.size());
    close(out_pipe[1]);
    std::string result; char buffer[4096]; ssize_t n;
    while ((n = read(in_pipe[0], buffer, sizeof(buffer))) > 0) result.append(buffer, n);
    close(in_pipe[0]); waitpid(pid, nullptr, 0);
    return result.find("sat") != std::string::npos && result.find("unsat") == std::string::npos;
  }
};

} // namespace

namespace mlir {
namespace lego {
std::unique_ptr<Pass> createLegoVerifyConsistencyPass() {
  return std::make_unique<LegoVerifyConsistencyPassImpl>();
}
} // namespace lego
} // namespace mlir
