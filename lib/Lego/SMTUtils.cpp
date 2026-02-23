#include "Lego/SMTUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace mlir {
namespace lego {

std::string SMTBuilder::getSSAName(Value v) {
  return "v" + std::to_string(nextId++);
}

Value SMTBuilder::getOrCreate(Value v) {
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

void SMTBuilder::buildRegion(Region &region, ValueRange args, SmallVectorImpl<Value> &results) {
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

} // namespace lego
} // namespace mlir
