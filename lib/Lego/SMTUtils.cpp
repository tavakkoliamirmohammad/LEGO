#include "Lego/SMTUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>

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

SMTSolverContext::SMTSolverContext(Location l, AsmState &state, unsigned &nextId) : loc(l) {
  smtModule = ModuleOp::create(loc);
  b = std::make_unique<OpBuilder>(smtModule->getBodyRegion());

  auto solver = smt::SolverOp::create(*b, loc, TypeRange{}, ValueRange{});
  if (solver.getRegion().empty()) solver.getRegion().emplaceBlock();

  b->setInsertionPointToStart(&solver.getRegion().front());
  smt::SetLogicOp::create(*b, loc, "QF_NIA");

  builder = std::make_unique<SMTBuilder>(*b, state, nextId);
}

SMTResult SMTSolverContext::checkSatisfiability(const SmallVector<std::string> &varNamesToExtract) {
  auto checkOp = smt::CheckOp::create(*b, loc, TypeRange{});
  for (Region &r : checkOp->getRegions()) {
    OpBuilder::InsertionGuard g(*b);
    b->setInsertionPointToStart(&r.emplaceBlock());
    smt::YieldOp::create(*b, loc, ValueRange{});
  }
  smt::YieldOp::create(*b, loc, ValueRange{});

  std::string smtLib;
  llvm::raw_string_ostream os(smtLib);
  if (failed(mlir::smt::exportSMTLIB(*smtModule, os))) {
    SMTResult failRes;
    failRes.isUnknown = true;
    return failRes;
  }

  size_t resetPos = smtLib.rfind("(reset)");
  if (resetPos != std::string::npos) {
    smtLib.erase(resetPos, 8); // Remove "(reset)\n"
  }

  smtLib += generateGetValueCommands(varNamesToExtract);
  return runZ3WithModel(smtLib);
}

bool runZ3(const std::string &smtLib) {
  SMTResult result = runZ3WithModel(smtLib);
  return result.isSat;
}

SMTResult runZ3WithModel(const std::string &smtLib) {
  SMTResult result;

#ifdef __EMSCRIPTEN__
  // Z3 process spawning is not available in WebAssembly.
  result.isUnknown = true;
  return result;
#else
  int out_pipe[2]; int in_pipe[2];
  if (pipe(out_pipe) == -1 || pipe(in_pipe) == -1) {
    result.isUnknown = true;
    return result;
  }

  pid_t pid = fork();
  if (pid == -1) {
    result.isUnknown = true;
    return result;
  }

  if (pid == 0) {
    dup2(out_pipe[0], STDIN_FILENO);
    dup2(in_pipe[1], STDOUT_FILENO);
    dup2(in_pipe[1], STDERR_FILENO);  // Capture stderr too
    close(out_pipe[0]); close(out_pipe[1]);
    close(in_pipe[0]); close(in_pipe[1]);
    execlp(LEGO_Z3_EXECUTABLE, "z3", "-in", nullptr);
    _exit(1);
  }

  close(out_pipe[0]);
  close(in_pipe[1]);

  write(out_pipe[1], smtLib.c_str(), smtLib.size());
  close(out_pipe[1]);

  std::string output;
  char buffer[4096];
  ssize_t n;
  while ((n = read(in_pipe[0], buffer, sizeof(buffer))) > 0) {
    output.append(buffer, n);
  }
  close(in_pipe[0]);
  waitpid(pid, nullptr, 0);

  result.rawOutput = output;

  // Debug: Print raw Z3 output (temporary)
  // fprintf(stderr, "=== Z3 RAW OUTPUT ===\n%s\n=== END Z3 OUTPUT ===\n", output.c_str());

  // Parse the result
  if (output.find("unsat") != std::string::npos) {
    result.isUnsat = true;
  } else if (output.find("sat") != std::string::npos &&
             output.find("unsat") == std::string::npos) {
    result.isSat = true;

    // Parse model from output
    // Z3 returns: sat\n((var1 val1)\n (var2 val2)\n ...)
    // Find the start of get-value response
    size_t valueStart = output.find("((");
    if (valueStart != std::string::npos) {
      size_t valueEnd = output.find("))", valueStart);
      if (valueEnd != std::string::npos) {
        std::string valuesBlock = output.substr(valueStart + 1, valueEnd - valueStart - 1);

        // Parse each (varname value) pair
        size_t pos = 0;
        while (pos < valuesBlock.length()) {
          // Find next '('
          pos = valuesBlock.find('(', pos);
          if (pos == std::string::npos) break;
          pos++; // Skip '('

          // Find matching ')'
          size_t endParen = valuesBlock.find(')', pos);
          if (endParen == std::string::npos) break;

          std::string pair = valuesBlock.substr(pos, endParen - pos);

          // Split by space to get varname and value
          size_t spacePos = pair.find(' ');
          if (spacePos != std::string::npos) {
            std::string varName = pair.substr(0, spacePos);
            std::string valueStr = pair.substr(spacePos + 1);

            // Trim whitespace
            varName.erase(0, varName.find_first_not_of(" \t\n\r"));
            varName.erase(varName.find_last_not_of(" \t\n\r") + 1);
            valueStr.erase(0, valueStr.find_first_not_of(" \t\n\r"));
            valueStr.erase(valueStr.find_last_not_of(" \t\n\r") + 1);

            // Handle negative numbers: (- 5)
            bool isNegative = false;
            if (valueStr.length() >= 2 && valueStr.substr(0, 2) == "(-") {
              isNegative = true;
              // Extract number from "(- number)"
              size_t numStart = valueStr.find(' ', 2);
              if (numStart != std::string::npos) {
                valueStr = valueStr.substr(numStart + 1);
                if (valueStr.back() == ')') {
                  valueStr.pop_back();
                }
              }
            }

            // Parse integer value
            char* endPtr = nullptr;
            errno = 0;
            long long value = std::strtoll(valueStr.c_str(), &endPtr, 10);
            if (errno == 0 && endPtr != valueStr.c_str() && !varName.empty()) {
              if (isNegative) value = -value;
              result.model[varName] = value;
              // fprintf(stderr, "  Parsed: %s = %lld\n", varName.c_str(), value);
            }
          }

          pos = endParen + 1;
        }
      }
    }
  } else {
    result.isUnknown = true;
  }

  return result;
#endif // !__EMSCRIPTEN__
}

std::string generateGetValueCommands(const SmallVector<std::string> &varNames) {
  if (varNames.empty()) return "";

  std::string commands = "(get-value (";
  for (size_t i = 0; i < varNames.size(); ++i) {
    if (i > 0) commands += " ";
    commands += varNames[i];
  }
  commands += "))\n";
  return commands;
}

} // namespace lego
} // namespace mlir
