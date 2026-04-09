#include "Lego/SMTUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Target/SMTLIB/ExportSMTLIB.h"
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <signal.h>
#include <sstream>

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
    // Unsigned division: operands are non-negative in the SMT integer model,
    // so Euclidean div matches unsigned semantics when operands >= 0.
    valMap[v] = smt::IntDivOp::create(builder, loc, valMap[divOp.getLhs()], valMap[divOp.getRhs()]);
  } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(defOp)) {
    // Signed division truncates toward zero (C semantics), but SMT-LIB div
    // is Euclidean (rounds toward negative infinity). Must handle all four
    // sign combinations of (a, b):
    //   trunc_div(a, b) = sign(a)*sign(b) * (|a| div |b|)
    // Encoded as:
    //   absA = ite(a >= 0, a, -a),  absB = ite(b > 0, b, -b)
    //   posQuot = absA div absB
    //   result = ite((a >= 0) == (b > 0), posQuot, -posQuot)
    Value a = valMap[divSIOp.getLhs()];
    Value bVal = valMap[divSIOp.getRhs()];
    Value zero = smt::IntConstantOp::create(builder, loc, builder.getI64IntegerAttr(0));
    Value aGeZero = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge, a, zero);
    Value bGtZero = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::gt, bVal, zero);
    Value negA = smt::IntSubOp::create(builder, loc, zero, a);
    Value negB = smt::IntSubOp::create(builder, loc, zero, bVal);
    Value absA = smt::IteOp::create(builder, loc, aGeZero, a, negA);
    Value absB = smt::IteOp::create(builder, loc, bGtZero, bVal, negB);
    Value posQuot = smt::IntDivOp::create(builder, loc, absA, absB);
    Value negQuot = smt::IntSubOp::create(builder, loc, zero, posQuot);
    // Same sign => positive result, different sign => negative result
    Value sameSign = smt::EqOp::create(builder, loc, aGeZero, bGtZero);
    valMap[v] = smt::IteOp::create(builder, loc, sameSign, posQuot, negQuot);
  } else if (auto remOp = dyn_cast<arith::RemUIOp>(defOp)) {
    // Unsigned remainder: Euclidean mod matches when operands >= 0.
    valMap[v] = smt::IntModOp::create(builder, loc, valMap[remOp.getLhs()], valMap[remOp.getRhs()]);
  } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(defOp)) {
    // Signed remainder (C semantics): a - trunc_div(a, b) * b
    // trunc_div uses absolute values, same as DivSIOp above.
    Value a = valMap[remSIOp.getLhs()];
    Value bVal = valMap[remSIOp.getRhs()];
    Value zero = smt::IntConstantOp::create(builder, loc, builder.getI64IntegerAttr(0));
    Value aGeZero = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::ge, a, zero);
    Value bGtZero = smt::IntCmpOp::create(builder, loc, smt::IntPredicate::gt, bVal, zero);
    Value negA = smt::IntSubOp::create(builder, loc, zero, a);
    Value negB = smt::IntSubOp::create(builder, loc, zero, bVal);
    Value absA = smt::IteOp::create(builder, loc, aGeZero, a, negA);
    Value absB = smt::IteOp::create(builder, loc, bGtZero, bVal, negB);
    Value posQuot = smt::IntDivOp::create(builder, loc, absA, absB);
    Value negQuot = smt::IntSubOp::create(builder, loc, zero, posQuot);
    Value sameSign = smt::EqOp::create(builder, loc, aGeZero, bGtZero);
    Value truncDiv = smt::IteOp::create(builder, loc, sameSign, posQuot, negQuot);
    Value prod = smt::IntMulOp::create(builder, loc, ValueRange{truncDiv, bVal});
    valMap[v] = smt::IntSubOp::create(builder, loc, a, prod);
  } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
    auto lhs = valMap[cmpOp.getLhs()];
    auto rhs = valMap[cmpOp.getRhs()];
    switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq: 
        valMap[v] = smt::EqOp::create(builder, loc, lhs, rhs); break;
      case arith::CmpIPredicate::ne: 
        valMap[v] = smt::DistinctOp::create(builder, loc, ValueRange{lhs, rhs}); break;
      // NOTE: Unsigned comparisons (ult, ule, ugt, uge) are mapped to the same
      // SMT integer predicates as signed ones. This is sound ONLY because all
      // layout index values are constrained to be non-negative (via dim > 0 and
      // idx >= 0 assertions in the verification passes). If negative values
      // could appear, unsigned comparisons would need bitvector encoding.
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
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    Value cond = valMap[selectOp.getCondition()];
    Value trueVal = valMap[selectOp.getTrueValue()];
    Value falseVal = valMap[selectOp.getFalseValue()];
    valMap[v] = smt::IteOp::create(builder, loc, cond, trueVal, falseVal);
  } else if (auto shliOp = dyn_cast<arith::ShLIOp>(defOp)) {
    // shli(a, b) = a * 2^b. Only supported for constant shifts.
    // Symbolic shifts fall through to the unrecognized-op warning below.
    if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(shliOp.getRhs().getDefiningOp())) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t shift = intAttr.getInt();
        Value a = valMap[shliOp.getLhs()];
        Value factor = smt::IntConstantOp::create(builder, loc, builder.getI64IntegerAttr(1LL << shift));
        valMap[v] = smt::IntMulOp::create(builder, loc, ValueRange{a, factor});
      }
    }
  } else if (auto shruiOp = dyn_cast<arith::ShRUIOp>(defOp)) {
    // shrui(a, b) = a div 2^b (unsigned, Euclidean div is correct for non-negative a)
    if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(shruiOp.getRhs().getDefiningOp())) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t shift = intAttr.getInt();
        Value divisor = smt::IntConstantOp::create(builder, loc, builder.getI64IntegerAttr(1LL << shift));
        valMap[v] = smt::IntDivOp::create(builder, loc, valMap[shruiOp.getLhs()], divisor);
      }
    }
  } else if (auto andiOp = dyn_cast<arith::AndIOp>(defOp)) {
    // andi(a, mask) where mask = 2^k - 1 is equivalent to a mod 2^k
    bool handled = false;
    if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(andiOp.getRhs().getDefiningOp())) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t mask = intAttr.getInt();
        if (mask > 0 && ((mask + 1) & mask) == 0) {
          // mask is 2^k - 1
          Value modulus = smt::IntConstantOp::create(builder, loc, builder.getI64IntegerAttr(mask + 1));
          valMap[v] = smt::IntModOp::create(builder, loc, valMap[andiOp.getLhs()], modulus);
          handled = true;
        } else {
          andiOp->emitWarning("SMT encoding: arith.andi with non-power-of-2 mask (0x")
              << llvm::Twine::utohexstr(mask)
              << ") cannot be encoded precisely — replaced with unconstrained variable. "
                 "Verification results may be unsound.";
        }
      }
    }
    if (!handled) {
      // Non-constant or non-power-of-2 mask: fall through to generic warning
    }
  } else if (auto truncOp = dyn_cast<arith::TruncIOp>(defOp)) {
    // trunci %x : iN to iM => x mod 2^M (wrapping semantics)
    Value inner = getOrCreate(truncOp.getIn());
    unsigned destWidth = truncOp.getType().getIntOrFloatBitWidth();
    int64_t modulus = 1LL << destWidth;
    Value mod = smt::IntConstantOp::create(builder, loc, builder.getI64IntegerAttr(modulus));
    valMap[v] = smt::IntModOp::create(builder, loc, inner, mod);
  } else if (isa<arith::IndexCastOp, arith::ExtUIOp, arith::ExtSIOp>(defOp)) {
    // IndexCast and zero/sign extension: identity in unbounded integer model.
    // This is safe for ExtUIOp (value is non-negative). ExtSIOp is also safe
    // since unbounded integers already represent the full signed range.
    valMap[v] = getOrCreate(defOp->getOperand(0));
  }

  if (!valMap.count(v)) {
    // Emit a warning for unrecognized ops instead of silently creating
    // unconstrained variables, which would make verification unsound.
    if (defOp) {
      defOp->emitWarning("SMT encoding: unsupported operation '")
          << defOp->getName()
          << "' — replaced with unconstrained symbolic variable. "
             "Verification results may be unsound.";
    }
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

SMTResult SMTSolverContext::checkSatisfiability(const SmallVector<std::string> &varNamesToExtract,
                                                unsigned timeoutMs) {
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
    smtLib.erase(resetPos, 7); // Remove "(reset)" (7 chars)
    // Also remove trailing newline if present
    if (resetPos < smtLib.size() && smtLib[resetPos] == '\n')
      smtLib.erase(resetPos, 1);
  }

  smtLib += generateGetValueCommands(varNamesToExtract);
  return runZ3WithModel(smtLib, timeoutMs);
}

bool runZ3(const std::string &smtLib) {
  SMTResult result = runZ3WithModel(smtLib);
  return result.isSat;
}

SMTResult runZ3WithModel(const std::string &smtLib, unsigned timeoutMs) {
  SMTResult result;

#ifdef __EMSCRIPTEN__
  // Z3 process spawning is not available in WebAssembly.
  result.isUnknown = true;
  return result;
#else
  // Prepend a timeout to prevent Z3 from hanging on hard QF_NIA instances.
  std::string smtLibWithTimeout;
  if (timeoutMs > 0)
    smtLibWithTimeout = "(set-option :timeout " + std::to_string(timeoutMs) + ")\n" + smtLib;
  else
    smtLibWithTimeout = smtLib;

  int out_pipe[2]; int in_pipe[2];
  if (pipe(out_pipe) == -1) {
    result.isUnknown = true;
    return result;
  }
  if (pipe(in_pipe) == -1) {
    close(out_pipe[0]); close(out_pipe[1]);
    result.isUnknown = true;
    return result;
  }

  pid_t pid = fork();
  if (pid == -1) {
    close(out_pipe[0]); close(out_pipe[1]);
    close(in_pipe[0]); close(in_pipe[1]);
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

  // Write SMT-LIB to Z3 stdin with retry on short writes and EINTR.
  const char *writePtr = smtLibWithTimeout.c_str();
  size_t remaining = smtLibWithTimeout.size();
  bool writeOk = true;
  while (remaining > 0) {
    ssize_t written = write(out_pipe[1], writePtr, remaining);
    if (written > 0) {
      writePtr += written;
      remaining -= written;
    } else if (written == -1 && errno == EINTR) {
      continue;
    } else {
      writeOk = false;
      break;
    }
  }
  close(out_pipe[1]);

  if (!writeOk) {
    close(in_pipe[0]);
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    result.isUnknown = true;
    return result;
  }

  std::string output;
  char buffer[4096];
  ssize_t n;
  while ((n = read(in_pipe[0], buffer, sizeof(buffer))) > 0) {
    output.append(buffer, n);
  }
  close(in_pipe[0]);

  int wstatus = 0;
  waitpid(pid, &wstatus, 0);

  result.rawOutput = output;

  // Check if Z3 exited abnormally (crash, missing binary, signal)
  if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
    if (output.empty()) {
      // Z3 likely crashed or was not found — treat as unknown with note
      result.isUnknown = true;
      return result;
    }
    // Non-zero exit but has output — Z3 may have printed sat/unsat before error,
    // fall through to normal parsing.
  }

  // Parse the result — match on line boundaries to avoid false matches
  // from variable names containing "sat" or "unsat".
  bool foundUnsat = false;
  bool foundSat = false;
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    // Trim whitespace
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) continue;
    std::string trimmed = line.substr(start);
    if (trimmed == "unsat") { foundUnsat = true; break; }
    if (trimmed == "sat") { foundSat = true; break; }
    if (trimmed == "unknown") { result.isUnknown = true; return result; }
  }

  if (foundUnsat) {
    result.isUnsat = true;
  } else if (foundSat) {
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
