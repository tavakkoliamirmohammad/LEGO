#ifndef LEGO_SMTUTILS_H
#define LEGO_SMTUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <string>
#include <optional>

namespace mlir {
namespace lego {

struct SMTBuilder {
  OpBuilder &builder;
  DenseMap<Value, Value> valMap;
  AsmState &state;
  unsigned &nextId;

  SMTBuilder(OpBuilder &b, AsmState &s, unsigned &nextId)
    : builder(b), state(s), nextId(nextId) {}

  std::string getSSAName(Value v);
  Value getOrCreate(Value v);
  void buildRegion(Region &region, ValueRange args, SmallVectorImpl<Value> &results);
};

// Result of an SMT query
struct SMTResult {
  bool isSat;
  bool isUnsat;
  bool isUnknown;
  llvm::StringMap<int64_t> model; // Counter-example model if SAT
  std::string rawOutput;

  SMTResult() : isSat(false), isUnsat(false), isUnknown(false) {}
};

// Run Z3 and return detailed result
SMTResult runZ3WithModel(const std::string &smtLib);

// Legacy interface (returns true if SAT)
bool runZ3(const std::string &smtLib);

// Helper to generate SMT-LIB get-value commands for a list of variables
std::string generateGetValueCommands(const SmallVector<std::string> &varNames);

} // namespace lego
} // namespace mlir

#endif // LEGO_SMTUTILS_H
