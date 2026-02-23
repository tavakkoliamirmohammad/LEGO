#ifndef LEGO_SMTUTILS_H
#define LEGO_SMTUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

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

bool runZ3(const std::string &smtLib);

} // namespace lego
} // namespace mlir

#endif // LEGO_SMTUTILS_H
