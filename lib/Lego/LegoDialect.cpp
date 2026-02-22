#include "Lego/LegoDialect.h"
#include "Lego/LegoOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::lego;

#define GET_TYPEDEF_CLASSES
#include "Lego/LegoOpsTypes.cpp.inc"

#include "Lego/LegoDialect.cpp.inc"

void LegoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Lego/LegoOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Lego/LegoOpsTypes.cpp.inc"
      >();
}
