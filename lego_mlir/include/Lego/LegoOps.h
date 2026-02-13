#ifndef LEGO_OPS_H
#define LEGO_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "Lego/LegoDialect.h"

#define GET_OP_CLASSES
#include "Lego/LegoOps.h.inc"

#endif // LEGO_OPS_H
