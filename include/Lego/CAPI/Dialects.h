//===- Dialects.h - C API for LEGO dialect registration ---------*- C++ -*-===//
//
// C API header for the LEGO MLIR dialect.
// Provides dialect handle for registration from Python bindings.
//
//===----------------------------------------------------------------------===//

#ifndef LEGO_CAPI_DIALECTS_H
#define LEGO_CAPI_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Lego, lego);

/// Register all LEGO passes and pipelines with the global pass registry.
/// Idempotent — safe to call multiple times.
MLIR_CAPI_EXPORTED void legoRegisterPasses(void);

/// Register LLVM IR translation interfaces on the given context.
/// Required before using mlir::ExecutionEngine for JIT compilation.
MLIR_CAPI_EXPORTED void legoRegisterLLVMTranslations(MlirContext context);

// --- Type accessors for Python bindings ---

MLIR_CAPI_EXPORTED bool mlirLegoTypeIsALayout(MlirType type);
MLIR_CAPI_EXPORTED bool mlirLegoTypeIsAView(MlirType type);

MLIR_CAPI_EXPORTED MlirTypeID mlirLegoLayoutTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirTypeID mlirLegoViewTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirLegoLayoutTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirType mlirLegoViewTypeGet(MlirContext context,
                                                 MlirType elementType);
MLIR_CAPI_EXPORTED MlirType mlirLegoViewTypeGetElementType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // LEGO_CAPI_DIALECTS_H
