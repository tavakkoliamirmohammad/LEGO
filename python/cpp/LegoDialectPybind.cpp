//===- LegoDialectPybind.cpp - nanobind LEGO dialect registration ---------===//
//
// Registers the LEGO dialect, required standard dialects, LEGO passes, and
// LLVM translation interfaces with MLIR's Python context using nanobind.
//
// Only uses C API functions — no C++ MLIR headers needed.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/IR.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/MemRef.h"
#ifdef LEGO_HAS_NVPTX
#include "mlir-c/Dialect/GPU.h"
#endif
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "Lego/CAPI/Dialects.h"

namespace nb = nanobind;

static void registerAndLoad(MlirDialectHandle handle, MlirContext ctx) {
  mlirDialectHandleRegisterDialect(handle, ctx);
  mlirDialectHandleLoadDialect(handle, ctx);
}

NB_MODULE(_legoDialects, m) {
  m.doc() = "LEGO MLIR dialect Python bindings";

  m.def(
      "register_lego_dialect",
      [](MlirContext ctx) {
        // Register LEGO passes and pipelines (idempotent)
        legoRegisterPasses();

        // Register LLVM IR translations (needed for ExecutionEngine)
        legoRegisterLLVMTranslations(ctx);

        // Register and load the LEGO dialect
        registerAndLoad(mlirGetDialectHandle__lego__(), ctx);

        // Register standard dialects used by the LEGO compiler
        registerAndLoad(mlirGetDialectHandle__arith__(), ctx);
        registerAndLoad(mlirGetDialectHandle__func__(), ctx);
        registerAndLoad(mlirGetDialectHandle__scf__(), ctx);
        registerAndLoad(mlirGetDialectHandle__memref__(), ctx);
#ifdef LEGO_HAS_NVPTX
        registerAndLoad(mlirGetDialectHandle__gpu__(), ctx);
#endif
      },
      nb::arg("context"),
      "Register and load the LEGO dialect and required standard dialects.");
}
