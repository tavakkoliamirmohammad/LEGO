//===- Dialects.cpp - C API for LEGO dialect registration -----------------===//
//
// Implements the C API for registering the LEGO MLIR dialect, passes, and
// LLVM translation interfaces.
//
//===----------------------------------------------------------------------===//

#include "Lego/CAPI/Dialects.h"
#include "Lego/LegoDialect.h"
#include "Lego/Passes.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// LEGO dialect
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Lego, lego, mlir::lego::LegoDialect)

// Standard dialects used by the LEGO compiler — register them here so they
// are available via load_all_available_dialects() without pulling in the
// heavyweight MLIRCAPIRegisterEverything (which drags in GPU, SparseTensor, etc.).
#include "mlir-c/Dialect/Arith.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arith, arith, mlir::arith::ArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SCF, scf, mlir::scf::SCFDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MemRef, memref, mlir::memref::MemRefDialect)

void legoRegisterPasses() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;

  // Individual LEGO passes (from GEN_PASS_REGISTRATION in Passes.h.inc)
  mlir::lego::registerLegoToArithPass();
  mlir::lego::registerLegoNormalizationPass();
  mlir::lego::registerLegoArithSimplificationPass();
  mlir::lego::registerLegoGenerateBoundsChecksPass();
  mlir::lego::registerLegoExternalSMTVerifierPass();
  mlir::lego::registerLegoVerifyBijectivityPass();
  mlir::lego::registerLegoVerifyCoalescingPass();
  mlir::lego::registerLegoVerifyBankConflictsPass();

  // Named pipelines: "lego-lower" and "lego-to-llvm"
  mlir::lego::registerLegoPipelines();
}

void legoRegisterLLVMTranslations(MlirContext context) {
  mlir::registerBuiltinDialectTranslation(*unwrap(context));
  mlir::registerLLVMDialectTranslation(*unwrap(context));
}
