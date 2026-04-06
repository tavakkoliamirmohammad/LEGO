//===- Dialects.cpp - C API for LEGO dialect registration -----------------===//
//
// Implements the C API for registering the LEGO MLIR dialect, passes, and
// LLVM translation interfaces.
//
// Pass registration strategy:
//   - Individual LEGO passes: registered (users run them standalone)
//   - Named pipelines: registered (lego-lower, lego-to-llvm, lego-to-spirv, lego-to-llvmspirv, lego-to-nvvm, lego-to-rocdl)
//   - Utility passes (canonicalize, cse): registered (used standalone)
//   - Arith int-range passes: registered (used in lego-lower fixed-point loop)
//   - LLVM/SPIR-V/GPU conversion passes: NOT registered — only used internally
//     by pipelines via createXxxPass(). No need for registry lookup.
//
//===----------------------------------------------------------------------===//

#include "Lego/CAPI/Dialects.h"
#include "Lego/LegoDialect.h"
#include "Lego/Passes.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#ifdef LEGO_HAS_NVPTX
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#endif
#ifdef LEGO_HAS_AMDGPU
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#endif
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

// LEGO dialect
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Lego, lego, mlir::lego::LegoDialect)

// --- LEGO type accessors ---

bool mlirLegoTypeIsALayout(MlirType type) {
  return mlir::isa<mlir::lego::LayoutType>(unwrap(type));
}

bool mlirLegoTypeIsAView(MlirType type) {
  return mlir::isa<mlir::lego::ViewType>(unwrap(type));
}

MlirTypeID mlirLegoLayoutTypeGetTypeID() {
  return wrap(mlir::lego::LayoutType::getTypeID());
}

MlirTypeID mlirLegoViewTypeGetTypeID() {
  return wrap(mlir::lego::ViewType::getTypeID());
}

MlirType mlirLegoLayoutTypeGet(MlirContext context) {
  return wrap(mlir::lego::LayoutType::get(unwrap(context)));
}

MlirType mlirLegoViewTypeGet(MlirContext context, MlirType elementType) {
  return wrap(
      mlir::lego::ViewType::get(unwrap(context), unwrap(elementType)));
}

MlirType mlirLegoViewTypeGetElementType(MlirType type) {
  return wrap(
      mlir::cast<mlir::lego::ViewType>(unwrap(type)).getElementType());
}

// Standard dialects used by the LEGO compiler
#include "mlir-c/Dialect/Arith.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir-c/Dialect/MemRef.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arith, arith, mlir::arith::ArithDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SCF, scf, mlir::scf::SCFDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MemRef, memref, mlir::memref::MemRefDialect)

// GPU + SPIR-V dialects (needed for lego-to-spirv pipeline, no GPU hardware)
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, mlir::gpu::GPUDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SPIRV, spirv, mlir::spirv::SPIRVDialect)

// Math dialect (needed for math.exp etc. in GPU kernels)
#include "mlir/Dialect/Math/IR/Math.h"
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Math, math, mlir::math::MathDialect)

void legoRegisterPasses() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;

  // Standard MLIR utility passes (used standalone and in pipelines)
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  // Arith int-range passes (used standalone in lit tests and in lego-lower)
  mlir::arith::registerArithPasses();

  // Individual LEGO passes (users may run standalone)
  mlir::lego::registerLegoToArithPass();
  mlir::lego::registerLegoNormalizationPass();
  mlir::lego::registerLegoArithSimplificationPass();
  mlir::lego::registerLegoGenerateBoundsChecksPass();
  mlir::lego::registerLegoExternalSMTVerifierPass();
  mlir::lego::registerLegoVerifyBijectivityPass();
  mlir::lego::registerLegoVerifyCoalescingPass();
  mlir::lego::registerLegoVerifyBankConflictsPass();
  mlir::lego::registerLegoStrengthReductionPass();

  // Named pipelines: "lego-lower", "lego-to-llvm", "lego-to-spirv", "lego-to-llvmspirv"
  // Internally these create LLVM/SPIR-V/GPU conversion passes via
  // createXxxPass() — no registration needed for those.
  mlir::lego::registerLegoPipelines();
}

void legoRegisterLLVMTranslations(MlirContext context) {
  mlir::registerBuiltinDialectTranslation(*unwrap(context));
  mlir::registerLLVMDialectTranslation(*unwrap(context));

#ifdef LEGO_HAS_NVPTX
  mlir::registerNVVMDialectTranslation(*unwrap(context));
#endif
#ifdef LEGO_HAS_AMDGPU
  mlir::registerROCDLDialectTranslation(*unwrap(context));
#endif
  mlir::registerGPUDialectTranslation(*unwrap(context));
}

void legoRegisterConvertToLLVMExtensions(MlirDialectRegistry registry) {
  mlir::DialectRegistry &reg = *unwrap(registry);
  mlir::arith::registerConvertArithToLLVMInterface(reg);
  mlir::cf::registerConvertControlFlowToLLVMInterface(reg);
  mlir::registerConvertFuncToLLVMInterface(reg);
  mlir::gpu::registerConvertGpuToLLVMInterface(reg);
  mlir::index::registerConvertIndexToLLVMInterface(reg);
  mlir::registerConvertMathToLLVMInterface(reg);
  mlir::registerConvertMemRefToLLVMInterface(reg);
}
