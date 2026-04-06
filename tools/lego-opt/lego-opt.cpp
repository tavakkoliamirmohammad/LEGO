#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "Lego/LegoDialect.h"
#include "Lego/CAPI/Dialects.h"

int main(int argc, char **argv) {
  // Register all passes and pipelines via the shared CAPI function.
  // This is the single source of truth — both lego-opt and the Python
  // bindings call the same function, preventing registration drift.
  legoRegisterPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::math::MathDialect>();
  registry.insert<mlir::lego::LegoDialect, mlir::smt::SMTDialect>();
  registry.insert<mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::gpu::GPUDialect, mlir::spirv::SPIRVDialect>();

  // Register ConvertToLLVMPatternInterface extensions for all dialects.
  // Required by GPU→ROCDL/NVVM passes to discover LLVM conversion patterns
  // for memref, arith, cf, func, etc. inside GPU kernels.
  mlir::registerConvertToLLVMDependentDialectLoading(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LEGO Magic Optimizer Driver", registry));
}
