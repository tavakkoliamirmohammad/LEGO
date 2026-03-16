#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "Lego/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#ifdef LEGO_HAS_NVPTX
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#endif
#include "mlir/Transforms/Passes.h"

#include "Lego/LegoDialect.h"
#include "Lego/LegoOps.h"
#include "mlir/Dialect/SMT/IR/SMTDialect.h"

// Needed for registerArithPasses
#include "mlir/Dialect/Arith/Transforms/Passes.h"

// Needed for lego-to-llvm pipeline conversion passes
#include "mlir/Conversion/Passes.h"

int main(int argc, char **argv) {
  // mlir::registerAllPasses();
  // Register minimal passes
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  // Register Arith passes (needed for int-range-optimizations)
  mlir::arith::registerArithPasses();
  // Register conversion passes (needed for lego-to-llvm pipeline)
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();
  // Register LEGO passes
  mlir::lego::registerLegoToArithPass();
  mlir::lego::registerLegoNormalizationPass();
  mlir::lego::registerLegoArithSimplificationPass();
  mlir::lego::registerLegoGenerateBoundsChecksPass();
  mlir::lego::registerLegoExternalSMTVerifierPass();
  mlir::lego::registerLegoVerifyBijectivityPass();
  mlir::lego::registerLegoVerifyCoalescingPass();
  mlir::lego::registerLegoVerifyBankConflictsPass();
  mlir::lego::registerLegoPipelines();

  mlir::DialectRegistry registry;
  // mlir::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::math::MathDialect>();
  registry.insert<mlir::lego::LegoDialect, mlir::smt::SMTDialect>();
  // Dialects needed for lego-to-llvm pipeline
  registry.insert<mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
#ifdef LEGO_HAS_NVPTX
  registry.insert<mlir::gpu::GPUDialect>();
#endif

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LEGO Magic Optimizer Driver", registry));
}
