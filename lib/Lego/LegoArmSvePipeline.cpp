//===- LegoArmSvePipeline.cpp ---------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → LLVM IR targeting
// ARM SVE.
//
// Pipeline:
//   1. buildLegoLowerPipeline   — shared front-end (LEGO → Arith)
//   2. canonicalize + CSE
//   3. convert-vector-to-llvm   — lower any vector dialect ops produced
//                                  upstream (linalg::vectorize is not run
//                                  on the SVE pipeline; non-affine loops
//                                  fall through as scalar scf.for)
//   4. SCF → CF → Arith/MemRef/Func/CF → LLVM tail
//
// For ARM SVE execution:
//   mlir-translate --mlir-to-llvmir out.mlir -o out.ll
//   llc -mtriple=aarch64-linux-gnu -mattr=+sve -O3 out.ll -filetype=obj -o out.o
//
//===----------------------------------------------------------------------===//

#include "Lego/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace lego {

void buildLegoToArmSvePipeline(OpPassManager &pm,
                                const LegoToArmSvePipelineOptions &opts) {
  // Phase 1: shared front-end (LEGO → Arith + strength reduction).
  buildLegoLowerPipeline(pm);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // LLVM tail.
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace lego
} // namespace mlir
