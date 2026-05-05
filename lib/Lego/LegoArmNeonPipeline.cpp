//===- LegoArmNeonPipeline.cpp --------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → LLVM IR targeting
// ARM NEON.
//
// Pipeline:
//   1. buildLegoLowerPipeline   — shared front-end (LEGO → Arith)
//   2. canonicalize + CSE
//   3. convert-vector-to-llvm   — lower any vector dialect ops produced
//                                  upstream (linalg::vectorize is not run
//                                  on the ARM pipeline; non-affine loops
//                                  fall through to LLVM as scalar scf.for)
//   4. SCF → CF → Arith/MemRef/Func/CF → LLVM tail
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

void buildLegoToArmNeonPipeline(OpPassManager &pm,
                                const LegoToArmNeonPipelineOptions &opts) {
  // For actual ARM execution, run mlir-translate → llc with
  //   -mtriple=aarch64-linux-gnu -mattr=+neon

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
