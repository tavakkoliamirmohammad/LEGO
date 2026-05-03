//===- LegoArmNeonPipeline.cpp --------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → vector dialect (via
// lego-vectorize) → LLVM IR targeting ARM NEON.
//
// V1: lego-vectorize accepts `target=neon` and emits 16-byte (128-bit)
// NEON-width vectors directly: vector<2xf64> for f64, vector<4xf32> for f32.
// Width-aware SVE emission (scalable vectors) is deferred to R15.
//
// Pipeline:
//   1. buildLegoLowerPipeline   — shared front-end (LEGO → Arith)
//   2. canonicalize + CSE       — clean up before vectorization
//   3. lego-vectorize           — emit vector.transfer_read/write etc.
//   4. convert-vector-to-llvm   — lower vector dialect to LLVM IR
//   5. SCF → CF → Arith/MemRef/Func/CF → LLVM tail
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
  // R15: lego-vectorize now accepts a `target` option; this pipeline passes
  // "neon" to emit 16-byte (128-bit) NEON-width vectors:
  //   f64: 16/8 = 2 lanes → vector<2xf64>
  //   f32: 16/4 = 4 lanes → vector<4xf32>
  // On an x86 host, these NEON-width vectors are correct MLIR and can be
  // FileCheck-verified. For actual ARM execution, add mlir-translate → llc
  // with -mtriple=aarch64-linux-gnu -mattr=+neon.

  // Phase 1: shared front-end (LEGO → Arith + strength reduction).
  buildLegoLowerPipeline(pm);

  // Phase 2: clean up, then vectorize with NEON lane widths.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Pass target="neon" to emit NEON-width vectors (2xf64, 4xf32).
  pm.addNestedPass<mlir::func::FuncOp>(createLegoVectorizePass("neon"));

  // Phase 3: LLVM tail — lower vector dialect, then the rest of the dialects.
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
