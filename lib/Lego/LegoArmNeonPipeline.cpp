//===- LegoArmNeonPipeline.cpp --------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → vector dialect (via
// lego-vectorize) → LLVM IR targeting ARM NEON.
//
// V1 note: lego-vectorize currently emits AVX-512-width vectors (8xf64 for
// f64 workloads).  When compiled for an aarch64 target the LLVM AArch64
// backend will split these to NEON-width (2xf64) automatically.  Proper
// NEON-width vector selection (2 f64 lanes, 4 f32 lanes) is captured as
// future-work item R15 and will be addressed in a follow-up task.
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
  // Phase 1: shared front-end (LEGO → Arith + strength reduction).
  buildLegoLowerPipeline(pm);

  // Phase 2: clean up, then vectorize. lego-vectorize emits vector dialect
  // ops.  At v1 the vector width is AVX-512 default; the LLVM AArch64 backend
  // will split <8xf64> to NEON-width <2xf64> pairs automatically.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLegoVectorizePass());

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
