//===- LegoX86VectorPipeline.cpp ------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → vector dialect (via
// lego-vectorize) → LLVM IR with x86 vector intrinsics (AVX-512 / AVX2).
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

void buildLegoToX86VectorPipeline(OpPassManager &pm,
                                  const LegoToX86VectorPipelineOptions &opts) {
  // Phase 1: shared front-end (LEGO → Arith + strength reduction).
  buildLegoLowerPipeline(pm);

  // Phase 2: clean up, then vectorize. lego-vectorize emits vector dialect
  // ops (vector.transfer_read/write, vector.broadcast, arith ops on vectors).
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLegoVectorizePass());

  // Phase 3: LLVM tail — lower vector dialect, then the rest of the dialects.
  // convert-vector-to-llvm must precede SCF→CF so that the vector loop body
  // (produced by lego-vectorize) still exists as scf.for at this point.
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
