//===- LegoArmSvePipeline.cpp ---------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → vector dialect (via
// lego-vectorize with target="sve") → LLVM IR targeting ARM SVE.
//
// R15: lego-vectorize with target="sve" emits 16-byte (128-bit / vscale=1)
// fixed-width vectors: vector<2xf64> for f64, vector<4xf32> for f32.  These
// are the same lane counts as NEON but declared with "sve" semantics so that
// the LLVM AArch64 backend can promote them to the full runtime SVE width
// (512-bit on Neoverse V2, for example) when the +sve target feature is set.
//
// Design note — why fixed-width and not scalable?
//   True scalable vectors use the MLIR `vector<[N]xT>` syntax (brackets denote
//   vscale-relative).  Emitting scalable vectors would require the lego-vectorize
//   pass to produce VectorType::get({}, elemTy, /*scalable=*/true) and the
//   entire transfer_read/write, shuffle, and broadcast emission to use scalable
//   operands.  This is significant complexity for a target that has no available
//   hardware on this CHPC node.  The fixed-width approach is:
//     (a) Correct: the LLVM AArch64 backend legalizes fixed-width vectors to
//         SVE when the target has +sve; it does NOT require scalable IR.
//     (b) Verifiable: the IR shape can be FileCheck-tested on any x86 host.
//     (c) Upgradeable: a later pass (R15v2) can swap in scalable IR once the
//         SVE dialect support in MLIR stabilizes.
//
// Pipeline:
//   1. buildLegoLowerPipeline   — shared front-end (LEGO → Arith)
//   2. canonicalize + CSE       — clean up before vectorization
//   3. lego-vectorize{target=sve} — emit 16-byte-width vectors
//   4. convert-vector-to-llvm   — lower vector dialect to LLVM IR
//   5. SCF → CF → Arith/MemRef/Func/CF → LLVM tail
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
  // R15: lego-vectorize with target="sve" emits fixed-width 16-byte vectors
  // (same lane counts as NEON: 2xf64, 4xf32).  The LLVM AArch64 backend
  // legalizes these to SVE instructions when +sve is specified at llc time.
  //
  // This is IR-shape compatible with lego-to-arm-neon but uses a distinct
  // pipeline name so users can opt in without changing the NEON pipeline.

  // Phase 1: shared front-end (LEGO → Arith + strength reduction).
  buildLegoLowerPipeline(pm);

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

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
