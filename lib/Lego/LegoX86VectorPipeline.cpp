//===- LegoX86VectorPipeline.cpp ------------------------------------------===//
//
// End-to-end MLIR pipeline that lowers Lego dialect → LLVM IR with x86 vector
// intrinsics (AVX-512 / AVX2).
//
// Pipeline:
//   1. buildLegoLowerPipeline   — shared front-end (LEGO → Arith)
//   2. canonicalize + CSE
//   3. convert-lego-to-linalg + upstream linalg::vectorize for affine loops
//   4. convert-vector-to-llvm
//   5. SCF → CF → Arith/MemRef/Func/CF → LLVM tail
//
// Loops that fall through (non-affine, e.g. Z-Morton) reach the LLVM tail
// as scf.for + arith + memref.  LLVM's LoopVectorize / SLP at opt_level=3
// handles them with the same target features clang -O3 -march=native sees
// (host-detected via JITTargetMachineBuilder::detectHost in MLIR's
// ExecutionEngine).
//
//===----------------------------------------------------------------------===//

#include "Lego/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
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

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Affine loops: raise to linalg.generic and call upstream linalg::vectorize.
  // Non-affine loops (Z-Morton et al) survive as scf.for and pass through to
  // the LLVM tail unchanged.
  if (opts.useLinalgVectorize) {
    pm.addNestedPass<mlir::func::FuncOp>(
        createConvertLegoToLinalgPass(/*vectorize=*/true));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // Phase 3: LLVM tail.
  //
  // reassociate-fp-reductions (opts.reassocFP, default true):
  //   Allows LLVM to reorder FP reductions across SIMD lanes — equivalent to
  //   gcc's -fassociative-math.  Off-by-default would force ordered (sequential)
  //   reductions; the lego-to-linalg + linalg::vectorize path emits
  //   vector.reduction ops that benefit from this flag.
  //
  // use-vector-alignment (opts.useVecAlignment, default false):
  //   Emits aligned load/store hints (64-byte for AVX-512).  Default false
  //   because NumPy/ctypes allocations are only 16-byte aligned; enabling on
  //   misaligned buffers causes SIGSEGV.
  ConvertVectorToLLVMPassOptions vecToLLVMOpts;
  vecToLLVMOpts.reassociateFPReductions = opts.reassocFP;
  vecToLLVMOpts.useVectorAlignment = opts.useVecAlignment;
  pm.addPass(createConvertVectorToLLVMPass(vecToLLVMOpts));
  pm.addPass(createConvertMathToLLVMPass());
  // Expand strided memref metadata before the final memref→LLVM conversion.
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace lego
} // namespace mlir
