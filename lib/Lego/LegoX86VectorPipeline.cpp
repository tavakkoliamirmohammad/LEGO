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

void buildLegoToX86VectorPipeline(OpPassManager &pm,
                                  const LegoToX86VectorPipelineOptions &opts) {
  // Phase 1: shared front-end (LEGO → Arith + strength reduction).
  buildLegoLowerPipeline(pm);

  // Phase 2: clean up, then vectorize. lego-vectorize emits vector dialect
  // ops (vector.transfer_read/write, vector.broadcast, arith ops on vectors).
  // Note: lego-vectorize is a func.func-level pass; nest inside func.func.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(createLegoVectorizePass());

  // Phase 3: LLVM tail — lower vector dialect, then the rest of the dialects.
  // convert-vector-to-llvm must precede SCF→CF so that the vector loop body
  // (produced by lego-vectorize) still exists as scf.for at this point.
  //
  // reassociate-fp-reductions (opts.reassocFP, default true):
  //   Allows LLVM to reorder floating-point reductions across SIMD lanes.
  //   Equivalent to GCC's -ffast-math reassoc.  The v1 default is *true* because:
  //   (a) benchmark correctness checks use rtol/atol tolerance, not exact equality;
  //   (b) GCC -O3 -march=native also reassociates, so this is the fair comparison.
  //   Users requiring strict IEEE-754 reproducibility should pass reassoc-fp=false.
  //   Reference: GCC manual §3.10 (-ffast-math); LLVM VectorToLLVM pass docs.
  //
  // use-vector-alignment (opts.useVecAlignment, default false):
  //   When true, emits 64-byte aligned load/store hints for AVX-512 transfers,
  //   which can yield ~30% additional throughput on fully-aligned buffers.
  //   Default false because NumPy/ctypes allocations are only 16-byte aligned;
  //   enabling on misaligned buffers causes SIGSEGV.  Users with posix_memalign
  //   or __attribute__((aligned(64))) buffers may safely set this true.
  ConvertVectorToLLVMPassOptions vecToLLVMOpts;
  vecToLLVMOpts.reassociateFPReductions = opts.reassocFP;
  vecToLLVMOpts.useVectorAlignment = opts.useVecAlignment;
  pm.addPass(createConvertVectorToLLVMPass(vecToLLVMOpts));
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace lego
} // namespace mlir
