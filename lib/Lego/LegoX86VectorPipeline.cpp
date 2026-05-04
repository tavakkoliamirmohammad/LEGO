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

  // Phase 2: clean up, then vectorize. lego-vectorize emits vector dialect
  // ops (vector.transfer_read/write, vector.broadcast, arith ops on vectors).
  // Note: lego-vectorize is a func.func-level pass; nest inside func.func.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Linalg path (opt-in): raise affine scf.for loops to linalg.generic and
  // call upstream linalg::vectorize on them. Loops that fail the affine
  // check (Z-Morton et al) survive as scf.for and the custom
  // lego-vectorize below handles them — strict per-loop discriminator,
  // not a fallback layer.
  if (opts.useLinalgVectorize) {
    // Raise affine scf.for loops to linalg.generic, then tile by SIMD width
    // and call upstream linalg::vectorize on the inner tile. Loops that
    // failed the affine check survive as scf.for and the lego-vectorize
    // pass below handles them.
    pm.addNestedPass<mlir::func::FuncOp>(
        createConvertLegoToLinalgPass(/*vectorize=*/true));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // R15: pass the target to lego-vectorize so lane widths are ISA-specific.
  // The x86 pipeline uses "avx512" by default (64-byte ZMM registers).
  // LLVM splits <16xf32>/<8xf64> ops to 256-bit pairs automatically on
  // AVX2-only hosts — so this is correct-by-splitting even on AVX2 CPUs.
  // Recognise compaction (stream-filter) loops first.  These have a shape
  // (single index iter_arg + scf.if + counter increment) that the main
  // lego-vectorize can't easily handle but is a textbook clang-miss:
  // clang scalarises entirely; LEGO emits ``vector.compressstore``.
  pm.addNestedPass<mlir::func::FuncOp>(
      createLegoVectorizeCompactPass("avx512"));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // NOTE: lego-vectorize-scatter-add (count[bin[i]] += A[i]) is implemented
  // (LegoVectorizeScatterAdd.cpp) but NOT wired by default because gather/
  // scatter throughput is hardware-dependent: on AMD Zen 4 ``vpscatterdps``
  // is microcoded at ~80 cycles per 16-lane scatter, so the vectorised path
  // loses to clang's unrolled scalar.  On Intel Sapphire Rapids gather/
  // scatter are competitive and the pass should help.  Run it explicitly
  // via ``--lego-vectorize-scatter-add`` to opt in on Intel hardware.

  // Recognise argmin / argmax loops: paired (val, idx) reduction with
  // data-dependent updates that clang scalarises.  Emits vector
  // accumulators + arith.minimumf + arith.select inside the loop, and a
  // single vector.reduction pair after.
  pm.addNestedPass<mlir::func::FuncOp>(
      createLegoVectorizeArgminPass("avx512"));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Recognise inclusive prefix-scan loops (cumulative sum).  Loop-carried
  // dependency defeats clang's auto-vec; LEGO emits a Hillis-Steele
  // in-vector prefix sum + scalar carry.
  pm.addNestedPass<mlir::func::FuncOp>(
      createLegoVectorizeScanPass("avx512"));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Recognise filtered (predicated) reductions: clang scalarises the
  // data-dependent branch.  LEGO emits a vector accumulator + per-chunk
  // arith.cmpf + arith.select(mask, v, identity) + combine; one
  // vector.reduction at the end.
  pm.addNestedPass<mlir::func::FuncOp>(
      createLegoVectorizeFilteredReducePass("avx512"));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<mlir::func::FuncOp>(createLegoVectorizePass("avx512"));

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
  // math.ctpop and friends (emitted by lego-vectorize-compact for popcount)
  // require math→llvm conversion before the ExecutionEngine.
  pm.addPass(createConvertMathToLLVMPass());
  // Expand strided memref metadata (memref.subview, memref.reinterpret_cast)
  // BEFORE the final memref→LLVM conversion. Linalg tiling produces these
  // ops on strided memref types; the canonical memref-to-llvm pass cannot
  // lower them directly, so we expand them into base/offset/stride
  // components first.
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
