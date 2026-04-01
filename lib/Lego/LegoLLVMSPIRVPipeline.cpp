/// LEGO LLVM SPIR-V pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → LLVM SPIR-V
///
/// Same three-phase pattern as lego-to-nvvm / lego-to-rocdl.
/// Phase 2 uses convert-gpu-to-llvm-spv to lower GPU ops to LLVM
/// dialect with SPIR-V calling conventions (e.g. _Z12get_local_idj).

#ifdef LEGO_HAS_SPIRV

#include "Lego/Passes.h"

#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Lowers gpu.all_reduce ops into shared memory + shuffle tree reduction.
struct LowerGpuAllReduceLLVMSPVPass
    : public PassWrapper<LowerGpuAllReduceLLVMSPVPass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuAllReduceLLVMSPVPass)

  StringRef getArgument() const override {
    return "lego-lower-gpu-all-reduce-llvmspv";
  }
  StringRef getDescription() const override {
    return "Lower gpu.all_reduce to shared memory + shuffle tree (LLVM SPV)";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGpuAllReducePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

/// Lowers gpu.subgroup_reduce to gpu.shuffle (butterfly pattern).
struct LowerGpuSubgroupReduceLLVMSPVPass
    : public PassWrapper<LowerGpuSubgroupReduceLLVMSPVPass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuSubgroupReduceLLVMSPVPass)

  StringRef getArgument() const override {
    return "lego-lower-subgroup-reduce-llvmspv";
  }
  StringRef getDescription() const override {
    return "Lower gpu.subgroup_reduce to gpu.shuffle (LLVM SPV)";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGpuLowerSubgroupReduceToShufflePatterns(patterns, 32, 32);
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(patterns, 32, 32);
    populateGpuBreakDownSubgroupReducePatterns(patterns, 32);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace lego {

void buildLegoToLLVMSPIRVPipeline(
    OpPassManager &pm, const LegoToLLVMSPIRVPipelineOptions &options) {
  // Phase 1: shared LEGO lower + GPU outline.
  buildLegoGPUOutlinePipeline(pm);

  // Phase 2: convert GPU ops to LLVM with SPIR-V calling conventions.
  ConvertGpuOpsToLLVMSPVOpsOptions spvOpts;
  spvOpts.use64bitIndex = true;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToLLVMSPVOps(spvOpts));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());

  // Phase 3: shared host-side LLVM lowering.
  buildGPUHostLLVMPipeline(pm);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_SPIRV
