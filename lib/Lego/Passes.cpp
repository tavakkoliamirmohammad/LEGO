#include "Lego/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

/// Runs arith-simplification + int-range + canonicalize + CSE in a loop
/// until the IR stops changing (fixed-point) or a max iteration count.
struct FixedPointSimplificationPass
    : public PassWrapper<FixedPointSimplificationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixedPointSimplificationPass)

  StringRef getArgument() const override {
    return "lego-fixed-point-simplification";
  }
  StringRef getDescription() const override {
    return "Run arith simplification + int-range to fixed point";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (;;) {
      // Fingerprint the IR before.
      OperationFingerPrint before(module);

      // Run the sub-pipeline.
      PassManager subPM(module.getContext());
      subPM.addPass(createLegoArithSimplificationPass());
      subPM.addPass(arith::createIntRangeOptimizationsPass());
      subPM.addPass(createCanonicalizerPass());
      subPM.addPass(createCSEPass());
      if (failed(subPM.run(module))) {
        signalPassFailure();
        return;
      }

      // Check if anything changed.
      OperationFingerPrint after(module);
      if (before == after)
        break;
    }
  }
};

} // namespace

namespace mlir {
namespace lego {

/// Populates the standard lego-lower pipeline (LEGO → Arith).
void buildLegoLowerPipeline(OpPassManager &pm) {
  pm.addPass(createLegoMaterializeAssumeBoundsPass());
  // Normalize Row/Col to RegP first so LegoToArith sees uniform ops.
  pm.addPass(createLegoNormalizationPass(/*skipTileBy=*/true));
  // Lower TileBy/Apply/ApplyInverse directly to arith.
  pm.addPass(createLegoToArithPass());
  // Normalize remaining TileBy→GroupBy for any ops not handled above.
  pm.addPass(createLegoNormalizationPass());
  // Lower any GroupBy ops produced by normalization.
  pm.addPass(createLegoToArithPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Run arith simplification + int-range to fixed point.
  pm.addPass(std::make_unique<FixedPointSimplificationPass>());
  // Remove remui wrappers from materialized assume bounds.
  pm.addPass(createLegoMaterializeAssumeBoundsPass(/*cleanup=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Strength-reduce power-of-2 divui/remui to shift/mask.
  // Runs after algebraic simplification to avoid interfering with
  // div/rem pattern matchers in the fixed-point loop.
  pm.addPass(createLegoStrengthReductionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

/// Populates the full lego-to-llvm pipeline (LEGO → Arith → LLVM).
void buildLegoToLLVMPipeline(OpPassManager &pm) {
  buildLegoLowerPipeline(pm);

  // Lower to LLVM dialect
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Clean up LLVM IR
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void buildLegoGPUOutlinePipeline(OpPassManager &pm) {
  buildLegoLowerPipeline(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createGpuKernelOutliningPass());
}

void buildGPUToLLVMAndBinaryPipeline(OpPassManager &pm, StringRef format) {
  // Lower arith/scf/cf/index/func everywhere (host + kernel).
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());

  // gpu-to-llvm converts host GPU runtime ops AND finishes memref type
  // conversion.  Must run while memref types still exist.
  pm.addPass(createGpuToLLVMConversionPass());

  // Finalize memref (converts remaining memref ops to LLVM).
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Compile gpu.module → binary (kernel is fully LLVM now).
  GpuModuleToBinaryPassOptions binOpts;
  binOpts.compilationTarget = std::string(format);
  pm.addPass(createGpuModuleToBinaryPass(binOpts));

  // Clean up any remaining casts.
  pm.addPass(createReconcileUnrealizedCastsPass());
}

void registerLegoPipelines() {
  PassPipelineRegistration<>("lego-lower",
    "Lego e2e lowering pipeline (LEGO -> Arith)",
    buildLegoLowerPipeline);

  PassPipelineRegistration<>("lego-to-llvm",
    "Full LEGO lowering to LLVM dialect (LEGO -> Arith -> LLVM)",
    buildLegoToLLVMPipeline);

  PassPipelineRegistration<LegoToSPIRVPipelineOptions>("lego-to-spirv",
    "Lower LEGO dialect through GPU to SPIR-V "
    "(LEGO -> Arith -> GPU outlined -> SPIR-V)",
    buildLegoToSPIRVPipeline);

#ifdef LEGO_HAS_NVPTX
  PassPipelineRegistration<LegoToNVVMPipelineOptions>("lego-to-nvvm",
    "Lower LEGO dialect through GPU to NVVM/CUDA "
    "(LEGO -> Arith -> GPU outlined -> NVVM -> PTX/cubin)",
    buildLegoToNVVMPipeline);
#endif

#ifdef LEGO_HAS_AMDGPU
  PassPipelineRegistration<LegoToROCDLPipelineOptions>("lego-to-rocdl",
    "Lower LEGO dialect through GPU to ROCDL/AMD "
    "(LEGO -> Arith -> GPU outlined -> ROCDL -> HSACO)",
    buildLegoToROCDLPipeline);
#endif
}

} // namespace lego
} // namespace mlir
