/// LEGO NVVM/CUDA pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → NVVM → PTX/cubin

#include "Lego/Passes.h"

// NVVM conversion headers
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
// Standard lowering
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

// ============================================================================
// SetNVVMTargetPass — stamps #nvvm.target on gpu.module ops
// ============================================================================

struct SetNVVMTargetPass
    : public PassWrapper<SetNVVMTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetNVVMTargetPass)

  std::string chip;
  std::string features;
  int optLevel;

  SetNVVMTargetPass(StringRef chip = "sm_70", StringRef features = "+ptx60",
                    int optLevel = 2)
      : chip(chip.str()), features(features.str()), optLevel(optLevel) {}

  StringRef getArgument() const override { return "lego-set-nvvm-target"; }
  StringRef getDescription() const override {
    return "Set NVVM compilation target on gpu.module ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registerNVVMDialectTranslation(registry);
    registerGPUDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registerBuiltinDialectTranslation(registry);
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    NVVM::registerNVVMTargetInterfaceExternalModels(*ctx);

    auto nvvmTarget = NVVM::NVVMTargetAttr::get(
        ctx, optLevel, "nvptx64-nvidia-cuda", chip, features);
    getOperation()->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod.setTargetsAttr(ArrayAttr::get(ctx, {nvvmTarget}));
    });
  }
};

} // namespace

// ============================================================================
// Pipeline definition
// ============================================================================

namespace mlir {
namespace lego {

void buildLegoToNVVMPipeline(OpPassManager &pm,
                              const LegoToNVVMPipelineOptions &options) {
  // Step 1: Lower LEGO ops to arith.
  buildLegoLowerPipeline(pm);

  // Step 2: Fold constants before outlining.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Step 3: Outline gpu.launch → gpu.module + gpu.func.
  pm.addPass(createGpuKernelOutliningPass());

  // Step 4: Set #nvvm.target on gpu.module ops.
  pm.addPass(std::make_unique<SetNVVMTargetPass>(
      options.chip, options.features, options.optLevel));

  // Step 5: Lower GPU ops to NVVM dialect inside gpu.module.
  pm.addNestedPass<gpu::GPUModuleOp>(::mlir::createConvertGpuOpsToNVVMOps());

  // Step 6: Lower GPU kernel ops to NVVM.
  pm.addNestedPass<gpu::GPUModuleOp>(::mlir::createConvertGpuOpsToNVVMOps());

  // Step 7: Lower arith/scf/cf/index/func everywhere (host + kernel).
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());

  // Step 8: gpu-to-llvm converts host GPU runtime ops AND finishes
  // memref type conversion. Must run while memref types still exist
  // (before FinalizeMemRefToLLVM).
  pm.addPass(createGpuToLLVMConversionPass());

  // Step 9: Finalize memref (converts remaining memref ops to LLVM).
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Step 10: Compile gpu.module → binary (kernel is fully LLVM now).
  GpuModuleToBinaryPassOptions binOpts;
  binOpts.compilationTarget = std::string(options.format);
  pm.addPass(createGpuModuleToBinaryPass(binOpts));

  // Step 11: Clean up any remaining casts.
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace lego
} // namespace mlir
