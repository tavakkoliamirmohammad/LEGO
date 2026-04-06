/// LEGO NVVM/CUDA pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → NVVM → PTX/cubin
///
/// Only the target-specific middle phase lives here (SetNVVMTargetPass +
/// GPU→NVVM conversion).  The shared front and tail are in Passes.cpp.

#ifdef LEGO_HAS_NVPTX

#include "Lego/Passes.h"

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/Passes.h"
using namespace mlir;

namespace {

struct SetNVVMTargetPass
    : public PassWrapper<SetNVVMTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetNVVMTargetPass)

  std::string chip;
  std::string features;
  int optLevel;

  SetNVVMTargetPass(StringRef chip = "sm_70", StringRef features = "+ptx60",
                    int optLevel = 3)
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
    auto target = NVVM::NVVMTargetAttr::get(
        ctx, optLevel, "nvptx64-nvidia-cuda", chip, features);
    getOperation()->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod.setTargetsAttr(ArrayAttr::get(ctx, {target}));
    });
  }
};

} // namespace

namespace mlir {
namespace lego {

void buildLegoToNVVMPipeline(OpPassManager &pm,
                              const LegoToNVVMPipelineOptions &options) {
  // Phase 1: shared LEGO lower + GPU outline.
  buildLegoGPUOutlinePipeline(pm);

  // Phase 1.5: Lower gpu.all_reduce and gpu.subgroup_reduce.
  addGpuAllReduceLoweringPass(pm);
  addGpuSubgroupReduceLoweringPass(pm);

  // Phase 2: NVVM-specific — set target + convert GPU dialect.
  pm.addPass(std::make_unique<SetNVVMTargetPass>(
      options.chip, options.features, options.optLevel));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());

  // Phase 3: shared host LLVM lowering + binary compilation.
  buildGPUToLLVMAndBinaryPipeline(pm, options.format);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_NVPTX
