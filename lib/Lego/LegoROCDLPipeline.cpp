/// LEGO ROCDL/AMD pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → ROCDL → HSACO
///
/// Only the target-specific middle phase lives here (SetROCDLTargetPass +
/// GPU→ROCDL conversion).  The shared front and tail are in Passes.cpp.

#ifdef LEGO_HAS_AMDGPU

#include "Lego/Passes.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
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

struct SetROCDLTargetPass
    : public PassWrapper<SetROCDLTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetROCDLTargetPass)

  std::string chip;
  std::string features;
  int optLevel;

  SetROCDLTargetPass(StringRef chip = "gfx900", StringRef features = "",
                     int optLevel = 2)
      : chip(chip.str()), features(features.str()), optLevel(optLevel) {}

  StringRef getArgument() const override { return "lego-set-rocdl-target"; }
  StringRef getDescription() const override {
    return "Set ROCDL compilation target on gpu.module ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registerROCDLDialectTranslation(registry);
    registerGPUDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registerBuiltinDialectTranslation(registry);
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    ROCDL::registerROCDLTargetInterfaceExternalModels(*ctx);
    auto target = ROCDL::ROCDLTargetAttr::get(
        ctx, optLevel, "amdgcn-amd-amdhsa", chip, features);
    getOperation()->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod.setTargetsAttr(ArrayAttr::get(ctx, {target}));
    });
  }
};

} // namespace

namespace mlir {
namespace lego {

void buildLegoToROCDLPipeline(OpPassManager &pm,
                               const LegoToROCDLPipelineOptions &options) {
  // Phase 1: shared LEGO lower + GPU outline.
  buildLegoGPUOutlinePipeline(pm);

  // Phase 2: ROCDL-specific — set target + convert GPU dialect.
  pm.addPass(std::make_unique<SetROCDLTargetPass>(
      options.chip, options.features, options.optLevel));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToROCDLOps());

  // Phase 3: shared host LLVM lowering + binary compilation.
  buildGPUToLLVMAndBinaryPipeline(pm, options.format);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_AMDGPU
