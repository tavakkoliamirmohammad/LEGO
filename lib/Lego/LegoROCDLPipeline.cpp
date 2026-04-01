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
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct LowerGpuAllReduceROCDLPass
    : public PassWrapper<LowerGpuAllReduceROCDLPass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuAllReduceROCDLPass)
  StringRef getArgument() const override { return "lego-lower-gpu-all-reduce-rocdl"; }
  StringRef getDescription() const override { return "Lower gpu.all_reduce (ROCDL)"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGpuAllReducePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct LowerGpuSubgroupReduceROCDLPass
    : public PassWrapper<LowerGpuSubgroupReduceROCDLPass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuSubgroupReduceROCDLPass)
  StringRef getArgument() const override { return "lego-lower-subgroup-reduce-rocdl"; }
  StringRef getDescription() const override { return "Lower gpu.subgroup_reduce to shuffle (ROCDL)"; }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // AMD wavefront size is 64 (GCN) or 32 (RDNA)
    populateGpuLowerSubgroupReduceToShufflePatterns(patterns, 32, 32);
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(patterns, 32, 32);
    populateGpuBreakDownSubgroupReducePatterns(patterns, 32);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

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

  // Phase 1.5: Lower gpu.subgroup_reduce → gpu.shuffle butterfly pattern.
  // Must run before ROCDL conversion. The ROCDL conversion handles
  // gpu.shuffle and gpu.all_reduce internally but NOT gpu.subgroup_reduce.
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<LowerGpuSubgroupReduceROCDLPass>());

  // Phase 2: ROCDL-specific — set target + convert GPU dialect.
  pm.addPass(std::make_unique<SetROCDLTargetPass>(
      options.chip, options.features, options.optLevel));
  ConvertGpuOpsToROCDLOpsOptions rocdlOpts;
  rocdlOpts.chipset = std::string(options.chip);
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToROCDLOps(rocdlOpts));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());

  // Phase 3: shared host LLVM lowering + binary compilation.
  buildGPUToLLVMAndBinaryPipeline(pm, options.format);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_AMDGPU
