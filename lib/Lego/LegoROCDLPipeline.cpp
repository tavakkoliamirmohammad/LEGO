/// LEGO ROCDL/AMD pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → ROCDL → HSACO
///
/// Only the target-specific middle phase lives here (SetROCDLTargetPass +
/// GPU→ROCDL conversion).  The shared front and tail are in Passes.cpp.

#ifdef LEGO_HAS_AMDGPU

#include "Lego/Passes.h"

#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
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
using namespace mlir;

namespace {

struct SetROCDLTargetPass
    : public PassWrapper<SetROCDLTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetROCDLTargetPass)

  std::string chip;
  std::string features;
  int optLevel;

  SetROCDLTargetPass(StringRef chip = "gfx900", StringRef features = "",
                     int optLevel = 3)
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
    // Register ConvertToLLVMPatternInterface extensions for all dialects
    // (memref, arith, cf, func, etc.). Without this, the GPU→ROCDL partial
    // conversion can't lower memref ops to LLVM, causing gpu.shuffle
    // legalization to fail when kernels have memref arguments.
    registerConvertToLLVMDependentDialectLoading(registry);
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
  // Use subgroupSize=32: the ROCDL GPUShuffleOpLowering (ds_bpermute) handles
  // all shuffle widths. Using 64 would cause the upstream greedy promotion to
  // partially convert some shuffles to amdgpu.swizzle_bitmode, leaving the
  // offset=32 shuffle unconverted (only offset < 32 is promotable).
  addGpuSubgroupReduceLoweringPass(pm);

  // Phase 1.6: Lower scf.if/for inside GPU kernels to cf branches.
  // The GPU→ROCDL conversion requires all structured control flow to be
  // lowered first — unlike NVVM, the ROCDL partial conversion doesn't
  // handle scf ops through the ConvertToLLVMPatternInterface.
  pm.addNestedPass<gpu::GPUModuleOp>(createSCFToControlFlowPass());

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
