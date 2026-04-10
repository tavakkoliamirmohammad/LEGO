/// LEGO XeVM/Intel GPU pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → LLVM SPIR-V + XeVM target → binary
///
/// Only the target-specific middle phase lives here (SetXeVMTargetPass +
/// GPU→LLVM SPIR-V conversion).  The shared front and tail are in Passes.cpp.

#ifdef LEGO_HAS_XEVM

#include "Lego/Passes.h"

#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Target/LLVM/XeVM/Target.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
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

struct SetXeVMTargetPass
    : public PassWrapper<SetXeVMTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetXeVMTargetPass)

  std::string chip;
  int optLevel;

  SetXeVMTargetPass(StringRef chip = "bmg", int optLevel = 2)
      : chip(chip.str()), optLevel(optLevel) {}

  StringRef getArgument() const override { return "lego-set-xevm-target"; }
  StringRef getDescription() const override {
    return "Set XeVM compilation target on gpu.module ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registerXeVMDialectTranslation(registry);
    registerGPUDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    registerBuiltinDialectTranslation(registry);
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    xevm::registerXeVMTargetInterfaceExternalModels(*ctx);
    auto target = xevm::XeVMTargetAttr::get(
        ctx, optLevel, "spirv64-unknown-unknown", chip);
    getOperation()->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod.setTargetsAttr(ArrayAttr::get(ctx, {target}));
    });
  }
};

} // namespace

namespace mlir {
namespace lego {

void buildLegoToXeVMPipeline(OpPassManager &pm,
                              const LegoToXeVMPipelineOptions &options) {
  // Phase 1: shared LEGO lower + GPU outline.
  buildLegoGPUOutlinePipeline(pm);

  // Phase 1.5: Lower gpu.subgroup_reduce.
  addGpuSubgroupReduceLoweringPass(pm);

  // Phase 2: XeVM-specific — set target + convert GPU dialect to LLVM SPIR-V.
  pm.addPass(std::make_unique<SetXeVMTargetPass>(
      options.chip, options.optLevel));
  ConvertGpuOpsToLLVMSPVOpsOptions spvOpts;
  spvOpts.use64bitIndex = true;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToLLVMSPVOps(spvOpts));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());

  // Phase 3: shared host LLVM lowering + binary compilation.
  buildGPUToLLVMAndBinaryPipeline(pm, options.format);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_XEVM
