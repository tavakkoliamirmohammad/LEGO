/// LEGO GPU pass pipelines: SPIR-V and NVVM.
///
/// Pipeline: LEGO ops → Arith (reuse lego-lower) → GPU outlining → SPIR-V
///
/// The LEGO lowering (lego-lower) uses greedy pattern rewriting on ModuleOp,
/// so it already works inside gpu.func / gpu.launch regions — no special
/// handling needed.

#include "Lego/Passes.h"

// SPIR-V conversion headers
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"
// createSCFToSPIRV() is declared via GEN_PASS_DECL in the pass header,
// not as createConvertSCFToSPIRVPass(). Include the generated decl.
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/SPIRV/Serialization.h"
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
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// ============================================================================
// SetSPIRVTargetEnvPass — stamps spirv.target_env on gpu.module ops
// ============================================================================

struct SetSPIRVTargetEnvPass
    : public PassWrapper<SetSPIRVTargetEnvPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVTargetEnvPass)

  StringRef getArgument() const override { return "lego-set-spirv-target-env"; }
  StringRef getDescription() const override {
    return "Set SPIR-V target environment on gpu.module ops";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();

    // Vulkan 1.3 / SPIR-V 1.5 compute baseline.
    // Capability::Shader is sufficient for LEGO's index arithmetic (i32 arith).
    auto triple = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_5,
        {spirv::Capability::Shader, spirv::Capability::Int64},
        {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, ctx);

    auto limits = spirv::getDefaultResourceLimits(ctx);

    auto targetEnv = spirv::TargetEnvAttr::get(
        triple, limits, spirv::ClientAPI::Vulkan, spirv::Vendor::Unknown,
        spirv::DeviceType::Unknown,
        spirv::TargetEnvAttr::kUnknownDeviceID);

    auto attrName = spirv::getTargetEnvAttrName();

    getOperation()->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod->setAttr(attrName, targetEnv);

      // Set spirv.entry_point_abi on each gpu.func that is a kernel.
      gpuMod->walk([&](gpu::GPUFuncOp funcOp) {
        if (!funcOp.isKernel())
          return;
        if (funcOp->hasAttr(spirv::getEntryPointABIAttrName()))
          return;

        // Extract workgroup size from known_block_size if available.
        SmallVector<int32_t, 3> workgroupSize = {1, 1, 1};
        if (auto blockSize = funcOp->getAttrOfType<DenseI32ArrayAttr>(
                "known_block_size")) {
          auto vals = blockSize.asArrayRef();
          for (size_t i = 0; i < std::min(vals.size(), (size_t)3); ++i)
            workgroupSize[i] = vals[i];
        }

        funcOp->setAttr(spirv::getEntryPointABIAttrName(),
                        spirv::getEntryPointABIAttr(ctx, workgroupSize));
      });
    });
  }
};

// ============================================================================
// SerializeSPIRVPass — serialize spirv.module ops to binary blobs
// ============================================================================

struct SerializeSPIRVPass
    : public PassWrapper<SerializeSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeSPIRVPass)

  StringRef getArgument() const override { return "lego-serialize-spirv"; }
  StringRef getDescription() const override {
    return "Serialize spirv.module ops to binary attributes";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool hadFailure = false;

    moduleOp->walk([&](spirv::ModuleOp spvMod) {
      SmallVector<uint32_t, 0> binary;
      spirv::SerializationOptions opts;
      opts.emitSymbolName = true;

      if (failed(spirv::serialize(spvMod, binary, opts))) {
        spvMod.emitError("failed to serialize SPIR-V module");
        hadFailure = true;
        return;
      }

      // Encode the binary as a comma-separated string of uint32 words.
      // This avoids MLIR verifier issues with dense attributes on builtin.module.
      std::string binaryStr;
      llvm::raw_string_ostream os(binaryStr);
      for (size_t i = 0; i < binary.size(); ++i) {
        if (i > 0) os << ",";
        os << binary[i];
      }
      moduleOp->setAttr("lego.spirv_binary",
                         StringAttr::get(spvMod.getContext(), binaryStr));

      // Remove the spirv.module op now that it's serialized.
      spvMod.erase();
    });

    if (hadFailure)
      signalPassFailure();
  }
};

// ============================================================================
// SetNVVMTargetPass — stamps #nvvm.target on gpu.module ops
// ============================================================================

struct SetNVVMTargetPass
    : public PassWrapper<SetNVVMTargetPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetNVVMTargetPass)

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

    auto nvvmTarget = NVVM::NVVMTargetAttr::get(ctx);
    getOperation()->walk([&](gpu::GPUModuleOp gpuMod) {
      gpuMod.setTargetsAttr(ArrayAttr::get(ctx, {nvvmTarget}));
    });
  }
};

// ============================================================================
// ConvertWorkgroupToAllocaPass — works around gpu-to-spirv crash
// by converting gpu.func workgroup attributions to memref.alloca
// ============================================================================

struct ConvertWorkgroupToAllocaPass
    : public PassWrapper<ConvertWorkgroupToAllocaPass, OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertWorkgroupToAllocaPass)

  StringRef getArgument() const override { return "lego-workgroup-to-alloca"; }
  StringRef getDescription() const override {
    return "Convert gpu.func workgroup attributions to memref.alloca for SPIR-V";
  }

  void runOnOperation() override {
    auto gpuMod = getOperation();
    bool hasWorkgroup = false;
    gpuMod->walk([&](gpu::GPUFuncOp funcOp) {
      if (funcOp.getNumWorkgroupAttributions() > 0)
        hasWorkgroup = true;
    });
    if (hasWorkgroup) {
      gpuMod.emitError(
          "SPIR-V backend does not yet support workgroup (shared) memory. "
          "Use target='cuda' for kernels with shared memory, or remove "
          "shared=True from LayoutBuffer.");
      signalPassFailure();
    }
  }
};

} // namespace

// ============================================================================
// Pipeline definition
// ============================================================================

namespace mlir {
namespace lego {

void buildLegoToSPIRVPipeline(OpPassManager &pm) {
  // Step 1: Lower LEGO ops to arith — reuses the existing lego-lower pipeline.
  // This works inside gpu.func because applyPatternsGreedily walks all regions.
  buildLegoLowerPipeline(pm);

  // Step 2: Fold constants and clean up before outlining.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Step 3: Map memref memory spaces to SPIR-V storage classes BEFORE outlining.
  // This must happen before gpu-kernel-outlining so that workgroup memory
  // attributions (memref with address_space=3) get converted to
  // #spirv.storage_class<Workgroup> before becoming gpu.func arguments.
  pm.addPass(createMapMemRefStorageClassPass());

  // Step 4: Outline inline gpu.launch into gpu.module + gpu.func.
  pm.addPass(createGpuKernelOutliningPass());

  // Step 5: Convert workgroup attributions to memref.alloca (SPIR-V workaround).
  // The upstream gpu-to-spirv crashes on workgroup attributions in gpu.func.
  // This pass converts them to alloca ops which SPIR-V handles correctly.
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<ConvertWorkgroupToAllocaPass>());

  // Step 6: Set SPIR-V target environment on gpu.module ops.
  pm.addPass(std::make_unique<SetSPIRVTargetEnvPass>());

  // Step 7: Convert GPU module to SPIR-V module.
  pm.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  // Step 7: Convert remaining dialects inside spirv.module.
  pm.addNestedPass<spirv::ModuleOp>(createConvertArithToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertFuncToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createSCFToSPIRV());
  pm.addNestedPass<spirv::ModuleOp>(createConvertMemRefToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertMathToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertIndexToSPIRVPass());

  // Step 8: Finalize SPIR-V module.
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

  // Step 9: Serialize spirv.module to binary blob attribute.
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
}

void buildLegoToNVVMPipeline(OpPassManager &pm) {
  // Step 1: Lower LEGO ops to arith.
  buildLegoLowerPipeline(pm);

  // Step 2: Fold constants before outlining.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Step 3: Outline gpu.launch → gpu.module + gpu.func.
  pm.addPass(createGpuKernelOutliningPass());

  // Step 4: Set #nvvm.target on gpu.module ops.
  pm.addPass(std::make_unique<SetNVVMTargetPass>());

  // Step 5: Lower GPU ops to NVVM dialect inside gpu.module.
  pm.addNestedPass<gpu::GPUModuleOp>(::mlir::createConvertGpuOpsToNVVMOps());

  // Steps 6-9: GPU compilation pipeline.
  // Key constraint: FinalizeMemRefToLLVM is ModuleOp-only.
  // So we run ALL LLVM lowering at module level, then gpu-module-to-binary.

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
  binOpts.compilationTarget = "fatbin";
  pm.addPass(createGpuModuleToBinaryPass(binOpts));

  // Step 11: Clean up any remaining casts.
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace lego
} // namespace mlir
