/// LEGO SPIR-V pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → SPIR-V

#include "Lego/Passes.h"

// SPIR-V conversion headers
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRVPass.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"
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
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// Parse SPIR-V version string (e.g., "1.5") to enum.
static spirv::Version parseSPIRVVersion(StringRef version) {
  return llvm::StringSwitch<spirv::Version>(version)
      .Case("1.0", spirv::Version::V_1_0)
      .Case("1.1", spirv::Version::V_1_1)
      .Case("1.2", spirv::Version::V_1_2)
      .Case("1.3", spirv::Version::V_1_3)
      .Case("1.4", spirv::Version::V_1_4)
      .Case("1.5", spirv::Version::V_1_5)
      .Case("1.6", spirv::Version::V_1_6)
      .Default(spirv::Version::V_1_5);
}

/// Parse client API string to enum.
static spirv::ClientAPI parseClientAPI(StringRef api) {
  return llvm::StringSwitch<spirv::ClientAPI>(api)
      .CaseLower("vulkan", spirv::ClientAPI::Vulkan)
      .CaseLower("opencl", spirv::ClientAPI::OpenCL)
      .Default(spirv::ClientAPI::Vulkan);
}

// ============================================================================
// SetSPIRVTargetEnvPass — stamps spirv.target_env on gpu.module ops
// ============================================================================

struct SetSPIRVTargetEnvPass
    : public PassWrapper<SetSPIRVTargetEnvPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVTargetEnvPass)

  std::string spirvVersionStr;
  std::string clientAPIStr;

  SetSPIRVTargetEnvPass(StringRef spirvVersion = "1.5",
                         StringRef clientAPI = "vulkan")
      : spirvVersionStr(spirvVersion.str()), clientAPIStr(clientAPI.str()) {}

  StringRef getArgument() const override { return "lego-set-spirv-target-env"; }
  StringRef getDescription() const override {
    return "Set SPIR-V target environment on gpu.module ops";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();

    auto version = parseSPIRVVersion(spirvVersionStr);
    auto clientAPI = parseClientAPI(clientAPIStr);

    auto triple = spirv::VerCapExtAttr::get(
        version,
        {spirv::Capability::Shader, spirv::Capability::Int64},
        {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, ctx);

    auto limits = spirv::getDefaultResourceLimits(ctx);

    auto targetEnv = spirv::TargetEnvAttr::get(
        triple, limits, clientAPI, spirv::Vendor::Unknown,
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
      std::string binaryStr;
      llvm::raw_string_ostream os(binaryStr);
      for (size_t i = 0; i < binary.size(); ++i) {
        if (i > 0) os << ",";
        os << binary[i];
      }
      moduleOp->setAttr("lego.spirv_binary",
                         StringAttr::get(spvMod.getContext(), binaryStr));

      spvMod.erase();
    });

    if (hadFailure)
      signalPassFailure();
  }
};

// ============================================================================
// ConvertWorkgroupToAllocaPass — works around gpu-to-spirv crash
// ============================================================================

struct ConvertWorkgroupToAllocaPass
    : public PassWrapper<ConvertWorkgroupToAllocaPass,
                          OperationPass<gpu::GPUModuleOp>> {
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

void buildLegoToSPIRVPipeline(OpPassManager &pm,
                               const LegoToSPIRVPipelineOptions &options) {
  // Step 1: Lower LEGO ops to arith.
  buildLegoLowerPipeline(pm);

  // Step 2: Fold constants and clean up before outlining.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Step 3: Map memref memory spaces to SPIR-V storage classes BEFORE outlining.
  pm.addPass(createMapMemRefStorageClassPass());

  // Step 4: Outline inline gpu.launch into gpu.module + gpu.func.
  pm.addPass(createGpuKernelOutliningPass());

  // Step 5: Convert workgroup attributions to memref.alloca (SPIR-V workaround).
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<ConvertWorkgroupToAllocaPass>());

  // Step 6: Set SPIR-V target environment on gpu.module ops.
  pm.addPass(std::make_unique<SetSPIRVTargetEnvPass>(
      options.spirvVersion, options.clientAPI));

  // Step 7: Convert GPU module to SPIR-V module.
  pm.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  // Step 8: Convert remaining dialects inside spirv.module.
  pm.addNestedPass<spirv::ModuleOp>(createConvertArithToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertFuncToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createSCFToSPIRV());
  pm.addNestedPass<spirv::ModuleOp>(createConvertMemRefToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertMathToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertIndexToSPIRVPass());

  // Step 9: Finalize SPIR-V module.
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

  // Step 10: Serialize spirv.module to binary blob attribute.
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
}

} // namespace lego
} // namespace mlir
