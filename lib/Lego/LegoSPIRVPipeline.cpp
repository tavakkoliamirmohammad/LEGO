/// LEGO SPIR-V pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → SPIR-V

#include "Lego/Passes.h"

// SPIR-V conversion headers
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
// PromoteWorkgroupToGlobalPass — promote gpu.func workgroup attributions to
// spirv.GlobalVariable ops with Workgroup storage class.
//
// The standard gpu-to-spirv pass ignores workgroup attributions. This pass
// runs BEFORE gpu-to-spirv on gpu.module ops:
// 1. For each workgroup attribution, record its type
// 2. Replace uses with a memref.get_global (to be lowered later)
// 3. Remove the attribution block args
// 4. After gpu-to-spirv converts gpu.module → spirv.module, a second nested
//    pass creates the actual spirv.GlobalVariable ops.
// ============================================================================

/// Pass (on top-level module): promote workgroup attributions to regular
/// function arguments AND add corresponding dummy operands to gpu.launch_func
/// so the ABI matches. After gpu-to-spirv, the post-pass fixes the storage class.
struct PromoteWorkgroupToArgsPass
    : public PassWrapper<PromoteWorkgroupToArgsPass,
                          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteWorkgroupToArgsPass)

  StringRef getArgument() const override {
    return "lego-promote-workgroup-to-args";
  }
  StringRef getDescription() const override {
    return "Promote gpu.func workgroup attributions to function arguments";
  }

  void runOnOperation() override {
    auto topModule = getOperation();
    auto *ctx = topModule.getContext();
    OpBuilder b(ctx);

    // First, collect workgroup info from all gpu.func ops.
    struct WGInfo {
      SmallVector<Type, 4> types;
      unsigned numWG = 0;
    };
    DenseMap<StringRef, WGInfo> funcWGInfo;

    topModule->walk([&](gpu::GPUFuncOp funcOp) {
      unsigned numWorkgroup = funcOp.getNumWorkgroupAttributions();
      if (numWorkgroup == 0)
        return;

      auto funcType = funcOp.getFunctionType();
      SmallVector<Type, 8> newInputTypes(funcType.getInputs());
      WGInfo info;
      info.numWG = numWorkgroup;

      for (unsigned i = 0; i < numWorkgroup; ++i) {
        auto idx = funcOp.getFirstWorkgroupAttributionIndex() + i;
        auto blockArg = funcOp.getBody().front().getArgument(idx);
        auto ty = blockArg.getType();
        newInputTypes.push_back(ty);
        info.types.push_back(ty);
      }

      auto newFuncType = FunctionType::get(ctx, newInputTypes,
                                            funcType.getResults());
      funcOp.setFunctionType(newFuncType);

      funcOp->setAttr(funcOp.getNumWorkgroupAttributionsAttrName(),
                      b.getI64IntegerAttr(0));
      funcOp->removeAttr("workgroup_attrib_attrs");

      // Store on the parent gpu.module for the post-pass.
      auto gpuMod = funcOp->getParentOfType<gpu::GPUModuleOp>();
      if (gpuMod) {
        gpuMod->setAttr("lego.num_workgroup_args",
                         b.getI64IntegerAttr(numWorkgroup));
      }

      funcWGInfo[funcOp.getName()] = std::move(info);
    });

    // Update gpu.launch_func calls to pass dummy operands for workgroup args.
    topModule->walk([&](gpu::LaunchFuncOp launchOp) {
      auto kernelName = launchOp.getKernelName();
      auto it = funcWGInfo.find(kernelName);
      if (it == funcWGInfo.end())
        return;

      auto &info = it->second;
      auto loc = launchOp.getLoc();
      OpBuilder lb(launchOp);

      // Create dummy memref.alloc for each workgroup type and add as
      // kernel operands. These are just placeholders — the host never
      // actually uses them. The post-pass will convert the corresponding
      // SPIR-V interface variables to Workgroup storage.
      SmallVector<Value, 8> newOperands(launchOp.getKernelOperands());
      for (auto ty : info.types) {
        auto memrefTy = cast<MemRefType>(ty);
        // Create a 1-element alloc as placeholder (host-side).
        auto alloc = memref::AllocOp::create(lb, loc, memrefTy);
        newOperands.push_back(alloc.getResult());
      }

      // Rebuild launch_func with updated operands.
      auto newLaunch = gpu::LaunchFuncOp::create(
          lb, loc, launchOp.getKernelAttr(),
          launchOp.getGridSizeOperandValues(),
          launchOp.getBlockSizeOperandValues(),
          launchOp.getDynamicSharedMemorySize(),
          newOperands,
          /*asyncToken=*/nullptr,
          launchOp.getAsyncDependencies());

      // Copy over any extra attributes.
      for (auto attr : launchOp->getAttrs()) {
        if (!newLaunch->hasAttr(attr.getName()))
          newLaunch->setAttr(attr.getName(), attr.getValue());
      }

      launchOp.erase();
    });
  }
};

/// Post-conversion pass (on spirv.module): find the interface variables that
/// correspond to former workgroup arguments and change their storage class
/// from StorageBuffer to Workgroup.
struct FixWorkgroupStorageClassPass
    : public PassWrapper<FixWorkgroupStorageClassPass,
                          OperationPass<spirv::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixWorkgroupStorageClassPass)

  StringRef getArgument() const override {
    return "lego-fix-workgroup-storage-class";
  }
  StringRef getDescription() const override {
    return "Fix storage class of workgroup variables from StorageBuffer to Workgroup";
  }

  void runOnOperation() override {
    auto spvMod = getOperation();

    // Find the workgroup count. The attribute is on the gpu.module
    // (sibling of spirv.module in the top-level module).
    int64_t numWG = 0;
    auto parentMod = spvMod->getParentOfType<ModuleOp>();
    if (parentMod) {
      for (auto &op : parentMod.getBody()->getOperations()) {
        auto attr = op.getAttrOfType<IntegerAttr>("lego.num_workgroup_args");
        if (attr) {
          numWG = attr.getInt();
          op.removeAttr("lego.num_workgroup_args");
          break;
        }
      }
    }
    if (numWG == 0)
      return;

    // Find interface variables (globals with binding attrs) and sort by
    // binding number. The workgroup args were appended last, so they
    // have the highest binding numbers.
    SmallVector<std::pair<int, spirv::GlobalVariableOp>, 8> boundGlobals;
    for (auto globalOp : spvMod.getOps<spirv::GlobalVariableOp>()) {
      auto bindingAttr = globalOp->getAttrOfType<IntegerAttr>("binding");
      if (bindingAttr) {
        boundGlobals.push_back({bindingAttr.getInt(), globalOp});
      }
    }

    // Sort by binding number.
    llvm::sort(boundGlobals, [](const auto &a, const auto &b) {
      return a.first < b.first;
    });

    // The last numWG bound globals are the workgroup ones.
    if (static_cast<int64_t>(boundGlobals.size()) < numWG)
      return;

    for (int64_t i = boundGlobals.size() - numWG;
         i < static_cast<int64_t>(boundGlobals.size()); ++i) {
      auto globalOp = boundGlobals[i].second;
      // Remove descriptor bindings — workgroup vars aren't interface variables.
      globalOp->removeAttr("descriptorSet");
      globalOp->removeAttr("binding");
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

  // Step 5: Promote workgroup attributions to regular function arguments
  // and add dummy operands to gpu.launch_func to match the new ABI.
  pm.addPass(std::make_unique<PromoteWorkgroupToArgsPass>());

  // Step 6: Set SPIR-V target environment on gpu.module ops.
  pm.addPass(std::make_unique<SetSPIRVTargetEnvPass>(
      options.spirvVersion, options.clientAPI));

  // Step 7: Convert GPU module to SPIR-V module.
  // mapMemorySpace=false because step 3 already mapped memory spaces
  // (0→StorageBuffer, 3→Workgroup). This preserves Workgroup storage
  // class on shared memory args instead of overriding to StorageBuffer.
  pm.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/false));

  // Step 7b: Remove descriptor set/binding from workgroup globals
  // (they're not interface variables).
  pm.addNestedPass<spirv::ModuleOp>(
      std::make_unique<FixWorkgroupStorageClassPass>());

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
