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

/// Pass (on gpu.module): strip workgroup attributions from gpu.func, store
/// their types as attributes, and erase the block args. The uses are replaced
/// with results of `arith.constant 0 : index` as temporary placeholders that
/// will be consumed by memref ops that get converted to SPIR-V AccessChain.
/// The post-pass creates proper spirv.GlobalVariable + spirv.mlir.addressof.
struct StripWorkgroupPass
    : public PassWrapper<StripWorkgroupPass,
                          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripWorkgroupPass)

  StringRef getArgument() const override { return "lego-strip-workgroup"; }
  StringRef getDescription() const override {
    return "Strip gpu.func workgroup attributions for SPIR-V";
  }

  void runOnOperation() override {
    auto topModule = getOperation();

    // Collect workgroup info from all gpu.func ops.
    DenseMap<StringRef, SmallVector<Type, 4>> funcWGTypes;

    topModule->walk([&](gpu::GPUFuncOp funcOp) {
      unsigned numWG = funcOp.getNumWorkgroupAttributions();
      if (numWG == 0)
        return;

      auto *ctx = funcOp.getContext();
      SmallVector<Attribute, 4> wgTypes;

      // Store types and erase attributions (reverse order).
      for (int i = numWG - 1; i >= 0; --i) {
        auto idx = funcOp.getFirstWorkgroupAttributionIndex() + i;
        auto arg = funcOp.getBody().front().getArgument(idx);
        wgTypes.push_back(TypeAttr::get(arg.getType()));

        // All uses of this block arg are memref.load/store ops.
        // We can't replace them yet — let gpu-to-spirv handle the
        // conversion, then fix up in the post-pass. For now, the
        // arg must remain because removing it would invalidate uses.
        // Instead, we leave attributions alone for gpu-to-spirv but
        // tell it these are NOT attributions (set count=0).
        // The block args stay, and we extend the function type to cover them.
      }

      // Extend function type to include workgroup args as regular args.
      auto funcType = funcOp.getFunctionType();
      SmallVector<Type, 8> newInputs(funcType.getInputs());
      SmallVector<Type, 4> wgTypesList;
      for (unsigned i = 0; i < numWG; ++i) {
        auto idx = funcOp.getFirstWorkgroupAttributionIndex() + i;
        auto ty = funcOp.getBody().front().getArgument(idx).getType();
        newInputs.push_back(ty);
        wgTypesList.push_back(ty);
      }
      funcOp.setFunctionType(FunctionType::get(ctx, newInputs, funcType.getResults()));
      funcWGTypes[funcOp.getName()] = std::move(wgTypesList);

      OpBuilder b(ctx);
      funcOp->setAttr(funcOp.getNumWorkgroupAttributionsAttrName(),
                      b.getI64IntegerAttr(0));
      funcOp->removeAttr("workgroup_attrib_attrs");

      std::reverse(wgTypes.begin(), wgTypes.end());
      auto gpuMod = funcOp->getParentOfType<gpu::GPUModuleOp>();
      gpuMod->setAttr("lego.workgroup_types", ArrayAttr::get(ctx, wgTypes));
      gpuMod->setAttr("lego.workgroup_func",
                       StringAttr::get(ctx, funcOp.getName()));
    });

    // Fix gpu.launch_func calls to match the new function ABI.
    topModule->walk([&](gpu::LaunchFuncOp launchOp) {
      auto it = funcWGTypes.find(launchOp.getKernelName());
      if (it == funcWGTypes.end())
        return;

      auto loc = launchOp.getLoc();
      OpBuilder lb(launchOp);
      SmallVector<Value, 8> newOperands(launchOp.getKernelOperands());

      for (auto ty : it->second) {
        auto memrefTy = cast<MemRefType>(ty);
        auto alloc = memref::AllocOp::create(lb, loc, memrefTy);
        newOperands.push_back(alloc.getResult());
      }

      auto newLaunch = gpu::LaunchFuncOp::create(
          lb, loc, launchOp.getKernelAttr(),
          launchOp.getGridSizeOperandValues(),
          launchOp.getBlockSizeOperandValues(),
          launchOp.getDynamicSharedMemorySize(),
          newOperands,
          /*asyncToken=*/nullptr,
          launchOp.getAsyncDependencies());

      for (auto attr : launchOp->getAttrs())
        if (!newLaunch->hasAttr(attr.getName()))
          newLaunch->setAttr(attr.getName(), attr.getValue());

      launchOp.erase();
    });
  }
};

/// Post-pass (on spirv.module): create spirv.GlobalVariable with Workgroup
/// storage for each stripped workgroup attribution, then replace the
/// corresponding spirv.mlir.addressof (which references a struct-wrapped
/// StorageBuffer interface variable) with an addressof to the new Workgroup
/// global. Finally, erase the old interface variable.
struct InjectWorkgroupGlobalsPass
    : public PassWrapper<InjectWorkgroupGlobalsPass,
                          OperationPass<spirv::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InjectWorkgroupGlobalsPass)

  StringRef getArgument() const override {
    return "lego-inject-workgroup-globals";
  }
  StringRef getDescription() const override {
    return "Inject spirv.GlobalVariable for workgroup memory";
  }

  void runOnOperation() override {
    auto spvMod = getOperation();

    // Find workgroup metadata from sibling gpu.module.
    ArrayAttr typesAttr;
    auto parentMod = spvMod->getParentOfType<ModuleOp>();
    if (!parentMod)
      return;
    for (auto &op : parentMod.getBody()->getOperations()) {
      typesAttr = op.getAttrOfType<ArrayAttr>("lego.workgroup_types");
      if (typesAttr) {
        op.removeAttr("lego.workgroup_types");
        op.removeAttr("lego.workgroup_func");
        break;
      }
    }
    if (!typesAttr || typesAttr.empty())
      return;

    int64_t numWG = typesAttr.size();

    // Find interface variables with binding attrs, sorted by binding.
    SmallVector<std::pair<int, spirv::GlobalVariableOp>, 8> boundGlobals;
    for (auto g : spvMod.getOps<spirv::GlobalVariableOp>()) {
      auto b = g->getAttrOfType<IntegerAttr>("binding");
      if (b)
        boundGlobals.push_back({b.getInt(), g});
    }
    llvm::sort(boundGlobals, [](auto &a, auto &b) { return a.first < b.first; });

    if (static_cast<int64_t>(boundGlobals.size()) < numWG)
      return;

    auto *ctx = spvMod.getContext();
    OpBuilder modBuilder(spvMod.getBody(), spvMod.getBody()->begin());
    auto loc = spvMod.getLoc();

    for (int64_t i = 0; i < numWG; ++i) {
      auto memrefTy = cast<ShapedType>(
          cast<TypeAttr>(typesAttr[i]).getValue());
      auto elemTy = memrefTy.getElementType();
      int64_t n = memrefTy.getNumElements();

      // Create bare array type (no struct wrapper) with Workgroup storage.
      auto arrayTy = spirv::ArrayType::get(elemTy, n);
      auto wgPtrTy = spirv::PointerType::get(arrayTy, spirv::StorageClass::Workgroup);

      std::string wgName = "__workgroup_" + std::to_string(i);
      auto wgGlobal = spirv::GlobalVariableOp::create(
          modBuilder, loc, wgPtrTy, wgName, /*initializer=*/nullptr);

      // Find the old struct-wrapped interface variable and replace addressof uses.
      int64_t oldIdx = boundGlobals.size() - numWG + i;
      auto oldGlobal = boundGlobals[oldIdx].second;

      // Find all addressof ops referencing the old global.
      SmallVector<spirv::AddressOfOp, 4> oldAddressOfs;
      spvMod->walk([&](spirv::AddressOfOp addrOf) {
        if (addrOf.getVariable() == oldGlobal.getSymName())
          oldAddressOfs.push_back(addrOf);
      });

      for (auto addrOf : oldAddressOfs) {
        OpBuilder b(addrOf);
        auto newAddrOf = spirv::AddressOfOp::create(b, loc, wgGlobal);

        // The old addressof produces ptr<struct{array<N x T>}, StorageBuffer>.
        // The new one produces ptr<array<N x T>, Workgroup>.
        // Users are AccessChain ops that index [0][idx] into the struct.
        // We need to skip the struct index (first AccessChain index = 0).
        for (auto &use : llvm::make_early_inc_range(addrOf.getResult().getUses())) {
          auto accessChain = dyn_cast<spirv::AccessChainOp>(use.getOwner());
          if (accessChain && accessChain.getIndices().size() >= 2) {
            // Old: AccessChain ptr<struct{array}> [0, idx]
            // New: AccessChain ptr<array> [idx]  (skip struct index)
            OpBuilder acb(accessChain);
            auto elemPtrTy = spirv::PointerType::get(
                elemTy, spirv::StorageClass::Workgroup);
            SmallVector<Value, 4> newIndices(
                accessChain.getIndices().begin() + 1,
                accessChain.getIndices().end());
            auto newAC = spirv::AccessChainOp::create(
                acb, loc, elemPtrTy, newAddrOf.getResult(), newIndices);
            accessChain.getResult().replaceAllUsesWith(newAC.getResult());
            accessChain.erase();
          } else {
            // Direct use — just replace.
            use.set(newAddrOf.getResult());
          }
        }
        addrOf.erase();
      }

      // Remove the old struct-wrapped interface variable.
      auto oldName = oldGlobal.getSymName();
      auto newName = wgGlobal.getSymName();
      oldGlobal.erase();

      // Fix spirv.EntryPoint interface: replace old global ref with new one.
      spvMod->walk([&](spirv::EntryPointOp entryPt) {
        auto iface = entryPt.getInterface();
        SmallVector<Attribute, 8> newIface;
        for (auto attr : iface) {
          auto sym = cast<FlatSymbolRefAttr>(attr);
          if (sym.getValue() == oldName)
            newIface.push_back(FlatSymbolRefAttr::get(ctx, newName));
          else
            newIface.push_back(attr);
        }
        entryPt.setInterfaceAttr(ArrayAttr::get(ctx, newIface));
      });
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

  // Step 5: Strip workgroup attributions (promote to regular function args),
  // add dummy operands to gpu.launch_func, and store type metadata.
  pm.addPass(std::make_unique<StripWorkgroupPass>());

  // Step 6: Set SPIR-V target environment on gpu.module ops.
  pm.addPass(std::make_unique<SetSPIRVTargetEnvPass>(
      options.spirvVersion, options.clientAPI));

  // Step 7: Convert GPU module to SPIR-V module.
  // mapMemorySpace=false because step 3 already mapped memory spaces.
  pm.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/false));

  // Step 8: Convert remaining dialects inside spirv.module.
  pm.addNestedPass<spirv::ModuleOp>(createConvertArithToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertFuncToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createSCFToSPIRV());
  pm.addNestedPass<spirv::ModuleOp>(createConvertMemRefToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertMathToSPIRVPass());
  pm.addNestedPass<spirv::ModuleOp>(createConvertIndexToSPIRVPass());

  // Step 9: Finalize SPIR-V module (ABI lowering, VCE update).
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

  // Step 9b: Replace struct-wrapped StorageBuffer interface variables
  // for workgroup args with bare-typed Workgroup globals. Must run
  // after ABI lowering (which processes interface variables) but
  // before serialization.
  pm.addNestedPass<spirv::ModuleOp>(
      std::make_unique<InjectWorkgroupGlobalsPass>());

  // Step 10: Serialize spirv.module to binary blob attribute.
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
}

} // namespace lego
} // namespace mlir
