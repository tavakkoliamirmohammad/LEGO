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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
        {spirv::Capability::Shader, spirv::Capability::Int64,
         spirv::Capability::GroupNonUniform,
         spirv::Capability::GroupNonUniformShuffle,
         spirv::Capability::GroupNonUniformShuffleRelative,
         spirv::Capability::GroupNonUniformArithmetic},
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
// LowerWorkgroupToSPIRVPass — convert gpu.func workgroup attributions to
// spirv.GlobalVariable + spirv.AccessChain/Load/Store BEFORE gpu-to-spirv.
//
// This pass runs on gpu.module after outlining and storage class mapping.
// For each workgroup attribution:
// 1. Creates spirv.GlobalVariable with bare !spirv.ptr<!spirv.array<N x T>,
//    Workgroup> at gpu.module scope
// 2. Replaces memref.load/store of the workgroup buffer with
//    spirv.mlir.addressof + spirv.AccessChain + spirv.Load/Store
// 3. Erases the workgroup block argument
//
// All generated ops are SPIR-V dialect ops, which gpu-to-spirv's conversion
// target marks as legal. They pass through the full conversion untouched
// and end up in the spirv.module.
// ============================================================================

struct LowerWorkgroupToSPIRVPass
    : public PassWrapper<LowerWorkgroupToSPIRVPass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerWorkgroupToSPIRVPass)

  StringRef getArgument() const override {
    return "lego-lower-workgroup-to-spirv";
  }
  StringRef getDescription() const override {
    return "Lower gpu.func workgroup attributions to SPIR-V ops";
  }

  void runOnOperation() override {
    auto gpuMod = getOperation();

    gpuMod->walk([&](gpu::GPUFuncOp funcOp) {
      unsigned numWG = funcOp.getNumWorkgroupAttributions();
      if (numWG == 0)
        return;

      auto *ctx = funcOp.getContext();
      auto loc = funcOp.getLoc();
      OpBuilder moduleBuilder(gpuMod.getBody(), gpuMod.getBody()->begin());

      // Process each workgroup attribution (reverse order for safe erasure).
      for (int i = static_cast<int>(numWG) - 1; i >= 0; --i) {
        auto idx = funcOp.getFirstWorkgroupAttributionIndex() + i;
        auto blockArg = funcOp.getBody().front().getArgument(idx);
        auto memrefTy = cast<MemRefType>(blockArg.getType());
        auto elemTy = memrefTy.getElementType();
        int64_t numElements = memrefTy.getNumElements();

        // 1. Create spirv.GlobalVariable at gpu.module scope.
        auto arrayTy = spirv::ArrayType::get(elemTy, numElements);
        auto ptrTy = spirv::PointerType::get(
            arrayTy, spirv::StorageClass::Workgroup);
        auto elemPtrTy = spirv::PointerType::get(
            elemTy, spirv::StorageClass::Workgroup);

        std::string name = (funcOp.getName() + "_workgroup_" +
                            Twine(i)).str();

        auto globalOp = spirv::GlobalVariableOp::create(
            moduleBuilder, loc, ptrTy, name, /*initializer=*/nullptr);

        // 2. Replace all uses: memref.load/store → spirv.Load/Store.
        OpBuilder funcBuilder(&funcOp.getBody().front(),
                              funcOp.getBody().front().begin());

        // Create addressof once at function entry.
        auto addrOf = spirv::AddressOfOp::create(funcBuilder, loc, globalOp);

        // Replace all uses of the workgroup block arg.
        SmallVector<Operation *, 16> toErase;
        for (auto &use : llvm::make_early_inc_range(blockArg.getUses())) {
          Operation *user = use.getOwner();
          OpBuilder b(user);

          if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
            // memref.load %smem[%idx] → spirv.AccessChain + spirv.Load
            assert(loadOp.getIndices().size() == 1 &&
                   "expected 1D workgroup memref");
            auto idx_val = loadOp.getIndices()[0];
            // Cast index to i32 for SPIR-V.
            auto i32Ty = IntegerType::get(ctx, 32);
            auto idxCast = arith::IndexCastOp::create(b, loc, i32Ty, idx_val);
            auto ac = spirv::AccessChainOp::create(
                b, loc, elemPtrTy, addrOf.getResult(),
                ValueRange{idxCast.getResult()});
            auto spvLoad = spirv::LoadOp::create(b, loc, ac.getResult());
            loadOp.getResult().replaceAllUsesWith(spvLoad.getResult());
            toErase.push_back(loadOp);

          } else if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
            // memref.store %val, %smem[%idx] → spirv.AccessChain + spirv.Store
            assert(storeOp.getIndices().size() == 1 &&
                   "expected 1D workgroup memref");
            auto idx_val = storeOp.getIndices()[0];
            auto i32Ty = IntegerType::get(ctx, 32);
            auto idxCast = arith::IndexCastOp::create(b, loc, i32Ty, idx_val);
            auto ac = spirv::AccessChainOp::create(
                b, loc, elemPtrTy, addrOf.getResult(),
                ValueRange{idxCast.getResult()});
            spirv::StoreOp::create(b, loc, ac.getResult(),
                                   storeOp.getValueToStore());
            toErase.push_back(storeOp);
          }
          // Other uses (shouldn't happen for well-formed workgroup buffers)
        }

        for (auto *op : toErase)
          op->erase();

        // 3. Erase the workgroup block argument.
        funcOp.getBody().front().eraseArgument(idx);
      }

      // Update workgroup attribution count to zero.
      OpBuilder b(ctx);
      funcOp->setAttr(funcOp.getNumWorkgroupAttributionsAttrName(),
                      b.getI64IntegerAttr(0));
      funcOp->removeAttr("workgroup_attrib_attrs");
    });
  }
};

/// Lowers gpu.all_reduce ops into shared memory + shuffle tree reduction.
struct LowerGpuAllReduceSPIRVPass
    : public PassWrapper<LowerGpuAllReduceSPIRVPass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuAllReduceSPIRVPass)

  StringRef getArgument() const override {
    return "lego-lower-gpu-all-reduce-spirv";
  }
  StringRef getDescription() const override {
    return "Lower gpu.all_reduce to shared memory + shuffle tree (SPIR-V)";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGpuAllReducePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

/// Lowers gpu.subgroup_reduce to gpu.shuffle (butterfly pattern).
struct LowerGpuSubgroupReduceToShufflePass
    : public PassWrapper<LowerGpuSubgroupReduceToShufflePass,
                          OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGpuSubgroupReduceToShufflePass)

  StringRef getArgument() const override {
    return "lego-lower-subgroup-reduce-to-shuffle";
  }
  StringRef getDescription() const override {
    return "Lower gpu.subgroup_reduce to gpu.shuffle butterfly pattern";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // 32 = subgroup size (NVIDIA), 32 = shuffle bitwidth
    populateGpuLowerSubgroupReduceToShufflePatterns(patterns, 32, 32);
    populateGpuLowerClusteredSubgroupReduceToShufflePatterns(patterns, 32, 32);
    populateGpuBreakDownSubgroupReducePatterns(patterns, 32);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
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

  // Step 3.5: Sink constants back into gpu.launch (CSE may have hoisted them).
  pm.addPass(createGpuLaunchSinkIndexComputationsPass());

  // Step 4: Outline inline gpu.launch into gpu.module + gpu.func.
  pm.addPass(createGpuKernelOutliningPass());

  // Step 4.5: Lower gpu.subgroup_reduce to gpu.shuffle (butterfly pattern).
  // While GPU-to-SPIR-V can handle subgroup_reduce natively, the shuffle
  // lowering is more portable across wgpu/Vulkan runtime implementations.
  // gpu.all_reduce and gpu.shuffle are handled natively by GPU-to-SPIR-V.
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<LowerGpuSubgroupReduceToShufflePass>());

  // Step 5: Lower workgroup attributions to SPIR-V ops (spirv.GlobalVariable
  // + spirv.Load/Store). These ops are legal in the gpu-to-spirv conversion
  // target and pass through untouched into the spirv.module.
  pm.addNestedPass<gpu::GPUModuleOp>(
      std::make_unique<LowerWorkgroupToSPIRVPass>());

  // Step 6: Set SPIR-V target environment on gpu.module ops.
  pm.addPass(std::make_unique<SetSPIRVTargetEnvPass>(
      options.spirvVersion, options.clientAPI));

  // Step 7: Convert GPU module to SPIR-V module.
  // The upstream pass runs ALL conversion patterns together (GPU + Arith +
  // SCF + MemRef + Func + Index) in a single conversion step.
  pm.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  // Step 8: Finalize SPIR-V module.
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
  pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

  // Step 10: Serialize spirv.module to binary blob attribute.
  pm.addPass(std::make_unique<SerializeSPIRVPass>());
}

} // namespace lego
} // namespace mlir
