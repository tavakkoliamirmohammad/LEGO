/// LEGO LLVM SPIR-V pass pipeline.
///
/// Pipeline: LEGO ops → Arith → GPU outlining → LLVM SPIR-V
///
/// Same three-phase pattern as lego-to-nvvm / lego-to-rocdl.
/// Phase 2 uses convert-gpu-to-llvm-spv to lower GPU ops to LLVM
/// dialect with SPIR-V calling conventions (e.g. _Z12get_local_idj).

#ifdef LEGO_HAS_SPIRV

#include "Lego/Passes.h"

#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace mlir {
namespace lego {

void buildLegoToLLVMSPIRVPipeline(
    OpPassManager &pm, const LegoToLLVMSPIRVPipelineOptions &options) {
  // Phase 1: shared LEGO lower + GPU outline.
  buildLegoGPUOutlinePipeline(pm);

  // Phase 2: convert GPU ops to LLVM with SPIR-V calling conventions.
  ConvertGpuOpsToLLVMSPVOpsOptions spvOpts;
  spvOpts.use64bitIndex = true;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToLLVMSPVOps(spvOpts));
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertMathToLLVMPass());

  // Phase 3: shared host-side LLVM lowering.
  buildGPUHostLLVMPipeline(pm);
}

} // namespace lego
} // namespace mlir

#endif // LEGO_HAS_SPIRV
