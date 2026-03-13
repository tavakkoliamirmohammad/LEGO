#include "Lego/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

using namespace mlir;
using namespace mlir::lego;

namespace mlir {
namespace lego {

/// Populates the standard lego-lower pipeline (LEGO → Arith).
static void buildLegoLowerPipeline(OpPassManager &pm) {
  pm.addPass(createLegoNormalizationPass());
  pm.addPass(createLegoToArithPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLegoArithSimplificationPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

/// Populates the full lego-to-llvm pipeline (LEGO → Arith → LLVM).
void buildLegoToLLVMPipeline(OpPassManager &pm) {
  buildLegoLowerPipeline(pm);

  // Lower to LLVM dialect
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Clean up LLVM IR
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void registerLegoPipelines() {
  PassPipelineRegistration<>("lego-lower",
    "Lego e2e lowering pipeline (LEGO -> Arith)",
    buildLegoLowerPipeline);

  PassPipelineRegistration<>("lego-to-llvm",
    "Full LEGO lowering to LLVM dialect (LEGO -> Arith -> LLVM)",
    buildLegoToLLVMPipeline);
}

} // namespace lego
} // namespace mlir
