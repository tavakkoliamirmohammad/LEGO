#include "Lego/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;
using namespace mlir::lego;

namespace {

/// Runs arith-simplification + int-range + canonicalize + CSE in a loop
/// until the IR stops changing (fixed-point) or a max iteration count.
struct FixedPointSimplificationPass
    : public PassWrapper<FixedPointSimplificationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixedPointSimplificationPass)

  StringRef getArgument() const override {
    return "lego-fixed-point-simplification";
  }
  StringRef getDescription() const override {
    return "Run arith simplification + int-range to fixed point";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (;;) {
      // Fingerprint the IR before.
      OperationFingerPrint before(module);

      // Run the sub-pipeline.
      PassManager subPM(module.getContext());
      subPM.addPass(createLegoArithSimplificationPass());
      subPM.addPass(arith::createIntRangeOptimizationsPass());
      subPM.addPass(createCanonicalizerPass());
      subPM.addPass(createCSEPass());
      if (failed(subPM.run(module))) {
        signalPassFailure();
        return;
      }

      // Check if anything changed.
      OperationFingerPrint after(module);
      if (before == after)
        break;
    }
  }
};

} // namespace

namespace mlir {
namespace lego {

/// Populates the standard lego-lower pipeline (LEGO → Arith).
static void buildLegoLowerPipeline(OpPassManager &pm) {
  pm.addPass(createLegoMaterializeAssumeBoundsPass());
  pm.addPass(createLegoNormalizationPass());
  pm.addPass(createLegoToArithPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Run arith simplification + int-range to fixed point.
  pm.addPass(std::make_unique<FixedPointSimplificationPass>());
  // Remove remui wrappers from materialized assume bounds.
  pm.addPass(createLegoMaterializeAssumeBoundsPass(/*cleanup=*/true));
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
