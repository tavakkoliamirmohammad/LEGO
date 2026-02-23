#include "Lego/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::lego;

void mlir::lego::registerLegoPipelines() {
  PassPipelineRegistration<>("lego-lower", "Lego e2e lowering pipeline",
    [](OpPassManager &pm) {
      pm.addPass(createLegoNormalizationPass());
      pm.addPass(createLegoToArithPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addPass(createLegoArithSimplificationPass());
      pm.addPass(arith::createIntRangeOptimizationsPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    });
}
