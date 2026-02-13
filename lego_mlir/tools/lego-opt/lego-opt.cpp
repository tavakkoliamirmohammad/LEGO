#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "Lego/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Transforms/Passes.h"

#include "Lego/LegoDialect.h"
#include "Lego/LegoOps.h"

int main(int argc, char **argv) {
  // mlir::registerAllPasses();
  // Register minimal passes
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  // Register LEGO passes
  mlir::lego::registerLegoToArithPass();

  mlir::DialectRegistry registry;
  // mlir::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::transform::TransformDialect,
                  mlir::math::MathDialect>();
  registry.insert<mlir::lego::LegoDialect>();
  
  // Register the transform dialect extension if needed, 
  // currently we just register the dialect which includes the ops.

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LEGO Magic Optimizer Driver", registry));
}
