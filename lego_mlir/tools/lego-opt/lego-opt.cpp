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
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Lego/LegoDialect.h"
#include "Lego/LegoOps.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  // TODO: Register lego passes here

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::lego::LegoDialect>();
  
  // Register the transform dialect extension if needed, 
  // currently we just register the dialect which includes the ops.

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LEGO Magic Optimizer Driver", registry));
}
