//===- SPIRVPlugin.cpp - SPIRV/Intel backend plugin for LEGO wheels -------===//
//
// Standalone shared library that registers lego-to-llvmspirv and
// lego-to-xevm pipelines.  Loaded via dlopen by lego_intel Python package.
//
//===----------------------------------------------------------------------===//

#include "Lego/Passes.h"
#ifdef LEGO_HAS_XEVM
#include "mlir/Target/LLVM/XeVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/XeVM/XeVMToLLVMIRTranslation.h"
#endif

extern "C" {

void legoPluginRegisterSPIRV() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
  mlir::lego::registerSPIRVPluginPipelines();
}

} // extern "C"
