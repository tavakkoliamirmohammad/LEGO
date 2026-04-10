//===- NVPTXPlugin.cpp - NVPTX/CUDA backend plugin for LEGO wheels --------===//
//
// Standalone shared library that registers the lego-to-nvvm pipeline.
// Loaded via dlopen by lego_cuda Python package.  Resolves MLIR/LLVM core
// symbols against the already-loaded LegoPythonCAPI (RTLD_GLOBAL).
//
//===----------------------------------------------------------------------===//

#include "Lego/Passes.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"

extern "C" {

void legoPluginRegisterNVPTX() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
  mlir::lego::registerNVPTXPipelines();
}

} // extern "C"
