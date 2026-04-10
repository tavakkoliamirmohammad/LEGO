//===- AMDGPUPlugin.cpp - AMDGPU/ROCm backend plugin for LEGO wheels -----===//
//
// Standalone shared library that registers the lego-to-rocdl pipeline.
// Loaded via dlopen by lego_rocm Python package.
//
//===----------------------------------------------------------------------===//

#include "Lego/Passes.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"

extern "C" {

void legoPluginRegisterAMDGPU() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;
  mlir::lego::registerAMDGPUPipelines();
}

} // extern "C"
