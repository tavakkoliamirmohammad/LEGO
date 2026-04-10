# Plugin shared libraries for split-wheel distribution.
#
# Each plugin is a self-contained .so that exports a single registration
# function.  It statically links only the GPU-specific MLIR libs and
# resolves MLIR/LLVM core symbols against the already-loaded
# LegoPythonCAPI (via RTLD_GLOBAL).
#
# These targets are ONLY needed for building plugin wheels.  The
# monolithic lego-opt and development builds do not use them.

if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
  return()
endif()

# --- NVPTX/CUDA plugin ---
if(LEGO_HAS_NVPTX)
  add_library(LegoNVPTXPlugin SHARED NVPTXPlugin.cpp)
  target_link_libraries(LegoNVPTXPlugin PRIVATE
    LegoPythonCAPI
    MLIRGPUToNVVMTransforms
    MLIRNVVMDialect
    MLIRNVVMTarget
    MLIRNVVMToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation
  )
  set_target_properties(LegoNVPTXPlugin PROPERTIES
    OUTPUT_NAME "LegoNVPTXPlugin"
    PREFIX "lib"
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
endif()

# --- AMDGPU/ROCm plugin ---
if(LEGO_HAS_AMDGPU)
  add_library(LegoAMDGPUPlugin SHARED AMDGPUPlugin.cpp)
  target_link_libraries(LegoAMDGPUPlugin PRIVATE
    LegoPythonCAPI
    MLIRGPUToROCDLTransforms
    MLIRROCDLDialect
    MLIRROCDLTarget
    MLIRROCDLToLLVMIRTranslation
  )
  set_target_properties(LegoAMDGPUPlugin PROPERTIES
    OUTPUT_NAME "LegoAMDGPUPlugin"
    PREFIX "lib"
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
endif()

# --- SPIRV/Intel plugin ---
if(LEGO_HAS_SPIRV)
  set(_SPIRV_PLUGIN_LIBS
    LegoPythonCAPI
    MLIRGPUToLLVMSPV
  )
  if(LEGO_HAS_XEVM)
    list(APPEND _SPIRV_PLUGIN_LIBS
      MLIRXeVMDialect
      MLIRXeVMTarget
      MLIRXeVMToLLVMIRTranslation
    )
  endif()

  add_library(LegoSPIRVPlugin SHARED SPIRVPlugin.cpp)
  target_link_libraries(LegoSPIRVPlugin PRIVATE ${_SPIRV_PLUGIN_LIBS})
  set_target_properties(LegoSPIRVPlugin PROPERTIES
    OUTPUT_NAME "LegoSPIRVPlugin"
    PREFIX "lib"
    SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}"
  )
endif()
