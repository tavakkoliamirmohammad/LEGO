# PIC is required — targets get linked into nanobind shared modules.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_BINARY_DIR}/include)

# --- Z3 solver (needed by SMT verification passes) ---
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(PYTHON_BIN_DIR "${Python3_EXECUTABLE}" DIRECTORY)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import z3"
  RESULT_VARIABLE Z3_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)

if(NOT Z3_IMPORT_RESULT EQUAL 0)
  message(STATUS "z3-solver not found, installing...")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install z3-solver
    RESULT_VARIABLE PIP_RESULT)
  if(NOT PIP_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to install z3-solver")
  endif()
endif()

find_program(Z3_BIN_PATH NAMES z3 HINTS "${PYTHON_BIN_DIR}" REQUIRED)
message(STATUS "Using Z3: ${Z3_BIN_PATH}")
add_definitions(-DLEGO_Z3_EXECUTABLE="${Z3_BIN_PATH}")

# --- GPU backend sources (conditional on LLVM targets) ---
set(_GPU_BACKEND_SOURCES LegoSPIRVPipeline.cpp LegoLLVMSPIRVPipeline.cpp)
set(_GPU_BACKEND_LIBS)

if(LEGO_HAS_SPIRV)
  list(APPEND _GPU_BACKEND_LIBS
    MLIRGPUToLLVMSPV
  )
endif()

if(LEGO_HAS_NVPTX)
  list(APPEND _GPU_BACKEND_SOURCES LegoNVVMPipeline.cpp)
  list(APPEND _GPU_BACKEND_LIBS
    MLIRGPUToNVVMTransforms
    MLIRNVVMDialect
    MLIRNVVMTarget
    MLIRNVVMToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToLLVMIRTranslation
  )
endif()

if(LEGO_HAS_AMDGPU)
  list(APPEND _GPU_BACKEND_SOURCES LegoROCDLPipeline.cpp)
  list(APPEND _GPU_BACKEND_LIBS
    MLIRGPUToROCDLTransforms
    MLIRROCDLDialect
    MLIRROCDLTarget
    MLIRROCDLToLLVMIRTranslation
  )
endif()

if(LEGO_HAS_WASM)
  list(APPEND _GPU_BACKEND_SOURCES LegoWasmPipeline.cpp)
  list(APPEND _GPU_BACKEND_LIBS
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
  )
endif()

# --- Core dialect library ---
add_mlir_dialect_library(MLIRLego
  # Dialect core
  LegoDialect.cpp
  LegoOps.cpp
  # Passes
  LegoNormalization.cpp
  LegoToArith.cpp
  LegoArithSimplification.cpp
  LegoMaterializeAssumeBounds.cpp
  Passes.cpp
  # Verification
  LegoGenerateBoundsChecks.cpp
  LegoExternalSMTVerifier.cpp
  LegoVerifyBijectivity.cpp
  LegoVerifyCoalescing.cpp
  LegoVerifyBankConflicts.cpp
  SMTUtils.cpp
  # JIT
  LegoJITEngine.cpp
  # GPU backends (conditional)
  ${_GPU_BACKEND_SOURCES}

  DEPENDS
  MLIRLegoDialectIncGen
  MLIRLegoOpsIncGen
  MLIRLegoPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  # SMT verification
  MLIRSMT
  MLIRExportSMTLIB
  # LLVM lowering pipeline
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRControlFlowDialect
  MLIRLLVMDialect
  # JIT engine
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation
  MLIRBuiltinToLLVMIRTranslation
  # GPU dialect (always available)
  MLIRGPUDialect
  MLIRGPUTransforms
  # SPIR-V backend (always available)
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRSPIRVUtils
  MLIRSPIRVSerialization
  MLIRSPIRVTarget
  MLIRGPUToSPIRV
  MLIRArithToSPIRV
  MLIRFuncToSPIRV
  MLIRSCFToSPIRV
  MLIRMemRefToSPIRV
  MLIRMathToSPIRV
  MLIRIndexToSPIRV
  # GPU backends (conditional on LLVM targets)
  ${_GPU_BACKEND_LIBS}
  # Index to LLVM (needed for GPU host lowering)
  MLIRIndexToLLVM
)

add_subdirectory(CAPI)
