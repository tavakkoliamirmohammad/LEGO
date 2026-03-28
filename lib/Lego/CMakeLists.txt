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
  # SPIR-V backend
  LegoSPIRVPipeline.cpp

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
  # GPU dialect (always available, not gated on NVPTX)
  MLIRGPUDialect
  MLIRGPUTransforms
  # SPIR-V backend
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSPIRVTransforms
  MLIRSPIRVUtils
  MLIRSPIRVSerialization
  MLIRGPUToSPIRV
  MLIRArithToSPIRV
  MLIRFuncToSPIRV
  MLIRSCFToSPIRV
  MLIRMemRefToSPIRV
  MLIRMathToSPIRV
  MLIRIndexToSPIRV
  # NVVM backend (CUDA, guarded by NVPTX target availability)
  MLIRGPUToNVVMTransforms
  MLIRNVVMDialect
  MLIRNVVMTarget
  MLIRNVVMToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  # Index to LLVM (needed for NVVM host lowering)
  MLIRIndexToLLVM
)

add_subdirectory(CAPI)
