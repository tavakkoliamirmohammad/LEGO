# PIC is required — targets get linked into nanobind shared modules.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_BINARY_DIR}/include)

# --- Z3 solver (needed by SMT verification passes, skip for Emscripten) ---
if(NOT EMSCRIPTEN)
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
endif()

# --- CPU vector pipeline sources (always built) ---
set(_CPU_VECTOR_SOURCES
  LegoX86VectorPipeline.cpp
  LegoArmNeonPipeline.cpp
  LegoArmSvePipeline.cpp
)

# --- GPU backend sources (conditional on LLVM targets) ---
set(_GPU_BACKEND_SOURCES
  LegoSPIRVPipeline.cpp
  LegoLLVMSPIRVPipeline.cpp
  LegoNVVMPipeline.cpp
  LegoROCDLPipeline.cpp
  LegoXeVMPipeline.cpp
)
set(_GPU_BACKEND_LIBS)

if(LEGO_HAS_SPIRV)
  list(APPEND _GPU_BACKEND_LIBS
    MLIRGPUToLLVMSPV
  )
endif()

if(LEGO_HAS_NVPTX)
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
  list(APPEND _GPU_BACKEND_LIBS
    MLIRGPUToROCDLTransforms
    MLIRROCDLDialect
    MLIRROCDLTarget
    MLIRROCDLToLLVMIRTranslation
  )
endif()

if(LEGO_HAS_XEVM)
  list(APPEND _GPU_BACKEND_LIBS
    MLIRXeVMDialect
    MLIRXeVMTarget
    MLIRXeVMToLLVMIRTranslation
  )
endif()

# LLVM target components (must use LINK_COMPONENTS, not LINK_LIBS)
set(_LLVM_COMPONENTS)
if(EMSCRIPTEN)
  # The WASM driver (lego_driver) translates LLVM dialect -> LLVM IR ->
  # wasm32 object code, so it needs the WebAssembly codegen backend.
  list(APPEND _LLVM_COMPONENTS
    WebAssemblyCodeGen WebAssemblyAsmParser WebAssemblyDesc WebAssemblyInfo
  )
endif()

# --- JIT engine (unavailable under Emscripten — requires dlopen/libffi) ---
if(NOT EMSCRIPTEN)
  set(_JIT_SOURCES LegoJITEngine.cpp)
  set(_JIT_LIBS MLIRExecutionEngine)
else()
  set(_JIT_SOURCES)
  set(_JIT_LIBS)
endif()

# --- Core dialect library ---
add_mlir_dialect_library(MLIRLego
  PARTIAL_SOURCES_INTENDED
  # Dialect core
  LegoDialect.cpp
  LegoOps.cpp
  # Passes
  LegoNormalization.cpp
  LegoToArith.cpp
  LegoArithSimplification.cpp
  LegoMaterializeAssumeBounds.cpp
  LegoAffineExtract.cpp
  LegoToLinalg.cpp
  LegoVectorizeAnalysis.cpp
  LegoVectorize.cpp
  LegoVectorizeCompact.cpp
  LegoVectorizeScatterAdd.cpp
  LegoVectorizeArgmin.cpp
  Passes.cpp
  # Verification
  LegoGenerateBoundsChecks.cpp
  LegoExternalSMTVerifier.cpp
  LegoVerifyBijectivity.cpp
  LegoVerify.cpp
  SMTUtils.cpp
  # JIT (excluded for Emscripten)
  ${_JIT_SOURCES}
  # CPU vector backends (always built)
  ${_CPU_VECTOR_SOURCES}
  # GPU backends (conditional)
  ${_GPU_BACKEND_SOURCES}

  DEPENDS
  MLIRLegoDialectIncGen
  MLIRLegoOpsIncGen
  MLIRLegoPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRVectorToLLVMPass
  # Linalg dialect + transforms (used by convert-lego-to-linalg + upstream
  # linalg::vectorize as the affine-case fast path).
  MLIRLinalgDialect
  MLIRLinalgTransforms
  # SMT verification
  MLIRSMT
  MLIRExportSMTLIB
  # LLVM lowering pipeline
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRControlFlowDialect
  MLIRLLVMDialect
  # JIT engine (excluded for Emscripten)
  ${_JIT_LIBS}
  # IR translation (needed by JIT pipeline)
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

  LINK_COMPONENTS
  ${_LLVM_COMPONENTS}
)

add_subdirectory(CAPI)
