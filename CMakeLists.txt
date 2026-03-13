cmake_minimum_required(VERSION 3.20.0)
project(lego-mlir)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build Options ---
option(LEGO_MONOLITHIC_LLVM "Build LLVM/MLIR as part of the project" OFF)
option(LEGO_ENABLE_CCACHE "Enable ccache for faster recompilation" ON)
option(LEGO_BUILD_TESTS "Build LEGO tests" ON)
set(LEGO_LLVM_TARGETS "X86;NVPTX" CACHE STRING "Semicolon-separated list of LLVM targets to build (e.g. X86;NVPTX;AArch64)")

# --- Python & Venv Support ---
# Prioritize the local virtual environment
set(LEGO_VENV_PATH "${CMAKE_SOURCE_DIR}/venv")
if(EXISTS "${LEGO_VENV_PATH}/bin/python3")
  set(Python3_EXECUTABLE "${LEGO_VENV_PATH}/bin/python3")
  set(Python3_ROOT_DIR "${LEGO_VENV_PATH}")
  message(STATUS "Prioritizing Python from venv: ${Python3_EXECUTABLE}")
endif()

# --- Fast Linker Support ---
if(NOT LLVM_USE_LINKER)
  find_program(LLD_PATH ld.lld)
  find_program(MOLD_PATH mold)
  
  # Check compiler version for mold support (GCC 12.1+ or Clang 13+)
  set(USE_MOLD_LINKER OFF)
  if(MOLD_PATH)
    if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.1") OR
       (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0"))
      set(USE_MOLD_LINKER ON)
    endif()
  endif()

  if(USE_MOLD_LINKER)
    set(LLVM_USE_LINKER "mold" CACHE STRING "Linker to use" FORCE)
    message(STATUS "Using fast linker: mold")
  elseif(LLD_PATH)
    set(LLVM_USE_LINKER "lld" CACHE STRING "Linker to use" FORCE)
    message(STATUS "Using fast linker: lld")
  else()
    message(STATUS "No fast linker (mold/lld) detected or supported by compiler. Using default.")
  endif()
endif()

# --- Ccache Support ---
if(LEGO_ENABLE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  endif()
endif()

# --- MLIR Configuration ---
if(LEGO_MONOLITHIC_LLVM)
  set(LLVM_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project")
  if(NOT EXISTS "${LLVM_SUBMODULE_DIR}")
    message(FATAL_ERROR "llvm-project submodule not found at ${LLVM_SUBMODULE_DIR}")
  endif()

  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
  set(LLVM_TARGETS_TO_BUILD "${LEGO_LLVM_TARGETS}" CACHE STRING "" FORCE)

  # Minimal LLVM/MLIR build: disable components LEGO does not need
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
  set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "" FORCE)
  set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "" FORCE)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  endif()

  add_subdirectory("${LLVM_SUBMODULE_DIR}/llvm" "llvm-build" EXCLUDE_FROM_ALL)

  set(MLIR_DIR "${LLVM_SUBMODULE_DIR}/mlir")
  set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}/llvm-build/tools/mlir")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_SUBMODULE_DIR}/llvm/cmake/modules")
  
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  include_directories(${LLVM_SUBMODULE_DIR}/llvm/include)
  include_directories(${CMAKE_BINARY_DIR}/llvm-build/include)
  include_directories(${MLIR_DIR}/include)
  include_directories(${MLIR_BINARY_DIR}/include)
else()
  # Decoupled build: Expect MLIR to be found via MLIR_DIR or system install
  find_package(MLIR CONFIG REQUIRED)
  find_package(LLVM CONFIG REQUIRED)
  
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)

  # Ensure TableGen can find MLIR and LLVM headers
  include_directories(${MLIR_INCLUDE_DIRS})
  include_directories(${LLVM_INCLUDE_DIRS})
  list(APPEND MLIR_TABLEGEN_FLAGS -I${MLIR_INCLUDE_DIRS})
  list(APPEND MLIR_TABLEGEN_FLAGS -I${LLVM_INCLUDE_DIRS})
endif()

set(MLIR_TABLEGEN_EXE mlir-tblgen)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(include/Lego)
add_subdirectory(lib/Lego)
add_subdirectory(tools/lego-opt)

# --- Python Bindings for LEGO Dialect ---
# Requires MLIR_ENABLE_BINDINGS_PYTHON=ON in the LLVM build.
if(MLIR_ENABLE_BINDINGS_PYTHON)
  # nanobind must be found before AddMLIRPython macros are used
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(nanobind CONFIG QUIET)
  if(NOT nanobind_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
      OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(nanobind CONFIG REQUIRED)
  endif()

  include(AddMLIRPython)

  # Declare top-level Python source group
  declare_mlir_python_sources(LegoPythonSources)
  declare_mlir_python_sources(LegoPythonSources.Dialects
    ADD_TO_PARENT LegoPythonSources)

  # Generate Python op bindings from ODS
  # LegoOps.td uses bare includes (e.g. "LegoDialect.td"), so add include/Lego
  # to the directory include paths (used by tablegen() for -I flags).
  include_directories(${CMAKE_SOURCE_DIR}/include/Lego)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT LegoPythonSources.Dialects
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python"
    TD_FILE lego/dialects/LegoOps.td
    SOURCES lego/dialects/lego_dialect.py
    DIALECT_NAME lego)

  # Build the nanobind extension for dialect registration
  declare_mlir_python_extension(LegoPythonSources.Extension
    MODULE_NAME _legoDialects
    ADD_TO_PARENT LegoPythonSources
    SOURCES python/cpp/LegoDialectPybind.cpp
    EMBED_CAPI_LINK_LIBS MLIRLegoCAPI)

  # Override the global MLIR_BINDINGS_PYTHON_NB_DOMAIN so that the buggy
  # ${MLIR_BINDINGS_PYTHON_NB_DOMAIN} (without ARG_ prefix) reference in
  # AddMLIRPython.cmake:add_mlir_python_extension also resolves to "lego".
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "lego")

  # Sources shared between the CAPI library and the module assembly step.
  set(_lego_python_sources
    # Core MLIR Python package (mlir.ir, mlir._mlir_libs, etc.)
    MLIRPythonSources.Core
    # Standard dialects used by the LEGO compiler
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.Dialects.scf
    MLIRPythonSources.Dialects.memref
    # MLIR JIT execution engine (mlir.execution_engine.ExecutionEngine)
    MLIRPythonSources.ExecutionEngine
    # LEGO dialect Python bindings
    LegoPythonSources
  )

  # Create a unified CAPI shared library.  Embeds all EMBED_CAPI_LINK_LIBS
  # collected from the declared sources plus explicit per-dialect CAPI libs
  # needed for dialect registration via load_all_available_dialects().
  add_mlir_python_common_capi_library(LegoPythonCAPI
    INSTALL_COMPONENT LegoPythonModules
    INSTALL_DESTINATION "python_packages/lego/mlir/_mlir_libs"
    OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/lego/mlir/_mlir_libs"
    RELATIVE_INSTALL_ROOT "../../../.."
    DECLARED_SOURCES
      ${_lego_python_sources}
    EMBED_LIBS
      MLIRCAPITransforms
  )

  # Assemble final Python package.
  add_mlir_python_modules(LegoPythonModules
    ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/lego/mlir"
    INSTALL_PREFIX "python_packages/lego/mlir"
    DECLARED_SOURCES
      ${_lego_python_sources}
    COMMON_CAPI_LINK_LIBS
      LegoPythonCAPI
    MLIR_BINDINGS_PYTHON_NB_DOMAIN lego)

  # Copy the generated _lego_ops_gen.py into the source tree so that
  # ``from lego.dialects.lego_dialect import RowOp, ...`` works at runtime.
  # The source-tree lego/dialects/ already has an _ods_common.py shim that
  # re-exports from mlir.dialects._ods_common.
  add_custom_command(
    TARGET LegoPythonModules POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/lego/dialects/_lego_ops_gen.py"
      "${CMAKE_SOURCE_DIR}/python/lego/dialects/_lego_ops_gen.py"
    COMMENT "Copying _lego_ops_gen.py to source tree for lego.dialects import")
endif()

# --- Tests (after Python bindings so check-lego-python can depend on LegoPythonModules) ---
if(LEGO_BUILD_TESTS)
  add_subdirectory(test)
endif()
