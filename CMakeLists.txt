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
if(LEGO_BUILD_TESTS)
  add_subdirectory(test)
endif()
