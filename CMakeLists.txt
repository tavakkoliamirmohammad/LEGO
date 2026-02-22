cmake_minimum_required(VERSION 3.20.0)
project(lego-mlir)

set(LLVM_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/third_party/llvm-project")
if(NOT EXISTS "${LLVM_SUBMODULE_DIR}")
  message(FATAL_ERROR "llvm-project submodule not found at ${LLVM_SUBMODULE_DIR}")
endif()

# Add the llvm-project submodule to the build
# This will build LLVM and MLIR as part of the project
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX" CACHE STRING "" FORCE)
add_subdirectory("${LLVM_SUBMODULE_DIR}/llvm" "llvm-build" EXCLUDE_FROM_ALL)

# Set the MLIR source and build directories
set(MLIR_DIR "${LLVM_SUBMODULE_DIR}/mlir")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}/llvm-build/tools/mlir")

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${LLVM_SUBMODULE_DIR}/llvm/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

include_directories(${LLVM_SUBMODULE_DIR}/llvm/include)
include_directories(${CMAKE_BINARY_DIR}/llvm-build/include)
include_directories(${MLIR_DIR}/include)
include_directories(${MLIR_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

link_directories(${MLIR_BINARY_DIR}/lib)

add_subdirectory(include/Lego)
add_subdirectory(lib/Lego)
add_subdirectory(tools/lego-opt)
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
