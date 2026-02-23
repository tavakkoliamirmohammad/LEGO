include_directories(${CMAKE_BINARY_DIR}/include)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Identify the z3 binary path relative to the Python interpreter
get_filename_component(PYTHON_BIN_DIR "${Python3_EXECUTABLE}" DIRECTORY)
set(Z3_BIN_PATH "${PYTHON_BIN_DIR}/z3")

# Check if z3 is available; if not, install z3-solver
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import z3; print(z3.get_version_string())"
    RESULT_VARIABLE Z3_IMPORT_RESULT
    OUTPUT_QUIET ERROR_QUIET
)

if(NOT Z3_IMPORT_RESULT EQUAL 0)
  message(STATUS "z3-solver not found in Python, installing...")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install z3-solver
    RESULT_VARIABLE PIP_INSTALL_RESULT
  )
  if(NOT PIP_INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to install z3-solver using ${Python3_EXECUTABLE}")
  endif()
endif()

# Fallback check for the z3 executable if it's not where we expect
if(NOT EXISTS "${Z3_BIN_PATH}")
    find_program(Z3_PROG_PATH NAMES z3 HINTS "${PYTHON_BIN_DIR}")
    if(Z3_PROG_PATH)
        set(Z3_BIN_PATH "${Z3_PROG_PATH}")
    else()
        message(FATAL_ERROR "z3 executable not found even after installing z3-solver. Looked in ${PYTHON_BIN_DIR}")
    endif()
endif()

message(STATUS "Using Z3 executable at: ${Z3_BIN_PATH}")
add_definitions(-DLEGO_Z3_EXECUTABLE="${Z3_BIN_PATH}")

add_mlir_dialect_library(MLIRLego
  LegoDialect.cpp
  LegoOps.cpp
  LegoDesugar.cpp
  LegoToArith.cpp
  LegoVerifyGenpConsistency.cpp
  LegoArithSimplification.cpp
  LegoGenerateBoundsChecks.cpp
  LegoExternalSMTVerifier.cpp
  SMTUtils.cpp
  Passes.cpp

  DEPENDS
  MLIRLegoDialectIncGen
  MLIRLegoOpsIncGen
  MLIRLegoPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSMT
  MLIRExportSMTLIB
)
