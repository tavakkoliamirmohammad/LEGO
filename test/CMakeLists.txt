
llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_EXAMPLES
  )

set(LEGO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LEGO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)
set(LEGO_TOOLS_DIR ${CMAKE_BINARY_DIR}/tools/lego-opt)

# In a monolithic build LLVM_TOOLS_DIR may be empty; always ensure it
# points at the llvm-build binary directory for FileCheck, count, not.
if(NOT LLVM_TOOLS_DIR OR LLVM_TOOLS_DIR STREQUAL "")
  set(LLVM_TOOLS_DIR "${CMAKE_BINARY_DIR}/llvm-build/bin")
endif()
# Also set LLVM_TOOLS_BINARY_DIR which configure_lit_site_cfg reads
if(NOT LLVM_TOOLS_BINARY_DIR OR LLVM_TOOLS_BINARY_DIR STREQUAL "")
  set(LLVM_TOOLS_BINARY_DIR "${CMAKE_BINARY_DIR}/llvm-build/bin")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

# Check for lit in system
if(NOT LLVM_EXTERNAL_LIT)
  find_program(LLVM_LIT_PROGRAM lit)
  if(LLVM_LIT_PROGRAM)
    set(LLVM_EXTERNAL_LIT ${LLVM_LIT_PROGRAM} CACHE STRING "Command to run lit" FORCE)
  endif()
endif()

set(LEGO_TEST_DEPENDS lego-opt FileCheck count not)

# Add Python binding target as a dependency when available
if(TARGET LegoPythonModules)
  list(APPEND LEGO_TEST_DEPENDS LegoPythonModules)
endif()

# CUDA runner tools (only needed when runners are enabled in monolithic build)
if(LEGO_ENABLE_RUNNERS AND LEGO_MONOLITHIC_LLVM)
  list(APPEND LEGO_TEST_DEPENDS mlir-opt mlir_cuda_runtime mlir_c_runner_utils mlir-runner)
endif()

add_lit_testsuite(check-lego "Running the Lego regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${LEGO_TEST_DEPENDS}
  )

# Python tests (pytest) — runs after wheel directory is prepared
if(TARGET LegoPythonModules)
  add_custom_target(check-lego-python
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python_packages/lego"
      "${Python_EXECUTABLE}" -P -m pytest
        "${CMAKE_SOURCE_DIR}/python/tests" -v
        --import-mode=importlib
        --rootdir "${CMAKE_BINARY_DIR}"
    DEPENDS lego-prepare-wheel
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/lego"
    COMMENT "Running LEGO Python tests"
    USES_TERMINAL
  )
  if(TARGET lego-naga)
    add_dependencies(check-lego-python lego-naga)
  endif()
endif()
