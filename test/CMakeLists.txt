
llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_EXAMPLES
  )

set(LEGO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(LEGO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/..)
set(LEGO_TOOLS_DIR ${CMAKE_BINARY_DIR}/tools/lego-opt)

# In a monolithic build these may be empty; ensure they point at the
# llvm-build binary directory for FileCheck, count, not.
foreach(_var LLVM_TOOLS_DIR LLVM_TOOLS_BINARY_DIR)
  if(NOT ${_var} OR ${_var} STREQUAL "")
    set(${_var} "${CMAKE_BINARY_DIR}/llvm-build/bin")
  endif()
endforeach()

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

# CUDA runner tools (only when user explicitly enabled MLIR CUDA runner)
if(MLIR_ENABLE_CUDA_RUNNER AND LEGO_MONOLITHIC_LLVM)
  list(APPEND LEGO_TEST_DEPENDS mlir-opt mlir_cuda_runtime mlir_c_runner_utils mlir-runner)
endif()

add_lit_testsuite(check-lego "Running the Lego regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${LEGO_TEST_DEPENDS}
  )

# --- Puzzle lit tests (GPU integration tests) ---
if(TARGET LegoPythonModules AND LEGO_ENABLE_RUNNERS)
  llvm_canonicalize_cmake_booleans(
    LEGO_HOST_HAS_NVIDIA_GPU LEGO_HOST_HAS_AMD_GPU LEGO_HOST_HAS_METAL
    LEGO_ENABLE_CUDA_RUNNER LEGO_ENABLE_ROCM_RUNNER LEGO_ENABLE_METAL_RUNNER)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/puzzles/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/puzzles/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/puzzles/lit.cfg.py
  )
  set(PUZZLE_TEST_DEPENDS lego-prepare-wheel)
  if(TARGET lego-naga)
    list(APPEND PUZZLE_TEST_DEPENDS lego-naga)
  endif()
  add_lit_testsuite(check-lego-puzzles "Running LEGO puzzle integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}/puzzles
    DEPENDS
    ${PUZZLE_TEST_DEPENDS}
  )
endif()

# --- Puzzle compile-only tests (no GPU required) ---
# Runs puzzles with LEGO_COMPILE_ONLY=1 to test cross-compilation to all
# GPU backends (CUDA PTX, ROCm, SPIR-V, Vulkan, WebGPU, Metal) without
# requiring GPU hardware.
if(TARGET LegoPythonModules)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/puzzles-compile/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/puzzles-compile/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/puzzles-compile/lit.cfg.py
  )
  set(PUZZLE_COMPILE_DEPENDS lego-prepare-wheel)
  if(TARGET lego-naga)
    list(APPEND PUZZLE_COMPILE_DEPENDS lego-naga)
  endif()
  add_lit_testsuite(check-lego-puzzles-compile
    "Running LEGO puzzle compilation tests (no GPU)"
    ${CMAKE_CURRENT_BINARY_DIR}/puzzles-compile
    DEPENDS
    ${PUZZLE_COMPILE_DEPENDS}
  )
endif()

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
