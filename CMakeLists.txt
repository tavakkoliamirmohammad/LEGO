cmake_minimum_required(VERSION 3.20.0)
project(lego-mlir)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Build Options ---
option(LEGO_MONOLITHIC_LLVM "Build LLVM/MLIR as part of the project" OFF)
option(LEGO_ENABLE_CCACHE "Enable ccache for faster recompilation" ON)
option(LEGO_BUILD_TESTS "Build LEGO tests" ON)
set(LEGO_LLVM_TARGETS "X86;NVPTX;AMDGPU;SPIRV" CACHE STRING "Semicolon-separated list of LLVM targets to build (e.g. X86;NVPTX;AMDGPU;SPIRV;AArch64)")

# --- Python & Venv Support ---
if(NOT EMSCRIPTEN)
  set(LEGO_VENV_PATH "${CMAKE_SOURCE_DIR}/venv" CACHE PATH "Path to Python venv")
  if(EXISTS "${LEGO_VENV_PATH}/bin/python3" AND NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${LEGO_VENV_PATH}/bin/python3")
    set(Python3_ROOT_DIR "${LEGO_VENV_PATH}")
    message(STATUS "Prioritizing Python from venv: ${Python3_EXECUTABLE}")
  endif()
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
  set(LLVM_SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/third_party/llvm-project")
  if(NOT EXISTS "${LLVM_SUBMODULE_DIR}")
    message(FATAL_ERROR "llvm-project submodule not found at ${LLVM_SUBMODULE_DIR}")
  endif()

  set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)

  # --- Emscripten cross-compilation ---
  # Cross-compiling LLVM/MLIR to WebAssembly requires pre-built native
  # tablegen tools (they run at build time to generate C++ from .td files).
  if(EMSCRIPTEN)
    if(NOT LLVM_NATIVE_TOOL_DIR)
      message(FATAL_ERROR
        "LLVM_NATIVE_TOOL_DIR is required for Emscripten cross-compilation.\n"
        "Point it at the bin/ directory of a native LLVM/MLIR build, e.g.:\n"
        "  -DLLVM_NATIVE_TOOL_DIR=${CMAKE_SOURCE_DIR}/build/llvm-build/bin")
    endif()
    # Only the WebAssembly backend is needed in the browser
    set(LEGO_LLVM_TARGETS "WebAssembly" CACHE STRING "" FORCE)
    # Disable features unavailable under Emscripten
    set(LLVM_ENABLE_THREADS OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "" FORCE)
    set(LLVM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLVM_BUILD_UTILS OFF CACHE BOOL "" FORCE)
  endif()

  set(LLVM_TARGETS_TO_BUILD "${LEGO_LLVM_TARGETS}" CACHE STRING "" FORCE)

  # LEGO_ENABLE_RUNNERS is auto-detected below; if the user sets it
  # explicitly we honour that and forward to the MLIR CUDA runner.
  if(LEGO_ENABLE_RUNNERS)
    set(MLIR_ENABLE_CUDA_RUNNER ON CACHE BOOL "" FORCE)
  endif()

  # Minimal LLVM/MLIR build: disable components LEGO does not need
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
  set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
  if(NOT DEFINED MLIR_ENABLE_BINDINGS_PYTHON)
    if(NOT EMSCRIPTEN)
      set(MLIR_ENABLE_BINDINGS_PYTHON ON CACHE BOOL "Enable MLIR Python bindings")
    else()
      set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "Enable MLIR Python bindings")
    endif()
  endif()
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
  message(STATUS "MLIR_ENABLE_BINDINGS_PYTHON (from MLIRConfig): ${MLIR_ENABLE_BINDINGS_PYTHON}")

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

# --- GPU dialect (always available) ---
add_definitions(-DLEGO_HAS_GPU)
message(STATUS "GPU + SPIR-V dialect support enabled")

# --- Host GPU detection ---
# Probe for GPUs on the host.  This determines whether to auto-enable
# GPU execution tests (Triton examples, CUDA runners, ROCm runners).

set(LEGO_HOST_HAS_NVIDIA_GPU OFF)
find_program(_NVIDIA_SMI nvidia-smi)
if(_NVIDIA_SMI)
  execute_process(
    COMMAND "${_NVIDIA_SMI}" --query-gpu=name --format=csv,noheader
    RESULT_VARIABLE _NVIDIA_SMI_RESULT
    OUTPUT_VARIABLE _GPU_NAME OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_NVIDIA_SMI_RESULT EQUAL 0 AND _GPU_NAME)
    set(LEGO_HOST_HAS_NVIDIA_GPU ON)
    message(STATUS "Detected NVIDIA GPU: ${_GPU_NAME}")
  endif()
endif()

set(LEGO_HOST_HAS_AMD_GPU OFF)
find_program(_ROCM_SMI rocm-smi)
if(_ROCM_SMI)
  execute_process(
    COMMAND "${_ROCM_SMI}" --showproductname
    RESULT_VARIABLE _ROCM_SMI_RESULT
    OUTPUT_VARIABLE _AMD_GPU_NAME OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_ROCM_SMI_RESULT EQUAL 0 AND _AMD_GPU_NAME)
    set(LEGO_HOST_HAS_AMD_GPU ON)
    # Extract just the first card name from rocm-smi output
    string(REGEX MATCH "Card series:[ \t]*([^\n]+)" _AMD_MATCH "${_AMD_GPU_NAME}")
    if(_AMD_MATCH)
      message(STATUS "Detected AMD GPU: ${CMAKE_MATCH_1}")
    else()
      message(STATUS "Detected AMD GPU (via rocm-smi)")
    endif()
  endif()
endif()

# --- LLVM target detection helper ---
macro(detect_llvm_target _target _msg)
  set(LEGO_HAS_${_target} OFF)
  if("${LLVM_TARGETS_TO_BUILD}" MATCHES "${_target}")
    set(LEGO_HAS_${_target} ON)
    add_definitions(-DLEGO_HAS_${_target})
    message(STATUS "${_msg}")
  endif()
endmacro()

detect_llvm_target(NVPTX  "NVPTX backend enabled — NVVM compilation support available")
detect_llvm_target(SPIRV  "SPIRV backend enabled — LLVM SPIR-V compilation support available")
detect_llvm_target(AMDGPU "AMDGPU backend enabled")

# Auto-enable runners when NVPTX target is built and a GPU is present,
# unless the user explicitly set LEGO_ENABLE_RUNNERS.
if(NOT DEFINED LEGO_ENABLE_RUNNERS)
  if(LEGO_HAS_NVPTX AND LEGO_HOST_HAS_NVIDIA_GPU)
    set(LEGO_ENABLE_RUNNERS ON)
    message(STATUS "Auto-enabled CUDA runners (NVPTX target + GPU detected)")
  else()
    set(LEGO_ENABLE_RUNNERS OFF)
  endif()
endif()


# For cross-compilation, use the native tablegen from LLVM_NATIVE_TOOL_DIR;
# for native builds, use the CMake target built by add_subdirectory.
if(EMSCRIPTEN AND LLVM_NATIVE_TOOL_DIR)
  set(MLIR_TABLEGEN_EXE "${LLVM_NATIVE_TOOL_DIR}/mlir-tblgen")
else()
  set(MLIR_TABLEGEN_EXE mlir-tblgen)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(include/Lego)
add_subdirectory(lib/Lego)

if(EMSCRIPTEN)
  # WASM build: only build the driver, skip lego-opt/Python/tests.
  # The resulting lego_driver.js + lego_driver.wasm contain the full
  # LEGO compiler with IR printing at every pipeline stage.
  add_executable(lego_driver viz/wasm/lego_driver.cpp)
  target_link_libraries(lego_driver PRIVATE MLIRLego)
  set_target_properties(lego_driver PROPERTIES SUFFIX ".js")
  target_link_options(lego_driver PRIVATE
    "SHELL:-s EXPORTED_FUNCTIONS=['_lego_compile','_lego_free','_malloc','_free']"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','stringToUTF8','lengthBytesUTF8']"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s INITIAL_MEMORY=67108864"
    "SHELL:-s STACK_SIZE=4194304"
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_NAME='LEGOCompiler'"
    "SHELL:-s ENVIRONMENT='web,node'"
    "SHELL:-s FILESYSTEM=0"
    "SHELL:-s DISABLE_EXCEPTION_CATCHING=1"
    -O2
  )
  add_custom_command(TARGET lego_driver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:lego_driver>
      ${CMAKE_SOURCE_DIR}/viz/wasm/lego_driver.js
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE_DIR:lego_driver>/lego_driver.wasm
      ${CMAKE_SOURCE_DIR}/viz/wasm/lego_driver.wasm
    COMMENT "Copying WASM driver to viz/wasm/"
  )

  # WASM integration tests (Node.js >= 16 required for BigInt + WASM).
  # Use Emscripten's bundled Node to avoid the ancient system Node.
  set(NODE_BIN "$ENV{EMSDK_NODE}")
  if(NODE_BIN)
    add_custom_target(check-lego-wasm
      COMMAND ${NODE_BIN}
        ${CMAKE_SOURCE_DIR}/test/wasm/test_wasm_codegen.mjs
        $<TARGET_FILE:lego_driver>
      DEPENDS lego_driver
      COMMENT "Running WASM codegen integration tests"
    )
  endif()

  return()
endif()

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

  # Relocate MLIR Python bindings under the lego.mlir namespace so that
  # imports become "from lego.mlir.ir import ..." instead of "from mlir.ir ...".
  # This prevents conflicts when other MLIR-based packages are installed.
  set(MLIR_PYTHON_PACKAGE_PREFIX "lego.mlir" CACHE STRING "" FORCE)
  add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

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
    TD_FILE lego/backend/dialects/LegoOps.td
    SOURCES lego/backend/dialects/lego_dialect.py
    DIALECT_NAME lego
    GEN_ENUM_BINDINGS)

  # Build the nanobind extension for dialect registration
  set(_lego_extension_capi_libs MLIRLegoCAPI MLIRCAPIGPU MLIRCAPISPIRV)
  declare_mlir_python_extension(LegoPythonSources.Extension
    MODULE_NAME _legoDialects
    ADD_TO_PARENT LegoPythonSources
    SOURCES python/cpp/LegoDialectPybind.cpp
    EMBED_CAPI_LINK_LIBS ${_lego_extension_capi_libs})

  # Override the global MLIR_BINDINGS_PYTHON_NB_DOMAIN so that the buggy
  # ${MLIR_BINDINGS_PYTHON_NB_DOMAIN} (without ARG_ prefix) reference in
  # AddMLIRPython.cmake:add_mlir_python_extension also resolves to "lego".
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "lego")

  # Sources shared between the CAPI library and the module assembly step.
  set(_lego_python_sources
    # Core MLIR Python package (lego.mlir.ir, lego.mlir._mlir_libs, etc.)
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

  # GPU dialect Python bindings (always available — needed for SPIR-V
  # cross-compilation, does not require GPU hardware)
  list(APPEND _lego_python_sources
    MLIRPythonSources.Dialects.gpu
  )

  # Create a unified CAPI shared library.  Embeds all EMBED_CAPI_LINK_LIBS
  # collected from the declared sources plus explicit per-dialect CAPI libs
  # needed for dialect registration via load_all_available_dialects().
  add_mlir_python_common_capi_library(LegoPythonCAPI
    INSTALL_COMPONENT LegoPythonModules
    INSTALL_DESTINATION "python_packages/lego/lego/mlir/_mlir_libs"
    OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/lego/lego/mlir/_mlir_libs"
    RELATIVE_INSTALL_ROOT "../../../../.."
    DECLARED_SOURCES
      ${_lego_python_sources}
    EMBED_LIBS
      MLIRCAPITransforms
  )

  # Assemble final Python package.
  add_mlir_python_modules(LegoPythonModules
    ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/lego/lego/mlir"
    INSTALL_PREFIX "python_packages/lego/lego/mlir"
    DECLARED_SOURCES
      ${_lego_python_sources}
    COMMON_CAPI_LINK_LIBS
      LegoPythonCAPI
    MLIR_BINDINGS_PYTHON_NB_DOMAIN lego)

  # --- Type stub generation (requires nanobind stubgen) ---
  # Generates .pyi stubs for LEGO dialect extensions (LayoutType, ViewType,
  # register_lego_dialect).  Core MLIR stubs (ir.pyi, passmanager.pyi) are
  # not generated here due to a nanobind domain conflict in out-of-tree builds.
  option(LEGO_PYTHON_STUBGEN "Generate .pyi type stubs for LEGO extensions" ON)
  if(LEGO_PYTHON_STUBGEN)
    get_target_property(_lego_extension_srcs
      LegoPythonSources.Extension INTERFACE_SOURCES)
    mlir_generate_type_stubs(
      MODULE_NAME lego.mlir._mlir_libs._legoDialects
      DEPENDS_TARGETS
        LegoPythonModules.extension._mlir.dso
        LegoPythonModules.extension._legoDialects.dso
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
      OUTPUTS
        _legoDialects/__init__.pyi
        _legoDialects/lego.pyi
      DEPENDS_TARGET_SRC_DEPS "${_lego_extension_srcs}"
      IMPORT_PATHS
        "${CMAKE_BINARY_DIR}/python_packages/lego"
    )
    declare_mlir_python_sources(LegoPythonSources.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT LegoPythonSources
      SOURCES
        _mlir_libs/_legoDialects/__init__.pyi
        _mlir_libs/_legoDialects/lego.pyi)
    add_dependencies(LegoPythonModules "${NB_STUBGEN_CUSTOM_TARGET}")

    # Copy generated stubs into the assembled package so they ship with
    # the wheel and are visible to IDEs via PYTHONPATH.
    set(_stub_src "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs/_legoDialects")
    set(_stub_dst "${CMAKE_BINARY_DIR}/python_packages/lego/lego/mlir/_mlir_libs/_legoDialects")
    add_custom_command(TARGET LegoPythonModules POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_stub_dst}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_stub_src}/__init__.pyi" "${_stub_dst}/__init__.pyi"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_stub_src}/lego.pyi" "${_stub_dst}/lego.pyi"
      COMMENT "Copying LEGO type stubs into assembled package")
  endif()

  # Copy generated _lego_ops_gen.py into source tree for dev imports.
  # This file is .gitignored and not tracked in version control.
  add_custom_command(
    TARGET LegoPythonModules POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/lego/backend/dialects/_lego_ops_gen.py"
      "${CMAKE_SOURCE_DIR}/python/lego/backend/dialects/_lego_ops_gen.py"
    COMMENT "Copying _lego_ops_gen.py to source tree")

  # --- Wheel preparation target ---
  set(LEGO_WHEEL_DIR "${CMAKE_BINARY_DIR}/python_packages/lego")

  add_custom_target(lego-prepare-wheel
    DEPENDS LegoPythonModules
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_SOURCE_DIR}/python/lego" "${LEGO_WHEEL_DIR}/lego"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_SOURCE_DIR}/python/pyproject.toml" "${LEGO_WHEEL_DIR}/pyproject.toml"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_SOURCE_DIR}/README.md" "${LEGO_WHEEL_DIR}/README.md"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_SOURCE_DIR}/python/setup_wheel.py" "${LEGO_WHEEL_DIR}/setup.py"
    COMMENT "Preparing wheel directory"
  )

  # --- Naga CLI (SPIR-V → WGSL / MSL / GLSL converter) ---
  # Build naga-cli via cargo and place the binary where _find_naga_binary() looks.
  # Must come after lego-prepare-wheel because copy_directory overwrites bin/.
  find_program(CARGO_EXECUTABLE cargo)
  set(LEGO_NAGA_BIN "${LEGO_WHEEL_DIR}/lego/bin/naga")
  if(CARGO_EXECUTABLE)
    add_custom_command(
      OUTPUT "${LEGO_NAGA_BIN}"
      COMMAND "${CARGO_EXECUTABLE}" install naga-cli
        --root "${CMAKE_BINARY_DIR}/naga-install"
        --quiet
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_BINARY_DIR}/naga-install/bin/naga"
        "${LEGO_NAGA_BIN}"
      COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_BINARY_DIR}/naga-install/bin/naga"
        "${CMAKE_SOURCE_DIR}/python/lego/bin/naga"
      COMMENT "Building naga-cli via cargo"
    )
    add_custom_target(lego-naga DEPENDS "${LEGO_NAGA_BIN}")
    add_dependencies(lego-naga lego-prepare-wheel)
    message(STATUS "cargo found — naga-cli will be built automatically at ${LEGO_NAGA_BIN}")
  else()
    message(STATUS "cargo not found — naga-cli will not be built (some tests will be skipped)")
  endif()

  # --- Import verification target ---
  # After lego-prepare-wheel the assembled package at python_packages/lego/
  # contains both lego/* (source) and lego/mlir/* (MLIR bindings), so a
  # single PYTHONPATH entry is sufficient.  The -P flag prevents Python from
  # adding the script's directory (python/) to sys.path, which would shadow
  # the assembled lego/ package with the source-tree version.
  set(LEGO_PYTHONPATH "${CMAKE_BINARY_DIR}/python_packages/lego")
  add_custom_target(check-lego-imports
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${LEGO_PYTHONPATH}"
      "${Python_EXECUTABLE}" -P "${CMAKE_SOURCE_DIR}/python/check_imports.py"
    DEPENDS lego-prepare-wheel
    COMMENT "Verifying LEGO Python imports"
    USES_TERMINAL
  )

  # --- Example / benchmark target helper ---
  # Creates individual check-<prefix>-<name> targets and an umbrella
  # check-<prefix>-examples target that depends on all of them.
  macro(add_example_group _prefix _dir _env_vars)
    add_custom_target(check-${_prefix}-examples
      COMMENT "Running all ${_prefix} examples"
    )
    foreach(_entry ${ARGN})
      # Each entry is "script.py" or "script.py:arg1:arg2"
      string(REPLACE ":" ";" _parts "${_entry}")
      list(GET _parts 0 _script)
      list(REMOVE_AT _parts 0)
      string(REPLACE ".py" "" _name "${_script}")
      add_custom_target(check-${_prefix}-${_name}
        COMMAND ${CMAKE_COMMAND} -E env
          "PYTHONPATH=${LEGO_PYTHONPATH}"
          ${_env_vars}
          "${Python_EXECUTABLE}" "${_dir}/${_script}" ${_parts}
        DEPENDS lego-prepare-wheel
        COMMENT "Running ${_prefix}: ${_script} ${_parts}"
        USES_TERMINAL
      )
      add_dependencies(check-${_prefix}-examples check-${_prefix}-${_name})
    endforeach()
  endmacro()

  # --- Triton GPU example tests (require NVPTX + GPU + triton module) ---
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import triton"
    RESULT_VARIABLE _TRITON_RESULT OUTPUT_QUIET ERROR_QUIET)
  if(LEGO_HAS_NVPTX AND _TRITON_RESULT EQUAL 0)
    add_example_group(triton "${CMAKE_SOURCE_DIR}/python/examples/triton" ""
      vecadd_lego.py
      matmul_lego.py
      softmax_lego.py
      layernorm_lego.py
      grouped_gemm_lego.py
    )
  endif()

  # MLIR_BUILD_DIR is needed by bench_utils.py to locate mlir-runner.
  get_filename_component(_MLIR_BUILD_DIR "${LLVM_DIR}/../../.." ABSOLUTE)

  # --- LEGO GPU examples (require NVPTX + GPU + runners) ---
  if(LEGO_ENABLE_RUNNERS AND LEGO_HOST_HAS_NVIDIA_GPU)
    add_example_group(transpose "${CMAKE_SOURCE_DIR}/paper/benchmarks/mlir"
      "MLIR_BUILD_DIR=${_MLIR_BUILD_DIR}"
      transpose_naive.py:1024:1024
      transpose_smem.py:1024:1024
    )

    add_example_group(gpu "${CMAKE_SOURCE_DIR}/paper/benchmarks/mlir"
      "MLIR_BUILD_DIR=${_MLIR_BUILD_DIR}"
      vecadd.py:1024
      matmul_naive.py:64
      matmul_smem.py:64
      reduce_sum.py:1024
    )

    # --- Mojo GPU Puzzles expressed in LEGO DSL ---
    add_example_group(puzzle "${CMAKE_SOURCE_DIR}/python/examples/puzzles"
      "MLIR_BUILD_DIR=${_MLIR_BUILD_DIR}"
      puzzle_01_map.py:256
      puzzle_02_zip.py:256
      puzzle_03_guards.py:20
      puzzle_04_2dmap.py:4
      puzzle_05_broadcast.py:4
      puzzle_06_blocks.py:1024
      puzzle_07_2dblocks.py:256
      puzzle_08_shared_mem.py:256
      puzzle_09_debugging.py:256
      puzzle_10_memory_errors.py:3
      puzzle_11_pooling.py:8
      puzzle_12_dot_product.py:256
      puzzle_13_conv1d.py:15:4
      puzzle_14_prefix_sum.py:16
      puzzle_15_axis_sum.py:4:256
      puzzle_16_matmul.py:64
      puzzle_17_conv1d_op.py:15:4
      puzzle_18_softmax.py:4:16
      puzzle_19_attention.py:4:8
      puzzle_21_embedding.py:4:8
      puzzle_22_fusion.py:256
      puzzle_23_functional.py:256
      puzzle_24_warp_reduce.py:256
      puzzle_25_warp_shuffle.py:256
      puzzle_26_warp_advanced.py:256
      puzzle_27_block_reduce.py:256
      puzzle_28_async_copy.py:1024
      puzzle_29_pipeline.py:1024
      puzzle_32_bank_conflicts.py:64
      puzzle_33_tensor_cores.py:64
      puzzle_34_clusters.py:1024
    )
  endif()

  # --- Cross-platform SPIR-V GPU benchmarks (Vulkan/WebGPU/Metal via wgpu) ---
  # These run on any machine with wgpu installed — no NVIDIA/AMD GPU required.
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import wgpu"
    RESULT_VARIABLE _WGPU_RESULT OUTPUT_QUIET ERROR_QUIET)
  if(NOT _WGPU_RESULT EQUAL 0)
    message(STATUS "wgpu not found — SPIR-V GPU benchmarks will not be registered")
  endif()
  if(_WGPU_RESULT EQUAL 0)
  macro(add_spirv_benchmark _name _script)
    add_custom_target(check-spirv-${_name}
      COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${LEGO_PYTHONPATH}:${CMAKE_SOURCE_DIR}/paper/benchmarks/mlir"
        "MLIR_BUILD_DIR=${_MLIR_BUILD_DIR}"
        "${Python_EXECUTABLE}" -P
          "${CMAKE_SOURCE_DIR}/paper/benchmarks/mlir/${_script}" ${ARGN}
      DEPENDS lego-prepare-wheel
      COMMENT "Running SPIR-V GPU benchmark: ${_script} ${ARGN}"
      USES_TERMINAL
    )
  endmacro()

  add_spirv_benchmark(vecadd           vecadd.py           1024)
  add_spirv_benchmark(transpose_naive  transpose_naive.py  1024 1024)
  add_spirv_benchmark(matmul_naive     matmul_naive.py     64)
  add_spirv_benchmark(reduce_sum       reduce_sum.py       1024)
  add_spirv_benchmark(transpose_smem   transpose_smem.py   1024 1024)
  add_spirv_benchmark(matmul_smem      matmul_smem.py      64)

  add_custom_target(check-spirv-examples
    DEPENDS
      check-spirv-vecadd
      check-spirv-transpose_naive
      check-spirv-matmul_naive
      check-spirv-reduce_sum
      check-spirv-transpose_smem
      check-spirv-matmul_smem
    COMMENT "Running all SPIR-V GPU benchmarks (Vulkan/WebGPU/Metal)"
  )
  endif() # wgpu available

  # --- Wheel build + verification target ---
  add_custom_target(check-lego-wheel
    COMMAND ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${LEGO_PYTHONPATH}"
      "${Python_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/python/check_wheel.py"
    DEPENDS lego-prepare-wheel
    COMMENT "Building and verifying LEGO wheel"
    USES_TERMINAL
  )
endif()

# --- Tests (after Python bindings so check-lego-python can depend on LegoPythonModules) ---
if(LEGO_BUILD_TESTS)
  add_subdirectory(test)
endif()

# --- Umbrella target: run all checks ---
add_custom_target(check-lego-all
  COMMENT "Running all LEGO checks"
)
if(LEGO_BUILD_TESTS)
  add_dependencies(check-lego-all check-lego)
endif()
if(MLIR_ENABLE_BINDINGS_PYTHON)
  if(LEGO_BUILD_TESTS)
    add_dependencies(check-lego-all check-lego-python)
  endif()
  add_dependencies(check-lego-all check-lego-imports)
  add_dependencies(check-lego-all check-lego-wheel)
  if(TARGET check-triton-examples)
    add_dependencies(check-lego-all check-triton-examples)
  endif()
  if(TARGET check-transpose-examples)
    add_dependencies(check-lego-all check-transpose-examples)
  endif()
  if(TARGET check-gpu-examples)
    add_dependencies(check-lego-all check-gpu-examples)
  endif()
  if(TARGET check-puzzle-examples)
    add_dependencies(check-lego-all check-puzzle-examples)
  endif()
  if(TARGET check-spirv-examples)
    add_dependencies(check-lego-all check-spirv-examples)
  endif()
endif()

# --- WASM Viz Driver ---
# The WASM driver is built via Emscripten cross-compilation in a separate
# build directory.  See build_wasm.sh for the automated workflow:
#
#   ./build_wasm.sh
#
# This produces viz/wasm/lego_driver.js + viz/wasm/lego_driver.wasm
# containing the full LEGO compiler with IR printing support.
