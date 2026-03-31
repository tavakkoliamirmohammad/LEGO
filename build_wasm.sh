#!/usr/bin/env bash
# Build the LEGO compiler as a WebAssembly module for browser visualization.
#
# Prerequisites:
#   1. A native build with LEGO_MONOLITHIC_LLVM=ON (provides tblgen tools)
#   2. Emscripten SDK (emsdk) installed and activated
#
# Usage:
#   ./build_wasm.sh                    # uses defaults
#   EMSDK=/path/to/emsdk ./build_wasm.sh
#   NATIVE_BUILD=build ./build_wasm.sh # custom native build dir
#
# Output:
#   viz/wasm/lego_driver.js   — JS glue (Emscripten module loader)
#   viz/wasm/lego_driver.wasm — WASM binary (full LEGO compiler)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NATIVE_BUILD="${NATIVE_BUILD:-${SCRIPT_DIR}/build}"
WASM_BUILD="${SCRIPT_DIR}/build-wasm"
EMSDK="${EMSDK:-${SCRIPT_DIR}/../emsdk}"
NATIVE_BIN="${NATIVE_BUILD}/llvm-build/bin"

# --- Verify native tblgen tools ---
for tool in llvm-tblgen mlir-tblgen; do
  if [[ ! -x "${NATIVE_BIN}/${tool}" ]]; then
    echo "ERROR: ${tool} not found at ${NATIVE_BIN}/"
    echo "Build the native compiler first:"
    echo "  cmake -B build -G Ninja -DLEGO_MONOLITHIC_LLVM=ON"
    echo "  cmake --build build"
    exit 1
  fi
done
echo "Native tblgen tools: ${NATIVE_BIN}"

# --- Activate Emscripten ---
if [[ ! -f "${EMSDK}/emsdk_env.sh" ]]; then
  echo "ERROR: Emscripten SDK not found at ${EMSDK}"
  echo "Set EMSDK env variable or install emsdk."
  exit 1
fi
source "${EMSDK}/emsdk_env.sh"
echo "Emscripten: $(emcc --version | head -1)"

# --- Configure ---
echo ""
echo "=== Configuring WASM build ==="
emcmake cmake -B "${WASM_BUILD}" -G Ninja \
  -DLEGO_MONOLITHIC_LLVM=ON \
  -DLLVM_NATIVE_TOOL_DIR="${NATIVE_BIN}" \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DLEGO_BUILD_TESTS=OFF \
  -Wno-dev

# --- Build ---
echo ""
echo "=== Building lego_driver (this will take a while) ==="
cmake --build "${WASM_BUILD}" --target lego_driver -- -j"$(nproc)"

echo ""
echo "=== WASM build complete ==="
ls -lh "${SCRIPT_DIR}/viz/wasm/lego_driver.js" "${SCRIPT_DIR}/viz/wasm/lego_driver.wasm" 2>/dev/null || true
