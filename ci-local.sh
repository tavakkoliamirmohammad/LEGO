#!/usr/bin/env bash
# ci-local.sh — mirrors the GitHub Actions 'build' job (ubuntu-22.04, py3.12)
# Run inside: apptainer exec --writable-tmpfs --fakeroot --cleanenv --bind /scratch \
#   /scratch/general/vast/u1419116/ubuntu_22.04.sif bash /scratch/general/vast/u1419116/LEGO/ci-local.sh
set -euo pipefail

REPO=/scratch/general/vast/u1419116/LEGO
BUILD="$REPO/build-ci-test"
PYTHON=python3.12
CI_VENV="$REPO/.ci-venv"
CHECKPOINT_DIR="$REPO/.ci-checkpoints"

mkdir -p "$CHECKPOINT_DIR"

# Checkpoint helper: skip step if already done (only for steps persisted in /scratch)
step_done()  { [ -f "$CHECKPOINT_DIR/$1" ]; }
mark_done()  { date > "$CHECKPOINT_DIR/$1"; echo "  ✓ checkpoint saved: $1"; }

# Clean slate for PATH
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# ══════════════════════════════════════════════════════════════════════════════
# Steps 1-2 & 4: ALWAYS run — these install into the container's tmpfs overlay
# which is lost between runs. They're fast with apt cache.
# ══════════════════════════════════════════════════════════════════════════════

echo "=== Step 1: Install system dependencies ==="
apt-get update -qq
apt-get install -y -qq ninja-build lld ccache cmake git curl \
  software-properties-common build-essential g++ patchelf > /dev/null

echo "=== Step 2: Install Python $PYTHON ==="
if ! command -v $PYTHON &>/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
fi
apt-get install -y -qq $PYTHON ${PYTHON}-venv ${PYTHON}-dev > /dev/null
echo "  Python: $($PYTHON --version)"

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Venv — persists in /scratch, so checkpoint works
# ══════════════════════════════════════════════════════════════════════════════

if step_done 03_venv && [ -f "$CI_VENV/bin/python" ]; then
  echo "=== Step 3: Python venv — SKIP (cached) ==="
else
  echo "=== Step 3: Create venv and install Python dependencies ==="
  rm -rf "$CI_VENV"
  $PYTHON -m venv "$CI_VENV"
  source "$CI_VENV/bin/activate"
  pip install -q nanobind sympy pytest lit numpy
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  rm -f "$CHECKPOINT_DIR/03_venv"
  mark_done 03_venv
fi
# Always activate the venv
source "$CI_VENV/bin/activate"

# ── Step 4: Rust + naga-cli (tmpfs — always check) ──
echo "=== Step 4: Install Rust + naga-cli ==="
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v naga &>/dev/null; then
  if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  cargo install naga-cli
fi
echo "  naga: $(naga --version 2>/dev/null || echo 'installed')"

# ══════════════════════════════════════════════════════════════════════════════
# Steps 5+: Build artifacts — persisted in /scratch, checkpoints work
# ══════════════════════════════════════════════════════════════════════════════

# ── Step 5: Configure CMake ──
NANOBIND_DIR=$(python -c 'import nanobind; print(nanobind.cmake_dir())')
PYTHON_BIN=$(which python)

if step_done 05_cmake_configure && [ -f "$BUILD/build.ninja" ]; then
  echo "=== Step 5: CMake configure — SKIP (cached) ==="
else
  echo "=== Step 5: Configure CMake ==="
  # Remove stale cache if it points to wrong Python
  if [ -f "$BUILD/CMakeCache.txt" ]; then
    cached_py=$(grep 'Python3_EXECUTABLE' "$BUILD/CMakeCache.txt" 2>/dev/null | head -1 || true)
    if [ -n "$cached_py" ] && ! echo "$cached_py" | grep -q ".ci-venv"; then
      echo "  Removing stale CMakeCache (wrong Python cached)"
      rm -f "$BUILD/CMakeCache.txt"
      rm -rf "$BUILD/CMakeFiles"
    fi
  fi
  cmake -G Ninja -B "$BUILD" -S "$REPO" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLEGO_MONOLITHIC_LLVM=ON \
    -DLEGO_LLVM_TARGETS="X86;NVPTX" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLEGO_BUILD_TESTS=ON \
    -DLLVM_USE_LINKER=lld \
    -DLEGO_VENV_PATH="$CI_VENV" \
    -DPython3_EXECUTABLE="$PYTHON_BIN" \
    -DPython_EXECUTABLE="$PYTHON_BIN" \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DPython_FIND_VIRTUALENV=ONLY \
    -Dnanobind_DIR="$NANOBIND_DIR" \
    -DLLVM_EXTERNAL_LIT="$(which lit)"
  rm -f "$CHECKPOINT_DIR/05_cmake_configure"
  mark_done 05_cmake_configure
fi

# ── Step 6: Build ──
if step_done 06_build; then
  echo "=== Step 6: Build — SKIP (cached) ==="
else
  echo "=== Step 6: Build all targets ==="
  cmake --build "$BUILD" -j8
  mark_done 06_build
fi

# ── Step 7: Tests ──
if step_done 07_tests; then
  echo "=== Step 7: Tests — SKIP (cached) ==="
else
  echo "=== Step 7: Run all tests ==="
  cmake --build "$BUILD" --target check-lego-all
  mark_done 07_tests
fi

# ── Step 8: Strip binaries ──
if step_done 08_strip; then
  echo "=== Step 8: Strip — SKIP (cached) ==="
else
  echo "=== Step 8: Resolve symlinks and strip binaries ==="
  cd "$BUILD/python_packages/lego"
  find . -type l | while read link; do
    target=$(python -c "import os; print(os.path.realpath('$link'))")
    if [ -e "$target" ]; then
      rm "$link"
      cp -a "$target" "$link"
    else
      rm "$link"
    fi
  done
  find . -name '*.so' -o -name '*.so.*' | xargs strip --strip-unneeded 2>/dev/null || true
  cd "$REPO"
  mark_done 08_strip
fi

# ── Step 9: Prepare wheel ──
if step_done 09_prepare_wheel; then
  echo "=== Step 9: Prepare wheel — SKIP (cached) ==="
else
  echo "=== Step 9: Prepare wheel ==="
  cmake --build "$BUILD" --target lego-prepare-wheel
  mark_done 09_prepare_wheel
fi

# ── Step 10: Bundle naga ──
if step_done 10_bundle_naga; then
  echo "=== Step 10: Bundle naga — SKIP (cached) ==="
else
  echo "=== Step 10: Bundle naga into wheel ==="
  mkdir -p "$BUILD/python_packages/lego/lego/bin"
  NAGA_BIN=$(which naga || echo "$HOME/.cargo/bin/naga")
  if [ -f "$NAGA_BIN" ]; then
    cp "$NAGA_BIN" "$BUILD/python_packages/lego/lego/bin/"
    chmod +x "$BUILD/python_packages/lego/lego/bin/naga"
    echo "Bundled naga: $(du -h "$BUILD/python_packages/lego/lego/bin/naga" | cut -f1)"
  else
    echo "WARNING: naga binary not found, wheel will not include naga"
  fi
  mark_done 10_bundle_naga
fi

# ── Step 11: Build wheel ──
if step_done 11_build_wheel; then
  echo "=== Step 11: Build wheel — SKIP (cached) ==="
else
  echo "=== Step 11: Build wheel ==="
  pip install -q wheel setuptools build auditwheel
  cd "$BUILD/python_packages/lego"
  rm -rf dist build
  pip wheel . --no-deps --use-pep517 -w dist
  for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat manylinux_2_35_x86_64 -w dist/
    rm "$whl"
  done
  ls -lh dist/
  cd "$REPO"
  mark_done 11_build_wheel
fi

# ── Step 12: Verify wheel ──
if step_done 12_verify_wheel; then
  echo "=== Step 12: Verify wheel — SKIP (cached) ==="
else
  echo "=== Step 12: Verify wheel installs ==="
  pip install --force-reinstall --no-deps --no-index --find-links "$BUILD/python_packages/lego/dist" lego-layout
  python -P "$REPO/python/check_imports.py"
  mark_done 12_verify_wheel
fi

echo ""
echo "========================================="
echo "=== BUILD + PACKAGE JOBS PASSED ==="
echo "========================================="

# ── WASM Viz Build ──

# ── Step 13: Emscripten (tmpfs — always check) ──
echo "=== WASM Step 1: Install Emscripten ==="
if [ ! -d /tmp/emsdk ] || ! command -v emcc &>/dev/null; then
  cd /tmp
  if [ ! -d emsdk ]; then
    git clone https://github.com/emscripten-core/emsdk.git
  fi
  cd emsdk
  git pull
  ./emsdk install latest
  ./emsdk activate latest
  cd "$REPO"
fi
source /tmp/emsdk/emsdk_env.sh

# ── Step 14: Native tblgen ──
WASM_NATIVE_BUILD="$REPO/build-wasm-native"
if step_done 14_wasm_native_tblgen && [ -f "$WASM_NATIVE_BUILD/llvm-build/bin/mlir-tblgen" ]; then
  echo "=== WASM Step 2: Native tblgen — SKIP (cached) ==="
else
  echo "=== WASM Step 2: Build native tblgen tools ==="
  # Clean stale cache that may have MLIR_ENABLE_BINDINGS_PYTHON=ON
  rm -f "$WASM_NATIVE_BUILD/CMakeCache.txt"
  rm -rf "$WASM_NATIVE_BUILD/CMakeFiles"
  cmake -G Ninja -B "$WASM_NATIVE_BUILD" -S "$REPO" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLEGO_MONOLITHIC_LLVM=ON \
    -DLEGO_LLVM_TARGETS="X86" \
    -DLLVM_USE_LINKER=lld \
    -DLEGO_BUILD_TESTS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DLEGO_VENV_PATH=/dev/null
  cmake --build "$WASM_NATIVE_BUILD" --target mlir-tblgen llvm-tblgen llvm-min-tblgen -j8
  mark_done 14_wasm_native_tblgen
fi

# ── Step 15: WASM configure ──
WASM_BUILD="$REPO/build-wasm"
if step_done 15_wasm_configure && [ -f "$WASM_BUILD/build.ninja" ]; then
  echo "=== WASM Step 3: WASM configure — SKIP (cached) ==="
else
  echo "=== WASM Step 3: Configure WASM build ==="
  emcmake cmake -B "$WASM_BUILD" -S "$REPO" -G Ninja \
    -DLEGO_MONOLITHIC_LLVM=ON \
    -DLLVM_NATIVE_TOOL_DIR="$WASM_NATIVE_BUILD/llvm-build/bin" \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLEGO_BUILD_TESTS=OFF \
    -Wno-dev
  mark_done 15_wasm_configure
fi

# ── Step 16: WASM build ──
if step_done 16_wasm_build; then
  echo "=== WASM Step 4: WASM build — SKIP (cached) ==="
else
  echo "=== WASM Step 4: Build WASM driver ==="
  cmake --build "$WASM_BUILD" --target lego_driver -j8
  mark_done 16_wasm_build
fi

# ── Step 17: WASM verify ──
if step_done 17_wasm_verify; then
  echo "=== WASM Step 5: WASM verify — SKIP (cached) ==="
else
  echo "=== WASM Step 5: Verify output ==="
  ls -lh "$REPO/viz/wasm/lego_driver.js" "$REPO/viz/wasm/lego_driver.wasm"
  echo "WASM size: $(du -h "$REPO/viz/wasm/lego_driver.wasm" | cut -f1)"
  mark_done 17_wasm_verify
fi

echo ""
echo "========================================="
echo "=== ALL CI STEPS PASSED ==="
echo "========================================="
