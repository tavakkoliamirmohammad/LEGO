# Portable environment for the CASTLE/LEGO benchmark suite. Source it from anywhere:
#   source /path/to/LEGO/scripts/env.sh
# Requirements: a Python env (PYTHON_ENV, default: ./venv or the active one) with torch, triton, cupy,
# z3-solver, sympy, numpy; a CUDA toolkit (CUDA_HOME, or nvcc on PATH); the LEGO build in $LEGO_BUILD
# (default ./build, built with cmake as in README.md). Everything else is derived from these.
LEGO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LEGO_ROOT
export LEGO_BUILD="${LEGO_BUILD:-$LEGO_ROOT/build}"
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH" && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Python: source tree first (python/lego/mlir must link to the built bindings, see setup-mlir-runtime-dir.sh),
# puzzles harness, then the assembled package as fallback.
export PYTHONPATH="$LEGO_ROOT/python:$LEGO_ROOT/python/examples/puzzles:$LEGO_BUILD/python_packages/lego:${PYTHONPATH:-}"
# mlir-runner + runtime libs in the standard bin/ lib/ shape (created by setup-mlir-runtime-dir.sh)
export MLIR_BUILD_DIR="${MLIR_BUILD_DIR:-$LEGO_BUILD/mlir-runtime}"
export LD_LIBRARY_PATH="$MLIR_BUILD_DIR/lib:$LD_LIBRARY_PATH"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LEGO_ROOT/.torch_ext}"
# Optional: a specific compiler toolchain for PyTorch inline extensions (must match the CUDA host compiler)
[ -n "${GCC_ROOT:-}" ] && export PATH="$GCC_ROOT/bin:$PATH" && export CC="$GCC_ROOT/bin/gcc" && export CXX="$GCC_ROOT/bin/g++" && export LD_LIBRARY_PATH="$GCC_ROOT/lib64:$LD_LIBRARY_PATH"
export PYTHON="${PYTHON:-$( [ -x "$LEGO_ROOT/venv/bin/python" ] && echo "$LEGO_ROOT/venv/bin/python" || command -v python3 )}"
