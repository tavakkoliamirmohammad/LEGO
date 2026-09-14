#!/usr/bin/env bash
# Assemble a directory shaped like a standard LLVM build (bin/mlir-runner, lib/libmlir_*.so)
# from the monolithic build tree, for the puzzles' MLIR_BUILD_DIR convention.
set -euo pipefail
B=/scratch/general/vast/u1419116/LEGO/build
R=$B/mlir-runtime
mkdir -p $R/bin $R/lib
ln -sf "$(find $B/llvm-build -name mlir-runner -type f | head -1)" $R/bin/mlir-runner
for lib in libmlir_cuda_runtime.so libmlir_runner_utils.so libmlir_c_runner_utils.so libmlir_async_runtime.so; do
  src=$(find $B/llvm-build -name "$lib" -type f | head -1 || true)
  [ -n "$src" ] && ln -sf "$src" $R/lib/$lib || echo "missing: $lib"
done
ls -la $R/bin $R/lib
