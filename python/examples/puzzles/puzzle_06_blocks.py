# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 1024
# REQUIRES: gpu
"""Puzzle 6 — Blocks: add 10 using multiple thread blocks.

LEGO layout: OrderBy(Row(N)).TileBy([N // TPB], [TPB])
  - block_id.x selects the block, thread_id.x the thread within.
  - Layout computes global_id = block_id.x * TPB + thread_id.x.

Mojo equivalent:
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < size:
        output[i] = a[i] + 10.0
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_06_blocks.py N")
    sys.exit(1)

N = int(sys.argv[1])
TPB = 256
assert N % TPB == 0

layout = OrderBy(Row(N)).TileBy([N // TPB], [TPB])


@gpu_kernel(grid=(N // TPB,), block=(TPB,))
def add_10_blocks(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    Out[bx, tx] = A[bx, tx] + 10.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + 10.0).astype(np.float32)

    run_benchmark(
        add_10_blocks, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"N={N}",
    )
