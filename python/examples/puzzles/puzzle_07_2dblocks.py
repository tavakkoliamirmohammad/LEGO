# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 7 — 2D Blocks: add 10 to matrix using multiple 2D blocks.

LEGO layout: OrderBy(Row(N, N)).TileBy([N // TPB, N // TPB], [TPB, TPB])
  - 4-level indexing: A[block_y, block_x, thread_y, thread_x]

Mojo equivalent:
    var row = block_dim.y * block_idx.y + thread_idx.y
    var col = block_dim.x * block_idx.x + thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_07_2dblocks.py N")
    sys.exit(1)

N = int(sys.argv[1])
TPB = 16
assert N % TPB == 0

layout = OrderBy(Row(N, N)).TileBy([N // TPB, N // TPB], [TPB, TPB])


@gpu_kernel(grid=(N // TPB, N // TPB), block=(TPB, TPB))
def add_10_2d_blocks(A: Buffer(layout, N, N), Out: Buffer(layout, N, N)):
    by = block_id.y
    bx = block_id.x
    ty = thread_id.y
    tx = thread_id.x
    Out[by, bx, ty, tx] = A[by, bx, ty, tx] + 10.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + 10.0).astype(np.float32)

    run_benchmark(
        add_10_2d_blocks, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"{N}x{N}",
    )
