"""Puzzle 4 — 2D Map: add 10 to each element of a 2D matrix.

Uses 2D thread indexing (thread_id.x for col, thread_id.y for row).

Mojo equivalent:
    var row = thread_idx.y
    var col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[row * size + col] + 10.0

LEGO: Row(SIZE, SIZE) layout makes A[row, col] work directly.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_04_2dmap.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])

layout = Row(SIZE, SIZE)


@gpu_kernel(grid=(1,), block=(SIZE, SIZE))
def add_10_2d(A: Buffer(layout, SIZE, SIZE), Out: Buffer(layout, SIZE, SIZE)):
    row = thread_id.y
    col = thread_id.x
    Out[row, col] = A[row, col] + 10.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + 10.0).astype(np.float32)

    run_benchmark(
        add_10_2d, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{SIZE}x{SIZE}",
    )
