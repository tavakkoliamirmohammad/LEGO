"""Puzzle 5 — Broadcast: add 1D vectors to form a 2D matrix.

output[row, col] = a[col] + b[row]

Mojo equivalent:
    var row = thread_idx.y
    var col = thread_idx.x
    if row < size and col < size:
        output[row * size + col] = a[col] + b[row]

LEGO: 1D vectors use Row(SIZE), 2D output uses Row(SIZE, SIZE).
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_05_broadcast.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])

vec_layout = Row(SIZE)
mat_layout = Row(SIZE, SIZE)


@gpu_kernel(grid=(1,), block=(SIZE, SIZE))
def broadcast_add(a: Buffer(vec_layout, SIZE), b: Buffer(vec_layout, SIZE),
                  out: Buffer(mat_layout, SIZE, SIZE)):
    row = thread_id.y
    col = thread_id.x
    out[row, col] = a[col] + b[row]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0][:SIZE]
        b = inputs[1][:SIZE]
        return (a[np.newaxis, :] + b[:, np.newaxis]).ravel().astype(np.float32)

    run_benchmark(
        broadcast_add, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{SIZE}x{SIZE}",
    )
