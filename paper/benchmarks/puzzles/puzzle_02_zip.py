"""Puzzle 2 — Zip: element-wise addition of two vectors.

LEGO layout: OrderBy(Row(N)).TileBy([N // WG], [WG])

Mojo equivalent:
    var i = thread_idx.x
    output[i] = a[i] + b[i]
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_02_zip.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0

layout = OrderBy(Row(N)).TileBy([N // WG], [WG])


@gpu_kernel(grid=(N // WG,), block=(WG,))
def zip_add(A: Buffer(layout, N), B: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    Out[bx, tx] = A[bx, tx] + B[bx, tx]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + inputs[1]).astype(np.float32)

    run_benchmark(
        zip_add, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
    )
