# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 24 — Warp Fundamentals: dot product using warp-level reduction.

Computes dot product of two vectors using subgroup_reduce_add (warp_sum).
Each thread computes a[i] * b[i], then the warp reduces the products.

Mojo equivalent:
    partial = a[i] * b[i]
    total = warp_sum(partial)
    if lane_id == 0:
        output[block_idx.x] = total
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_24_warp_reduce.py N")
    sys.exit(1)

N = int(sys.argv[1])
WARP_SIZE = 32
assert N % WARP_SIZE == 0
num_warps = N // WARP_SIZE

layout = OrderBy(Row(N)).TileBy([num_warps], [WARP_SIZE])
out_layout = Row(num_warps)


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def warp_dot_product(A: Buffer(layout, N), B: Buffer(layout, N),
                     Out: Buffer(out_layout, num_warps)):
    bx = block_id.x
    tx = thread_id.x
    # Each thread computes its partial product
    partial = A[bx, tx] * B[bx, tx]
    # Warp-level reduction: sum all partial products
    total = subgroup_reduce_add(partial)
    # Only lane 0 writes the result
    if lane_id() == 0:
        Out[bx] = total


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        b = inputs[1].reshape(-1, WARP_SIZE)
        return (a * b).sum(axis=1).astype(np.float32)

    run_benchmark(
        warp_dot_product, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "webgl", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-4,
    )
