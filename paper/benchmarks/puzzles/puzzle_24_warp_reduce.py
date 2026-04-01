"""Puzzle 24 — Warp Fundamentals: reduce values using warp-level intrinsics.

Instead of shared memory + tree reduction, uses subgroup_reduce_add()
which maps to a single hardware instruction on modern GPUs.

Mojo equivalent:
    var val = input[i]
    var result = warp_sum(val)
    if lane_id == 0:
        output[block_idx.x] = result

LEGO: subgroup_reduce_add() wraps gpu.subgroup_reduce add.
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
def warp_reduce(A: Buffer(layout, N), Out: Buffer(out_layout, num_warps)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    # Warp-level reduction: all lanes contribute, result broadcast to all
    result = subgroup_reduce_add(val)
    # Only lane 0 writes the result
    if lane_id() == 0:
        Out[bx] = result


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return inputs[0].reshape(-1, WARP_SIZE).sum(axis=1).astype(np.float32)

    run_benchmark(
        warp_reduce, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-4,
    )
