"""Puzzle 25 — Warp Communication: shuffle values between lanes.

Demonstrates shuffle_down for finite-difference style communication:
  output[i] = input[i+1] - input[i]  (last lane gets 0)

Uses shuffle_down(val, 1) to get the value from the next lane.

Mojo equivalent:
    var val = input[i]
    var next_val = shuffle_down(val, 1)
    if lane_id < WARP_SIZE - 1:
        output[i] = next_val - val
    else:
        output[i] = 0.0
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_25_warp_shuffle.py N")
    sys.exit(1)

N = int(sys.argv[1])
WARP_SIZE = 32
assert N % WARP_SIZE == 0
num_warps = N // WARP_SIZE

layout = OrderBy(Row(N)).TileBy([num_warps], [WARP_SIZE])


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def warp_diff(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    # Get value from the next lane (lane + 1)
    next_val = shuffle_down(val, 1)
    if tx < WARP_SIZE - 1:
        Out[bx, tx] = next_val - val
    # Last lane: difference is 0 (output buffer initialized to 0)


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        out[:, :-1] = np.diff(a, axis=1)
        return out.ravel().astype(np.float32)

    run_benchmark(
        warp_diff, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=100,
        atol=1e-4,
    )
