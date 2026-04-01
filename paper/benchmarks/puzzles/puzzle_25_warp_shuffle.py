"""Puzzle 25 — Warp Communication: shuffle_down and broadcast patterns.

Two sub-tests matching the Mojo puzzle:
  1. neighbor_difference: output[i] = input[i+1] - input[i] via shuffle_down
  2. moving_average_3: 3-point moving average via shuffle_down with offsets 1,2

Mojo also has broadcast sub-tests (basic, conditional, coordination) which
require the `broadcast()` intrinsic — not yet in the DSL. The shuffle_down
tests are the core of this puzzle.
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

# --- Sub-test 1: neighbor_difference ---
# output[i] = input[i+1] - input[i], last lane = 0


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def neighbor_difference(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    next_val = shuffle_down(val, 1)
    if tx < WARP_SIZE - 1:
        Out[bx, tx] = next_val - val
    # Last lane: output stays 0 (initialized)

# --- Sub-test 2: moving_average_3 ---
# 3-point average: (val + val_next + val_next2) / 3
# Boundary: 2nd-to-last = avg of 2, last = raw value


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def moving_average_3(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    next1 = shuffle_down(val, 1)
    next2 = shuffle_down(val, 2)
    if tx < WARP_SIZE - 2:
        Out[bx, tx] = (val + next1 + next2) * 0.333333343267441
    if tx == WARP_SIZE - 2:
        Out[bx, tx] = (val + next1) * 0.5
    if tx == WARP_SIZE - 1:
        Out[bx, tx] = val


from bench_utils import run_benchmark

if __name__ == "__main__":

    # Sub-test 1: neighbor_difference
    def compute_expected_diff(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        out[:, :-1] = np.diff(a, axis=1)
        return out.ravel().astype(np.float32)

    print("Sub-test 1: neighbor_difference", file=sys.stderr)
    run_benchmark(
        neighbor_difference, compute_expected_diff,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"neighbor N={N}",
        init_mod=100,
        atol=1e-4,
    )

    # Sub-test 2: moving_average_3
    def compute_expected_avg(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            for i in range(WARP_SIZE):
                if i < WARP_SIZE - 2:
                    out[w, i] = (a[w, i] + a[w, i+1] + a[w, i+2]) / 3.0
                elif i == WARP_SIZE - 2:
                    out[w, i] = (a[w, i] + a[w, i+1]) / 2.0
                else:
                    out[w, i] = a[w, i]
        return out.ravel().astype(np.float32)

    print("Sub-test 2: moving_average_3", file=sys.stderr)
    run_benchmark(
        moving_average_3, compute_expected_avg,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"avg3 N={N}",
        init_mod=100,
        atol=1e-4,
    )
