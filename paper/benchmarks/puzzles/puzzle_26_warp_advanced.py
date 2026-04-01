"""Puzzle 26 — Advanced Warp Patterns: XOR shuffle and butterfly reduction.

Three sub-tests matching the Mojo puzzle:
  1. pair_swap: shuffle_xor(val, 1) — swap adjacent pairs
  2. butterfly_max: XOR tree reduction for broadcast max
  3. butterfly_sum: XOR tree reduction for broadcast sum

Mojo also has conditional_max, prefix_sum, and partition sub-tests
which require min/max intrinsics and prefix_sum — these use the same
shuffle_xor primitive demonstrated here.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_26_warp_advanced.py N")
    sys.exit(1)

N = int(sys.argv[1])
WARP_SIZE = 32
assert N % WARP_SIZE == 0
num_warps = N // WARP_SIZE

layout = OrderBy(Row(N)).TileBy([num_warps], [WARP_SIZE])

# --- Sub-test 1: pair_swap ---
# shuffle_xor(val, 1): lane k gets value from lane k^1
# [0,1,2,3,...] -> [1,0,3,2,5,4,...]


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def pair_swap(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    swapped = shuffle_xor(val, 1)
    Out[bx, tx] = swapped

# --- Sub-test 2: butterfly_max ---
# XOR tree reduction: every lane gets the global max
# offset = WARP_SIZE//2, WARP_SIZE//4, ..., 1


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def butterfly_max(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    max_val = A[bx, tx]
    offset = WARP_SIZE // 2
    while offset > 0:
        other = shuffle_xor(max_val, offset)
        # max(a, b) = (a + b + |a - b|) / 2, but we don't have abs/max
        # Use: if other > max_val: max_val = other
        # scf.if with yield handles this correctly
        if other > max_val:
            max_val = other
        offset = offset // 2
    Out[bx, tx] = max_val

# --- Sub-test 3: butterfly_sum ---
# XOR tree reduction: every lane gets the full sum


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def butterfly_sum(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    mask = 1
    while mask < WARP_SIZE:
        other = shuffle_xor(val, mask)
        val = val + other
        mask = mask * 2
    Out[bx, tx] = val


from bench_utils import run_benchmark

if __name__ == "__main__":

    # Sub-test 1: pair_swap
    def compute_expected_swap(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            for i in range(0, WARP_SIZE, 2):
                out[w, i] = a[w, i + 1]
                out[w, i + 1] = a[w, i]
        return out.ravel().astype(np.float32)

    print("Sub-test 1: pair_swap", file=sys.stderr)
    run_benchmark(
        pair_swap, compute_expected_swap,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"swap N={N}",
    )

    # Sub-test 2: butterfly_max
    def compute_expected_max(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        maxes = a.max(axis=1, keepdims=True)
        return np.broadcast_to(maxes, a.shape).ravel().astype(np.float32)

    print("Sub-test 2: butterfly_max", file=sys.stderr)
    run_benchmark(
        butterfly_max, compute_expected_max,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"max N={N}",
        atol=1e-4,
    )

    # Sub-test 3: butterfly_sum
    def compute_expected_sum(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        sums = a.sum(axis=1, keepdims=True)
        return np.broadcast_to(sums, a.shape).ravel().astype(np.float32)

    print("Sub-test 3: butterfly_sum", file=sys.stderr)
    run_benchmark(
        butterfly_sum, compute_expected_sum,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"sum N={N}",
        init_mod=10,
        atol=1e-4,
    )
