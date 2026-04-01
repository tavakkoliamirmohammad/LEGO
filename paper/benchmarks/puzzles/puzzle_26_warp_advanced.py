"""Puzzle 26 — Advanced Warp Patterns: butterfly reduction with XOR shuffle.

Butterfly reduction: each step exchanges with a partner at distance 2^k.
  step 0: exchange with lane ^ 1   (neighbors)
  step 1: exchange with lane ^ 2   (pairs)
  step 2: exchange with lane ^ 4   (quads)
  ...
  step k: exchange with lane ^ 2^k

After log2(WARP_SIZE) steps, every lane has the full reduction.

Mojo equivalent:
    var val = input[i]
    for k in range(log2(WARP_SIZE)):
        var other = shuffle_xor(val, 1 << k)
        val += other
    output[i] = val
"""
import sys
import math
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
LOG_WARP = int(math.log2(WARP_SIZE))

layout = OrderBy(Row(N)).TileBy([num_warps], [WARP_SIZE])


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def butterfly_reduce(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    # Butterfly reduction: compile-time unrolled via while loop
    mask = 1
    while mask < WARP_SIZE:
        other = shuffle_xor(val, mask)
        val = val + other
        mask = mask * 2
    # Every lane has the full sum — write it
    Out[bx, tx] = val


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        sums = a.sum(axis=1, keepdims=True)
        return np.broadcast_to(sums, a.shape).ravel().astype(np.float32)

    run_benchmark(
        butterfly_reduce, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"N={N}",
        init_mod=10,
        atol=1.0,
    )
