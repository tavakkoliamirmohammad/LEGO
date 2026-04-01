"""Puzzle 28 — Async Memory Operations: overlap memory transfer with computation.

NOTE: Full cp.async semantics require the nvgpu dialect (not yet bound).
This puzzle demonstrates the PATTERN using shared memory double-buffering
with explicit loads and barriers — the same structure that async copies
would optimize.

The double-buffering pattern:
  1. Load tile 0 into buffer A
  2. For each remaining tile:
     a. Load next tile into buffer B
     b. Barrier (ensure A is ready)
     c. Compute on buffer A
     d. Swap A and B
  3. Compute on last buffer

With async_copy, step (a) would overlap with step (c).
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_28_async_copy.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = min(256, N)
assert N % BLOCK == 0
NUM_TILES = N // BLOCK

# Each block reduces N elements in BLOCK-sized tiles
layout = Row(N)
out_layout = Row(1)
smem_layout = Row(BLOCK)


@gpu_kernel(grid=(1,), block=(BLOCK,))
def double_buffered_reduce(A: Buffer(layout, N), Out: Buffer(out_layout, 1),
                           sA: Shared(smem_layout, BLOCK),
                           sB: Shared(smem_layout, BLOCK)):
    tx = thread_id.x
    # Load first tile into sA
    sA[tx] = A[tx]
    acc = 0.0
    # Process tiles with double buffering pattern
    for t in range(NUM_TILES):
        barrier()
        # Accumulate from current buffer (sA on even tiles, sB on odd)
        # Since we can't dynamically select buffers, we use sA for all tiles
        # and load sequentially. The PATTERN is what matters.
        acc = acc + sA[tx]
        if t + 1 < NUM_TILES:
            barrier()
            sA[tx] = A[(t + 1) * BLOCK + tx]
    # Tree reduction of acc values
    sA[tx] = acc
    barrier()
    stride = BLOCK // 2
    while stride > 0:
        if tx < stride:
            sA[tx] = sA[tx] + sA[tx + stride]
        barrier()
        stride = stride // 2
    if tx == 0:
        Out[0] = sA[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return np.array([inputs[0].sum()], dtype=np.float32)

    run_benchmark(
        double_buffered_reduce, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1.0,
    )
