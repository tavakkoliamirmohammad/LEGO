"""Puzzle 28 — Async Memory Operations: 1D convolution with shared memory.

Mojo puzzle: 1D convolution of a large vector with a 5-element kernel,
using async DRAM→shared memory copies. We implement the equivalent
synchronous version (same computation, same shared memory tiling).

  output[i] = sum_{k=0}^{4} input[i + k - 2] * kernel[k]

Boundary handling: elements within 2 of tile edges are copied directly.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_28_async_copy.py N")
    sys.exit(1)

N = int(sys.argv[1])
TPB = 256
assert N % TPB == 0
num_blocks = N // TPB

CONV_SIZE = 5
RADIUS = CONV_SIZE // 2  # = 2

layout = OrderBy(Row(N)).TileBy([num_blocks], [TPB])
out_layout = OrderBy(Row(N)).TileBy([num_blocks], [TPB])
smem_layout = Row(TPB)
# Convolution kernel weights stored as compile-time constants
# Mojo uses [1, 2, 3, 4, 5]
W0 = 1.0
W1 = 2.0
W2 = 3.0
W3 = 4.0
W4 = 5.0


@gpu_kernel(grid=(num_blocks,), block=(TPB,))
def conv1d_tiled(A: Buffer(layout, N), Out: Buffer(out_layout, N),
                 smem: Shared(smem_layout, TPB)):
    bx = block_id.x
    tx = thread_id.x
    # Load tile into shared memory
    smem[tx] = A[bx, tx]
    barrier()
    # Boundary elements: just copy input (no convolution at tile edges)
    if tx < RADIUS:
        Out[bx, tx] = smem[tx]
    if tx >= TPB - RADIUS:
        Out[bx, tx] = smem[tx]
    # Center elements: full 5-point convolution
    if tx >= RADIUS:
        if tx < TPB - RADIUS:
            Out[bx, tx] = smem[tx - 2] * W0 + smem[tx - 1] * W1 + smem[tx] * W2 + smem[tx + 1] * W3 + smem[tx + 2] * W4


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0]
        weights = np.array([W0, W1, W2, W3, W4], dtype=np.float32)
        out = np.zeros(N, dtype=np.float32)
        for blk in range(num_blocks):
            for t in range(TPB):
                i = blk * TPB + t
                if t < RADIUS or t >= TPB - RADIUS:
                    out[i] = a[i]
                else:
                    out[i] = sum(a[i + k - RADIUS] * weights[k] for k in range(CONV_SIZE))
        return out.astype(np.float32)

    run_benchmark(
        conv1d_tiled, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-4,
    )
