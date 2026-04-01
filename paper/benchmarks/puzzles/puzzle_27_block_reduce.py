"""Puzzle 27 — Block-Wide Patterns: dot product via block-level reduction.

Computes dot product using shared memory tree reduction (matches Mojo's
traditional_dot_product variant). Each thread computes a[i]*b[i],
then tree-reduce in shared memory.

Mojo also has block.sum(), block.prefix_sum(), block.broadcast() variants
which are Mojo-specific higher-level APIs. The shared memory tree reduction
is the portable pattern that all backends support.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_27_block_reduce.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = min(256, N)
assert N % BLOCK == 0
num_blocks = N // BLOCK

A_layout = OrderBy(Row(N)).TileBy([num_blocks], [BLOCK])
out_layout = Row(num_blocks)
smem_layout = Row(BLOCK)


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def block_dot_product(A: Buffer(A_layout, N), B: Buffer(A_layout, N),
                      Out: Buffer(out_layout, num_blocks),
                      smem: Shared(smem_layout, BLOCK)):
    bx = block_id.x
    tx = thread_id.x
    # Each thread computes partial product
    smem[tx] = A[bx, tx] * B[bx, tx]
    barrier()
    # Tree reduction
    stride = BLOCK // 2
    while stride > 0:
        if tx < stride:
            smem[tx] = smem[tx] + smem[tx + stride]
        barrier()
        stride = stride // 2
    # Thread 0 writes result
    if tx == 0:
        Out[bx] = smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, BLOCK)
        b = inputs[1].reshape(-1, BLOCK)
        return (a * b).sum(axis=1).astype(np.float32)

    run_benchmark(
        block_dot_product, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-4,
    )
