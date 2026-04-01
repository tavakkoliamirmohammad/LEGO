"""Puzzle 27 — Block-Wide Patterns: reduce values across entire block.

Uses shared memory tree reduction (the proven portable pattern).
Each block reduces its tile, thread 0 writes the result.

Mojo equivalent:
    shared[i] = input[i]
    barrier()
    stride = BLOCK // 2
    while stride > 0:
        if i < stride: shared[i] += shared[i + stride]
        barrier()
        stride //= 2
    if i == 0: output[block_idx.x] = shared[0]
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

layout = OrderBy(Row(N)).TileBy([num_blocks], [BLOCK])
out_layout = Row(num_blocks)
smem_layout = Row(BLOCK)


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def block_reduce(A: Buffer(layout, N), Out: Buffer(out_layout, num_blocks),
                 smem: Shared(smem_layout, BLOCK)):
    bx = block_id.x
    tx = thread_id.x
    # Load to shared memory
    smem[tx] = A[bx, tx]
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
        return inputs[0].reshape(-1, BLOCK).sum(axis=1).astype(np.float32)

    run_benchmark(
        block_reduce, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1.0,
    )
