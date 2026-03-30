"""Parallel sum reduction with shared memory — @gpu_kernel DSL.

Layouts:
  A:    OrderBy(Row(N)).TileBy([N // BLOCK], [BLOCK])
        — 2-level: [block_idx, thread_idx]
  Out:  Row(num_blocks) — one partial sum per block
  smem: Row(BLOCK)      — shared memory for tree reduction

The while loop is compile-time unrolled (stride is a Python int).
Each unrolled step emits a runtime `scf.if` for `tx < stride`.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python reduce_sum.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = 256

assert N % BLOCK == 0, f"N ({N}) must be a multiple of BLOCK ({BLOCK})"
num_blocks = N // BLOCK

# --- Layouts ---
A_layout = OrderBy(Row(N)).TileBy([num_blocks], [BLOCK])
out_layout = Row(num_blocks)
smem_layout = Row(BLOCK)


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def reduce_sum(A: Buffer(A_layout, N), Out: Buffer(out_layout, num_blocks),
               smem: Shared(smem_layout, BLOCK)):
    tx = thread_id.x
    bx = block_id.x

    smem[tx] = A[bx, tx]
    barrier()

    stride = BLOCK // 2
    while stride > 0:
        if tx < stride:
            smem[tx] = smem[tx] + smem[tx + stride]
        barrier()
        stride = stride // 2

    if tx == 0:
        Out[bx] = smem[0]


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        return inputs[0].reshape(-1, BLOCK).sum(axis=1).astype(np.float32)

    run_benchmark(
        reduce_sum, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"N={N}",
        init_mod=10,
    )
