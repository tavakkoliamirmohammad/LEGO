"""Parallel sum reduction with shared memory — @gpu_kernel DSL.

Classic GPU tree reduction: each block loads a chunk into shared memory,
then reduces via a binary tree.  The `while stride > 0` loop is
compile-time unrolled (stride is a Python int), while `if tx < stride`
lowers to `scf.if` at each unrolled step.  Thread 0 of each block
writes the partial sum to the output buffer.

Demonstrates: native if/while, shared memory, barrier(), Python-level
unrolling, all via the @gpu_kernel AST transformer.
"""
import sys
import numpy as np
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python reduce_sum.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = 256  # threads per block

assert N % BLOCK == 0, f"N ({N}) must be a multiple of BLOCK ({BLOCK})"

num_blocks = N // BLOCK


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def reduce_sum(A: Buffer[N], Out: Buffer[num_blocks], smem: Shared[BLOCK]):
    tx = thread_id.x
    bx = block_id.x
    gid = bx * BLOCK + tx

    smem[tx] = A[gid]
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
        # Each block sums BLOCK elements → one partial sum per block
        return inputs[0].reshape(-1, BLOCK).sum(axis=1).astype(np.float32)

    # init_mod=10 keeps values 0-9 so f32 accumulation is exact
    run_benchmark(
        reduce_sum, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"N={N}",
        init_mod=10,
    )
