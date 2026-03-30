"""Parallel sum reduction with shared memory — unified KernelBuilder API.

Classic GPU tree reduction: each block loads a chunk into shared memory,
then reduces via a binary tree (Python-level unrolled since BLOCK is known
at kernel-build time).  Thread 0 of each block writes the partial sum to
the output buffer.  The host sums the partial results for the full reduction.

Demonstrates: ctx.if_(), ctx.lt(), ctx.eq(), Python-level loop unrolling,
shared memory, barriers.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer

if len(sys.argv) != 2:
    print("Usage: python reduce_sum.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = 256  # threads per block

assert N % BLOCK == 0, f"N ({N}) must be a multiple of BLOCK ({BLOCK})"

num_blocks = N // BLOCK

# --- Buffers ---
A   = LayoutBuffer(Row(N),          shape=(N,),          dtype=DType.f32)
Out = LayoutBuffer(Row(num_blocks), shape=(num_blocks,), dtype=DType.f32)
Smem = LayoutBuffer(Row(BLOCK),     shape=(BLOCK,),      dtype=DType.f32, shared=True)

grid = (num_blocks, 1, 1)
block = (BLOCK, 1, 1)


def reduce_sum_kernel(ctx):
    tx = ctx.thread_id.x
    bx = ctx.block_id.x
    gid = ctx.addi(ctx.muli(bx, ctx.const_index(BLOCK)), tx)

    # Load global → shared memory
    val = ctx.load_flat(0, gid)
    ctx.store_flat(val, 2, tx)  # smem (buf index 2)
    ctx.barrier()

    # Tree reduction (Python-level unroll — BLOCK is a compile-time constant)
    stride = BLOCK // 2
    while stride > 0:
        s = stride  # capture for lambda

        def reduce_step(s=s):
            neighbor = ctx.addi(tx, ctx.const_index(s))
            mine   = ctx.load_flat(2, tx)
            theirs = ctx.load_flat(2, neighbor)
            ctx.store_flat(ctx.addf(mine, theirs), 2, tx)

        ctx.if_(ctx.lt(tx, ctx.const_index(stride)), reduce_step)
        ctx.barrier()
        stride //= 2

    # Thread 0 writes block result to output
    def write_result():
        result = ctx.load_flat(2, ctx.const_index(0))
        ctx.store_flat(result, 1, bx)  # Out[bx]

    ctx.if_(ctx.eq(tx, ctx.const_index(0)), write_result)


builder = KernelBuilder(
    buffers=[A, Out, Smem],
    kernel_body=reduce_sum_kernel,
    name="reduce_sum",
    grid=grid,
    block=block,
)


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        # Each block sums BLOCK elements → one partial sum per block
        return inputs[0].reshape(-1, BLOCK).sum(axis=1).astype(np.float32)

    # init_mod=10 keeps values 0-9 so f32 accumulation is exact
    run_benchmark(
        builder, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"N={N}",
        init_mod=10,
    )
