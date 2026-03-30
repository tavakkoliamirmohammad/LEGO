"""Naive matrix multiply C = A @ B — unified KernelBuilder API (all GPU backends).

Each thread computes one element of C by iterating over the K dimension
with a carried accumulator via ctx.for_range().
No shared memory required.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) == 4:
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python matmul_naive.py N  or  python matmul_naive.py M N K")
    sys.exit(1)

TILE = 16  # block dimension: TILE x TILE threads

# --- Global buffer layouts: 2D row-major ---
A_buf = LayoutBuffer(Row(M, K), shape=(M, K), dtype=DType.f32)
B_buf = LayoutBuffer(Row(K, N), shape=(K, N), dtype=DType.f32)
C_buf = LayoutBuffer(Row(M, N), shape=(M, N), dtype=DType.f32)

# Grid: one block per TILE x TILE tile of C
grid = (N // TILE, M // TILE, 1)
block = (TILE, TILE, 1)


def matmul_naive_kernel(ctx):
    tx = ctx.thread_id.x   # column within tile
    ty = ctx.thread_id.y   # row within tile
    bx = ctx.block_id.x    # block column
    by = ctx.block_id.y    # block row

    # Global row/col of the C element this thread computes
    row = ctx.addi(ctx.muli(by, ctx.const_index(TILE)), ty)
    col = ctx.addi(ctx.muli(bx, ctx.const_index(TILE)), tx)

    # Accumulate over K dimension
    zero = ctx.const_f32(0.0)

    def k_body(k, acc):
        a_val = ctx.load(0, [row, k])   # A[row, k]
        b_val = ctx.load(1, [k, col])   # B[k, col]
        return [ctx.addf(acc, ctx.mulf(a_val, b_val))]

    [result] = ctx.for_range(K, k_body, init_vals=[zero])

    ctx.store(result, 2, [row, col])  # C[row, col]


builder = KernelBuilder(
    buffers=[A_buf, B_buf, C_buf],
    kernel_body=matmul_naive_kernel,
    name="matmul_naive",
    grid=grid,
    block=block,
)


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel()

    # init_mod=10 keeps values 0-9 so f32 accumulation is exact
    run_benchmark(
        builder, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
    )
