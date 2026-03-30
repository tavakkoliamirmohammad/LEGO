"""Tiled matrix multiply with shared memory C = A @ B — unified KernelBuilder API.

Outer loop tiles the K dimension in chunks of TILE.  Each tile:
  1. Load A and B sub-tiles from global → shared memory
  2. Barrier
  3. Accumulate from shared memory (inner k-loop)
  4. Barrier
Uses nested ctx.for_range() for outer K-tiles and inner accumulation.
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
    print("Usage: python matmul_smem.py N  or  python matmul_smem.py M N K")
    sys.exit(1)

TILE = 16  # tile size for output and shared memory (TM = TN = TK)

# --- Global buffer layouts ---
A_buf = LayoutBuffer(Row(M, K), shape=(M, K), dtype=DType.f32)
B_buf = LayoutBuffer(Row(K, N), shape=(K, N), dtype=DType.f32)
C_buf = LayoutBuffer(Row(M, N), shape=(M, N), dtype=DType.f32)

# --- Shared memory buffers ---
Smem_A = LayoutBuffer(Row(TILE, TILE), shape=(TILE, TILE), dtype=DType.f32, shared=True)
Smem_B = LayoutBuffer(Row(TILE, TILE), shape=(TILE, TILE), dtype=DType.f32, shared=True)

grid = (N // TILE, M // TILE, 1)
block = (TILE, TILE, 1)


def matmul_smem_kernel(ctx):
    tx = ctx.thread_id.x
    ty = ctx.thread_id.y
    bx = ctx.block_id.x
    by = ctx.block_id.y

    row = ctx.addi(ctx.muli(by, ctx.const_index(TILE)), ty)
    col = ctx.addi(ctx.muli(bx, ctx.const_index(TILE)), tx)

    num_tiles = K // TILE
    zero = ctx.const_f32(0.0)

    # Outer loop: iterate over K-tiles
    def tile_body(t, acc):
        tile_offset = ctx.muli(t, ctx.const_index(TILE))

        # Load A tile: A[row, tile_offset + tx] → Smem_A[ty, tx]
        a_col = ctx.addi(tile_offset, tx)
        a_val = ctx.load(0, [row, a_col])
        ctx.store(a_val, 3, [ty, tx])       # smem_A index 3

        # Load B tile: B[tile_offset + ty, col] → Smem_B[ty, tx]
        b_row = ctx.addi(tile_offset, ty)
        b_val = ctx.load(1, [b_row, col])
        ctx.store(b_val, 4, [ty, tx])       # smem_B index 4

        ctx.barrier()

        # Inner loop: accumulate from shared memory
        def k_body(k, inner_acc):
            sa = ctx.load(3, [ty, k])       # Smem_A[ty, k]
            sb = ctx.load(4, [k, tx])       # Smem_B[k, tx]
            return [ctx.addf(inner_acc, ctx.mulf(sa, sb))]

        [new_acc] = ctx.for_range(TILE, k_body, init_vals=[acc])

        ctx.barrier()
        return [new_acc]

    [result] = ctx.for_range(num_tiles, tile_body, init_vals=[zero])

    ctx.store(result, 2, [row, col])


builder = KernelBuilder(
    buffers=[A_buf, B_buf, C_buf, Smem_A, Smem_B],
    kernel_body=matmul_smem_kernel,
    name="matmul_smem",
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
