"""Tiled matrix multiply with shared memory C = A @ B — @gpu_kernel DSL.

Layouts:
  A, B: Row(M, K) / Row(K, N) — standard row-major for global loads
  C:    OrderBy(Row(M, N)).TileBy(…) — tiled output
  sA, sB: Row(TILE, TILE) — shared memory tiles

Outer for-loop tiles K.  Each tile: load → barrier → accumulate → barrier.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) == 4:
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python matmul_smem.py N  or  python matmul_smem.py M N K")
    sys.exit(1)

TILE = 16

# --- Layouts ---
A_layout = Row(M, K)
B_layout = Row(K, N)
C_layout = OrderBy(Row(M, N)).TileBy([M // TILE, N // TILE], [TILE, TILE])
smem_layout = Row(TILE, TILE)


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def matmul_smem(A: Buffer(A_layout, M, K), B: Buffer(B_layout, K, N),
                C: Buffer(C_layout, M, N),
                sA: Shared(smem_layout, TILE, TILE),
                sB: Shared(smem_layout, TILE, TILE)):
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for t in range(K // TILE):
        sA[thread_id.y, thread_id.x] = A[row, t * TILE + thread_id.x]
        sB[thread_id.y, thread_id.x] = B[t * TILE + thread_id.y, col]
        barrier()
        for kk in range(TILE):
            acc += sA[thread_id.y, kk] * sB[kk, thread_id.x]
        barrier()
    C[block_id.y, block_id.x, thread_id.y, thread_id.x] = acc


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel()

    run_benchmark(
        matmul_smem, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
    )
