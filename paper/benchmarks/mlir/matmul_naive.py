"""Naive matrix multiply C = A @ B — @gpu_kernel DSL.

Layouts:
  A: Row(M, K)  — standard row-major, indexed as A[row, k]
  B: Row(K, N)  — standard row-major, indexed as B[k, col]
  C: OrderBy(Row(M, N)).TileBy([M//T, N//T], [T, T])
     — tiled by block grid, indexed as C[by, bx, ty, tx]
     — lego.apply computes: (by*T + ty) * N + (bx*T + tx)

Each thread computes one element of C by iterating over K.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) == 4:
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python matmul_naive.py N  or  python matmul_naive.py M N K")
    sys.exit(1)

TILE = 16

# --- Layouts ---
A_layout = Row(M, K)
B_layout = Row(K, N)
C_layout = OrderBy(Row(M, N)).TileBy([M // TILE, N // TILE], [TILE, TILE])


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def matmul_naive(A: Buffer(A_layout, M, K), B: Buffer(B_layout, K, N),
                 C: Buffer(C_layout, M, N)):
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for k in range(K):
        acc += A[row, k] * B[k, col]
    C[block_id.y, block_id.x, thread_id.y, thread_id.x] = acc


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel()

    run_benchmark(
        matmul_naive, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
    )
