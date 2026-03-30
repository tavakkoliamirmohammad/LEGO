"""Tiled matrix multiply with shared memory C = A @ B — @gpu_kernel DSL.

Outer loop tiles the K dimension in chunks of TILE.  Each tile:
  1. Load A and B sub-tiles from global → shared memory
  2. barrier()
  3. Inner for loop accumulates from shared memory
  4. barrier()
Nested `for … in range()` loops lower to nested `scf.for` with iter_args.
"""
import sys
import numpy as np
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) == 4:
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python matmul_smem.py N  or  python matmul_smem.py M N K")
    sys.exit(1)

TILE = 16  # tile size for output and shared memory (TM = TN = TK)


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def matmul_smem(A: Buffer[M, K], B: Buffer[K, N], C: Buffer[M, N],
                sA: Shared[TILE, TILE], sB: Shared[TILE, TILE]):
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
    C[row, col] = acc


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel()

    # init_mod=10 keeps values 0-9 so f32 accumulation is exact
    run_benchmark(
        matmul_smem, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
    )
