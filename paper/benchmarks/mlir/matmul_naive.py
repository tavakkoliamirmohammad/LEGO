"""Naive matrix multiply C = A @ B — @gpu_kernel DSL.

Each thread computes one element of C by iterating over the K dimension.
The `for k in range(K)` loop lowers to `scf.for` with an automatically
detected f32 iter_arg for the accumulator.  No shared memory required.
"""
import sys
import numpy as np
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) == 4:
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python matmul_naive.py N  or  python matmul_naive.py M N K")
    sys.exit(1)

TILE = 16  # block dimension: TILE x TILE threads


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def matmul_naive(A: Buffer[M, K], B: Buffer[K, N], C: Buffer[M, N]):
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for k in range(K):
        acc += A[row, k] * B[k, col]
    C[row, col] = acc


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel()

    # init_mod=10 keeps values 0-9 so f32 accumulation is exact
    run_benchmark(
        matmul_naive, compute_expected,
        targets=["cuda", "llvmspirv"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
    )
