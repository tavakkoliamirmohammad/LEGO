"""02_gemm_row_major: C[i,j] += sum_k A[i,k]*B[k,j], row-major blocked.

Expected verdict: WIN.  The register+L1 tiling pattern (candidates 04, 05,
06, 35 in the full eval) shows 2–7x speedups on both AMD and Intel.
This captures the essence of that class at small scale so it compiles quickly.

Kernel formulation: tile over i (rows), with j (columns) as the innermost
unit-stride loop. This avoids integer division (j//N, j%N) that prevented
vectorization in the flat-grid formulation and matches how gcc sees the loop
order: inner j is unit-stride in both C[i*N+j] and B[k*N+j].
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = 64
N = 64
K = 64
TILE_M = 8
TILE_N = 8
# Flat buffer sizes
_MK = M * K
_KN = K * N
_MN = M * N

_N = N   # column count as a compile-time constant visible in outer scope


def kernel_scalar(A, B, C):
    """NumPy reference: C += A @ B (accumulate into C)."""
    C += A @ B


# Tile over rows (M), inner k loop for accumulation, innermost j loop (N)
# is unit-stride in B and C — the vectorizer can vectorize over j.
# Each tile covers TILE_M consecutive rows; tile_range iterates over those rows.
# The outer grid=(M,) means one tile per row; with tile=(TILE_M,) the
# lego-vectorize pass sees a flat loop over M and vectorizes the inner j body.
#
# Flat structure emitted: for i in [0,M):  for k in [0,K):  for j in [0,N): ...
# The j-loop (innermost, unit-stride) is the vectorization target.
@cpu_kernel(grid=(M,), tile=(TILE_M,))
def kernel_cpu_dsl(A: Buffer[_MK], B: Buffer[_KN], C: Buffer[_MN]):
    for i in tile_range:
        # For each row i and accumulation index k, the j-loop over columns
        # is unit-stride in B[k*N+j] and C[i*N+j] — vectorizable.
        for k in range(K):
            for j in range(_N):
                C[i * _N + j] = C[i * _N + j] + A[i * K + k] * B[k * _N + j]
