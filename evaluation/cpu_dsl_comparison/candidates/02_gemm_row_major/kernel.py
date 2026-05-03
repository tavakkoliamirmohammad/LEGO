"""02_gemm_row_major: C[i,j] += sum_k A[i,k]*B[k,j], row-major blocked.

Expected verdict: WIN.  The register+L1 tiling pattern (candidates 04, 05,
06, 35 in the full eval) shows 2–7x speedups on both AMD and Intel.
This captures the essence of that class at small scale so it compiles quickly.
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


def kernel_scalar(A, B, C):
    """NumPy reference: C += A @ B (accumulate into C)."""
    C += A @ B


# 2-D kernel expressed as a flat 1-D tile over the M dimension.
# Each tile covers one row of C.
@cpu_kernel(grid=(_MN,), tile=(TILE_N,))
def kernel_cpu_dsl(A: Buffer[_MK], B: Buffer[_KN], C: Buffer[_MN]):
    # tile_range iterates over TILE_N consecutive j-columns of one row.
    # The outer tile_id encodes: tile_i = tile_id // (N // TILE_N),
    #                             tile_j = tile_id  % (N // TILE_N).
    # We handle this with a flat loop; each call to `tile_range` gives
    # a contiguous block of j indices for a single i row.
    for j in tile_range:
        # j is the global column index.
        # row i = j // N  ... but with grid=(M*N,) and tile=(TILE_N,)
        # tile_id = (i * N + j_base) // TILE_N  →  i = tile_id // (N // TILE_N)
        # We can compute i at compile-time from the outer tile id.
        # For simplicity use the flat index to recover (i, j).
        for k in range(K):
            C[j] = C[j] + A[(j // N) * K + k] * B[k * N + (j % N)]
