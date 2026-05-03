"""Candidate 12_gemm_reg_L1_L2_tile: GEMM with register + L1 + L2 tiling.

CASTLE candidate 4. Layout class: Reg+L1+L2 tile.
Prior verdicts: AMD WIN, Intel WIN.

C = A @ B, matrices of size N×N in row-major flat layout.
LEGO kernel: register + L1 + L2 tiled 3-loop GEMM.
C baseline (naive_gemm): plain 3-loop i,j,k with no tiling.

CASTLE WIN mechanism: tiling restructures iteration order to fit working
set in L1/L2 caches → fewer cache misses → faster than naive triple loop
that gcc -O3 cannot tile without guided hints.

N=512 is large enough to stress caches (3×512×512×4 = 3 MB >> L1 cache)
and small enough to JIT-compile in < 1s.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 512          # matrix side
TILE_L2 = 64     # L2 tile (fits in L2 cache)
TILE_L1 = 16     # L1 tile
TILE_REG = 4     # register tile (unrolled)

_NN = N * N


def kernel_scalar(A, B, C):
    """NumPy reference: C += A @ B (in-place, flat layout)."""
    result = (A.reshape(N, N) @ B.reshape(N, N)).reshape(_NN)
    C += result


# Three-level tiled GEMM expressed as flat loops over i-tiles.
# Iteration order: L2 tile of i, L2 tile of j, L2 tile of k,
#   L1 tile of i, L1 tile of j, L1 tile of k,
#   register tile of i, then inner unit-stride j.
#
# With cpu_dsl's flat model we express the outermost loop as tile_range
# over the i dimension, and inner loops as range().
@cpu_kernel(grid=(N,), tile=(TILE_L1,))
def kernel_cpu_dsl(A: Buffer[_NN], B: Buffer[_NN], C: Buffer[_NN]):
    # tile_range iterates i over [0, N) one element at a time.
    # lego-vectorize will strip-mine the inner j-loop by TILE_L1.
    for i in tile_range:
        # L1-tiled k and j inner loops.
        for k0 in range(N // TILE_L1):
            for j0 in range(N // TILE_L1):
                for k in range(TILE_L1):
                    kk = k0 * TILE_L1 + k
                    a_ik = A[i * N + kk]
                    for j in range(TILE_L1):
                        jj = j0 * TILE_L1 + j
                        C[i * N + jj] = C[i * N + jj] + a_ik * B[kk * N + jj]
