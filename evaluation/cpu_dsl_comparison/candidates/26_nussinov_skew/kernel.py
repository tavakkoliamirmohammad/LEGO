"""Candidate 26_nussinov_skew: NPDP Nussinov with skew tiling (unit-stride after skew transform)

CASTLE candidate 18. Layout class: Skew tile.
Prior verdicts: AMD WIN, Intel WIN.

Approximation: models strided memory access pattern characteristic of Skew tile.
Stride = 2 (elements between successive logical accesses).
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 4096
STRIDE = 2
TILE = 16
# Buffer must be large enough to hold strided accesses.
N_BUF = N * STRIDE


def kernel_scalar(A, B):
    """Strided read: B[i] = A[i * STRIDE]."""
    for i in range(N):
        B[i] = A[i * STRIDE] * 2.0


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_BUF], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i * 2] * 2.0
