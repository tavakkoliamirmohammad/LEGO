"""Candidate 30_lulesh_aosoA: LULESH element-centered with AoSoA layout (LOSS on both arches)

CASTLE candidate 22. Layout class: AoSoA.
Prior verdicts: AMD LOSS, Intel LOSS.

Approximation: models strided memory access pattern characteristic of AoSoA.
Stride = 4 (elements between successive logical accesses).
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 4096
STRIDE = 4
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
        B[i] = A[i * 4] * 2.0
