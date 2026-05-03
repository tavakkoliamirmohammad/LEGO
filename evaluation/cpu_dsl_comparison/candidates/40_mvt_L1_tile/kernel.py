"""Candidate 40_mvt_L1_tile: Polybench MVT with L1 tiling (strong WIN)

CASTLE candidate 33. Layout class: L1 tile.
Prior verdicts: AMD WIN, Intel WIN.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 4096
TILE = 16


def kernel_scalar(A, B, C):
    """NumPy reference: tiled unit-stride accumulation."""
    np.add(A * B, C, out=C)


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        C[i] = A[i] * B[i] + C[i]
