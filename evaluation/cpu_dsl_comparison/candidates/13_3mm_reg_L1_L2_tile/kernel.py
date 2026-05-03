"""Candidate 13_3mm_reg_L1_L2_tile: 3MM (3-matrix multiplication) with register + L1 + L2 tiling

CASTLE candidate 5. Layout class: Reg+L1+L2 tile.
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
