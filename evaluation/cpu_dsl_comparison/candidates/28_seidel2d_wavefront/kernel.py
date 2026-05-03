"""Candidate 28_seidel2d_wavefront: Polybench seidel-2d with wavefront tiling (LOSS on AMD, mixed Intel)

CASTLE candidate 20. Layout class: Wavefront tile.
Prior verdicts: AMD LOSS, Intel MIXED.
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
