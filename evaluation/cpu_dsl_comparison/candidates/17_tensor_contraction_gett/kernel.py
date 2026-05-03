"""Candidate 17_tensor_contraction_gett: Tensor contraction with GETT-style tiling (unit-stride innermost loop)

CASTLE candidate 9. Layout class: GETT tile.
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
