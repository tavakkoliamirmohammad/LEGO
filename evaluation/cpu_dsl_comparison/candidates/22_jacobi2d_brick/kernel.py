"""Candidate 22_jacobi2d_brick: Polybench jacobi-2d with brick layout (simplified).

CASTLE candidate 14. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

Jacobi-2d: B[i,j] = 0.25 * (A[i-1,j] + A[i+1,j] + A[i,j-1] + A[i,j+1]).
Brick layout approximation: flat 1D buffer with stride-N y-neighbors and
stride-1 x-neighbors. No integer division in the kernel body.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY = 256, 256
N_FLAT = NX * NY
# Interior: skip first and last rows (for ±NX neighbors)
_OFFSET = NX      # first safe row
_INNER  = (NX - 2) * NY
TILE = 16

_NY = NY


def kernel_scalar(A, B):
    """Jacobi-2d one step on flat 2D-like buffer."""
    a = A.reshape(NX, NY)
    b = B.reshape(NX, NY)
    b[1:-1, 1:-1] = 0.25 * (
        a[0:-2, 1:-1] + a[2:, 1:-1]    # ±row neighbors
        + a[1:-1, 0:-2] + a[1:-1, 2:]  # ±col neighbors
    )


@cpu_kernel(grid=(_INNER,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """Jacobi-2d — flat 1D tiling over interior elements."""
    for n in tile_range:
        flat = n + _OFFSET
        B[flat] = (A[flat - _NY]    # -row neighbor (strided)
                   + A[flat + _NY]  # +row neighbor (strided)
                   + A[flat - 1]    # -col neighbor (unit stride)
                   + A[flat + 1]    # +col neighbor (unit stride)
                   ) * 0.25
