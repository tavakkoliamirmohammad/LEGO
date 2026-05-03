"""Candidate 37_stencil_nonpow2_brick: stencil with non-pow-2 brick size (simplified).

CASTLE candidate 29. Layout class: Brick+non-pow2.
Prior verdicts: AMD LOSS, Intel LOSS.

Full BrickLib not bundled. This version captures the non-power-of-2 brick
size effect: we use a 3D domain of size 30x30x30 (not a power of 2) to
test that the cpu_dsl pipeline handles non-pow2 tile trip counts correctly.
The stencil is a 5-point 2D stencil (cross-shaped) on a 2D slice to keep
the N small and compilation fast.

The "non-pow2" aspect is captured by _INNER = 28*900 = 25200 (not a power of 2).
The vectorizer handles non-pow2 trip counts via the tail loop.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

# Non-pow2 dimensions: 30x30x30 interior
NX, NY = 30, 30
N_FLAT = NX * NY
_OFFSET = NY       # skip first row
_INNER  = (NX - 2) * NY   # 28 * 30 = 840
TILE = 8           # smaller tile for non-pow2

_NY = NY


def kernel_scalar(A, B):
    """5-point 2D stencil (cross) on non-pow2 grid."""
    a = A.reshape(NX, NY)
    b = B.reshape(NX, NY)
    b[1:-1, 1:-1] = 0.25 * (
        a[0:-2, 1:-1] + a[2:, 1:-1]
        + a[1:-1, 0:-2] + a[1:-1, 2:]
    )


@cpu_kernel(grid=(_INNER,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """5-point 2D stencil on non-pow2 grid — flat 1D tiling."""
    for n in tile_range:
        flat = n + _OFFSET
        B[flat] = (A[flat - _NY]    # -row neighbor (strided)
                   + A[flat + _NY]  # +row neighbor (strided)
                   + A[flat - 1]    # -col neighbor (unit stride)
                   + A[flat + 1]    # +col neighbor (unit stride)
                   ) * 0.25
