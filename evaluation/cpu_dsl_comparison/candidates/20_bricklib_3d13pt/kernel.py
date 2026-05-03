"""Candidate 20_bricklib_3d13pt: simplified 3D 13-point stencil (brick layout pattern).

CASTLE candidate 12. Layout class: Brick.
Prior verdicts: AMD PARITY, Intel WIN.

Full BrickLib API not bundled. This version captures the 13-point stencil
access pattern: each element reads 12 face+edge neighbors at offsets
±1, ±NZ, ±NX*NY, and the 4 face-diagonal neighbors (±1±NZ, etc.).

All address arithmetic uses compile-time constants — no integer division.
The ±1 neighbors are unit-stride; ±NZ and ±NX*NY are strided gathers.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY, NZ = 32, 32, 32
N_FLAT = NX * NY * NZ
_OFFSET = NX * NY      # skip first layer
_INNER  = (NX - 2) * NY * NZ
TILE = 16

_NYNZ = NY * NZ
_NZ   = NZ


def kernel_scalar(A, B):
    """13-point stencil on flat buffer (approximation: 6-point + 6 diagonal)."""
    a = A.reshape(NX, NY, NZ)
    b = B.reshape(NX, NY, NZ)
    # 7-point core + 6 edge neighbors
    b[1:-1, :, :] = (
        a[1:-1, :, :]
        + a[0:-2, :, :] + a[2:, :, :]     # ±x
    ) * (1.0 / 13.0)
    b[1:-1, 1:-1, :] += (a[1:-1, 0:-2, :] + a[1:-1, 2:, :]) * (1.0 / 13.0)
    b[1:-1, :, 1:-1] += (a[1:-1, :, 0:-2] + a[1:-1, :, 2:]) * (1.0 / 13.0)
    # Edge neighbors (±x±y, etc.) — use cross-product of pairs
    b[1:-1, 1:-1, :] += (a[0:-2, 0:-2, :] + a[2:, 2:, :]) * (1.0 / 13.0)
    b[1:-1, :, 1:-1] += (a[0:-2, :, 0:-2] + a[2:, :, 2:]) * (1.0 / 13.0)
    b[1:-1, 1:-1, :] += (a[0:-2, 2:,  :] + a[2:, 0:-2, :]) * (1.0 / 13.0)


@cpu_kernel(grid=(_INNER,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """13-point stencil — flat 1D tiling with compile-time neighbor offsets."""
    for n in tile_range:
        flat = n + _OFFSET
        B[flat] = (A[flat]
                   + A[flat - _NYNZ]    # -x
                   + A[flat + _NYNZ]    # +x
                   + A[flat - _NZ]      # -y
                   + A[flat + _NZ]      # +y
                   + A[flat - 1]        # -z
                   + A[flat + 1]        # +z
                   + A[flat - _NYNZ - _NZ]   # -x-y diagonal
                   + A[flat + _NYNZ + _NZ]   # +x+y diagonal
                   + A[flat - _NYNZ + _NZ]   # -x+y diagonal
                   + A[flat + _NYNZ - _NZ]   # +x-y diagonal
                   + A[flat - _NYNZ - 1]     # -x-z diagonal
                   + A[flat + _NYNZ + 1]     # +x+z diagonal
                   ) * (1.0 / 13.0)
