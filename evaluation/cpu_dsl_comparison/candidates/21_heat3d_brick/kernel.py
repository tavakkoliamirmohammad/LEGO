"""Candidate 21_heat3d_brick: Polybench heat-3d with brick layout (simplified).

CASTLE candidate 13. Layout class: Brick.
Prior verdicts: AMD MIXED, Intel LOSS.

The heat-3d stencil computes B[x,y,z] = 0.5*(A[x,y,z] + A[x±1,y,z] + A[x,y±1,z] + A[x,y,z±1])
— identical to the 7-point stencil. The brick layout variation is captured via
the same flat 1D offset-based approach (avoids integer division in the body).
The update is applied in-place over multiple time steps; here we capture one step.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY, NZ = 32, 32, 32
N_FLAT = NX * NY * NZ
_OFFSET = NX * NY
_INNER  = (NX - 2) * NY * NZ
TILE = 16

_NYNZ = NY * NZ
_NZ   = NZ


def kernel_scalar(A, B):
    """Heat-3d one step: B = 0.5 * (A + neighbors)."""
    a = A.reshape(NX, NY, NZ)
    b = B.reshape(NX, NY, NZ)
    b[1:-1, :, :] = 0.5 * (
        a[1:-1, :, :]
        + a[0:-2, :, :] + a[2:, :, :]
    )
    b[1:-1, 1:-1, :] += 0.5 * (a[1:-1, 0:-2, :] + a[1:-1, 2:, :])
    b[1:-1, :, 1:-1] += 0.5 * (a[1:-1, :, 0:-2] + a[1:-1, :, 2:])


@cpu_kernel(grid=(_INNER,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """Heat-3d one step — flat 1D tiling with compile-time neighbor offsets."""
    for n in tile_range:
        flat = n + _OFFSET
        B[flat] = (A[flat]
                   + A[flat - _NYNZ] + A[flat + _NYNZ]   # ±x
                   + A[flat - _NZ]   + A[flat + _NZ]      # ±y
                   + A[flat - 1]     + A[flat + 1]         # ±z
                   ) * 0.5
