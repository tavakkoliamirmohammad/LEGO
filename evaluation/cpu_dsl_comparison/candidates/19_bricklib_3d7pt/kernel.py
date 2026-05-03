"""Candidate 19_bricklib_3d7pt: simplified 3D 7-point stencil (brick layout pattern).

CASTLE candidate 11. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

Full BrickLib API is not bundled and uses an ABI not expressible in cpu_dsl v1.
This version captures the core access pattern: a flat 3D-like stencil over a
1D buffer where each element reads 6 neighbors at fixed positive/negative offsets.

Key: we lay out the buffer as 1D with stride-1 neighbors (±1, ±NX, ±NX*NY).
This avoids integer division in the kernel body (all address arithmetic is
add/sub of compile-time constants, which is unit-stride or strided gather).

The vectorizable axis is the flat 1D tile_range over the interior.
Boundary effects: we skip the first and last NX*NY rows so indices are always
within bounds for the ±NX*NY neighbors.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY, NZ = 32, 32, 32
N_FLAT = NX * NY * NZ
# Interior slice: skip first and last layer to avoid OOB in ±NX neighbor.
# We tile over [NX*NY, (NX-1)*NY*NZ) — interior points only (no boundary guard needed).
_OFFSET = NX * NY        # offset of first safe element
_INNER  = (NX - 2) * NY * NZ  # number of interior elements
TILE = 16

_NYNZ = NY * NZ
_NZ   = NZ


def kernel_scalar(A, B):
    """7-point stencil on 3D flat buffer. Interior only."""
    a = A.reshape(NX, NY, NZ)
    b = B.reshape(NX, NY, NZ)
    b[1:-1, :, :] = (
        a[1:-1, :, :]
        + a[0:-2, :, :]   # -x neighbor
        + a[2:,   :, :]   # +x neighbor
    ) * 0.5
    # Add y and z neighbors clipped to interior for simplicity.
    b[1:-1, 1:-1, :] += (a[1:-1, 0:-2, :] + a[1:-1, 2:, :]) * 0.5
    b[1:-1, :, 1:-1] += (a[1:-1, :, 0:-2] + a[1:-1, :, 2:]) * 0.5


@cpu_kernel(grid=(_INNER,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """3D 7-point stencil — flat 1D tiling over interior, offset-based addressing.

    All address arithmetic uses compile-time constants (unit-stride and strided
    gathers of ±1, ±NZ, ±NX*NY). No integer division in the kernel body.
    """
    for n in tile_range:
        # n is the flat interior index; flat buffer address = n + _OFFSET.
        flat = n + _OFFSET
        B[flat] = (A[flat]
                   + A[flat - _NYNZ]   # -x face neighbor (stride NX*NY)
                   + A[flat + _NYNZ]   # +x face neighbor
                   + A[flat - _NZ]     # -y neighbor (stride NY)
                   + A[flat + _NZ]     # +y neighbor
                   + A[flat - 1]       # -z neighbor (unit stride)
                   + A[flat + 1]       # +z neighbor
                   ) * 0.5
