"""Candidate 20_bricklib_3d13pt: BrickLib 3D 13-point stencil (brick layout — marginal Intel WIN from AVX-512 gathers)

CASTLE candidate 12. Layout class: Brick.
Prior verdicts: AMD PARITY, Intel WIN.

XFAIL: XFAIL pending R12: brick stride not threaded through; BrickLib not bundled

The measure.py for this candidate reports SKIP with the reason above.
A stub scalar reference is provided for documentation purposes.
"""
import numpy as np

N = 1024


def kernel_scalar(A, B):
    """Stub scalar reference (not executed in XFAIL mode)."""
    B[:] = A * 2.0
