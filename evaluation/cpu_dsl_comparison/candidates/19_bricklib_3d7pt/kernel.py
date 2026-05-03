"""Candidate 19_bricklib_3d7pt: BrickLib 3D 7-point stencil (brick layout — gather overhead dominates)

CASTLE candidate 11. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

XFAIL: XFAIL pending R12: brick stride not threaded through; BrickLib not bundled

The measure.py for this candidate reports SKIP with the reason above.
A stub scalar reference is provided for documentation purposes.
"""
import numpy as np

N = 1024


def kernel_scalar(A, B):
    """Stub scalar reference (not executed in XFAIL mode)."""
    B[:] = A * 2.0
