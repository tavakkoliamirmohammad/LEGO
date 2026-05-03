"""Candidate 22_jacobi2d_brick: Polybench jacobi-2d with brick layout (severe LOSS ~5.7× overhead)

CASTLE candidate 14. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

XFAIL: XFAIL pending R12: brick stride not threaded through

The measure.py for this candidate reports SKIP with the reason above.
A stub scalar reference is provided for documentation purposes.
"""
import numpy as np

N = 1024


def kernel_scalar(A, B):
    """Stub scalar reference (not executed in XFAIL mode)."""
    B[:] = A * 2.0
