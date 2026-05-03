"""Candidate 21_heat3d_brick: Polybench heat-3d with brick layout (consistent LOSS)

CASTLE candidate 13. Layout class: Brick.
Prior verdicts: AMD MIXED, Intel LOSS.

XFAIL: XFAIL pending R12: brick stride not threaded through

The measure.py for this candidate reports SKIP with the reason above.
A stub scalar reference is provided for documentation purposes.
"""
import numpy as np

N = 1024


def kernel_scalar(A, B):
    """Stub scalar reference (not executed in XFAIL mode)."""
    B[:] = A * 2.0
