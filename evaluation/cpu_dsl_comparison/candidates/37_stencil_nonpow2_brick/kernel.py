"""Candidate 37_stencil_nonpow2_brick: BrickLib stencil with non-pow-2 brick size (severe LOSS)

CASTLE candidate 29. Layout class: Brick+non-pow2.
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
