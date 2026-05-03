"""Candidate 18_tblis_notranspose: TBLIS-style tensor contraction without transposition (borderline WIN/LOSS)

CASTLE candidate 10. Layout class: TBLIS.
Prior verdicts: AMD WIN, Intel LOSS.

XFAIL: XFAIL pending R18: TBLIS runtime microkernel selection not expressible in cpu_dsl v1

The measure.py for this candidate reports SKIP with the reason above.
A stub scalar reference is provided for documentation purposes.
"""
import numpy as np

N = 1024


def kernel_scalar(A, B):
    """Stub scalar reference (not executed in XFAIL mode)."""
    B[:] = A * 2.0
