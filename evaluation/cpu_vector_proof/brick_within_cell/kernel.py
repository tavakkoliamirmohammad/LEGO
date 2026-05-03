"""Within-brick compute kernel — no cross-brick reads.

Demonstrates the new @cpu_kernel + lego-to-x86-vector pipeline on a kernel
that's structurally similar to a brick-stencil's within-cell computation
(but without the cross-brick neighbor reads — those are R12 future work).
"""
import numpy as np

from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX = NY = NZ = 64
BRICK = 8
N = NX * NY * NZ


@cpu_kernel(grid=(N,), tile=(BRICK,))
def brick_within(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i] * 2.0 + 1.0


def brick_within_scalar(A, B):
    """NumPy scalar reference."""
    B[:] = A * 2.0 + 1.0
