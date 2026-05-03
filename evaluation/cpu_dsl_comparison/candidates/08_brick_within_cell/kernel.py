"""08_brick_within_cell: within-brick vectorisation proof-point.

Copied from evaluation/cpu_vector_proof/brick_within_cell/kernel.py.
Expected verdict: PARITY or slight LOSS vs NumPy (NumPy BLAS loop is
already heavily optimised; beating it requires FMA intrinsic quality
per roadmap R12 + better cost model).

This is the only candidate already proven to compile and run correctly
end-to-end in v1 (see roadmap.md R1 proof-point results).
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX = NY = NZ = 64
BRICK = 8
N = NX * NY * NZ


@cpu_kernel(grid=(N,), tile=(BRICK,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i] * 2.0 + 1.0


def kernel_scalar(A, B):
    """NumPy scalar reference."""
    B[:] = A * 2.0 + 1.0
