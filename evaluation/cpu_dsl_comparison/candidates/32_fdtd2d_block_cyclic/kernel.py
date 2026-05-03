"""Candidate 32_fdtd2d_block_cyclic: Polybench FDTD-2D with block-cyclic layout (AMD 4-thread WIN, Intel 1-thread LOSS)

CASTLE candidate 24. Layout class: Block-cyclic.
Prior verdicts: AMD MIXED, Intel LOSS.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 4096
TILE = 16


def kernel_scalar(A, B, C):
    """NumPy reference: tiled unit-stride accumulation."""
    np.add(A * B, C, out=C)


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        C[i] = A[i] * B[i] + C[i]
