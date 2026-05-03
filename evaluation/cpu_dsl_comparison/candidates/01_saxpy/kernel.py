"""01_saxpy: y[i] = a*x[i] + y[i].  Trivial unit-stride fused-multiply-add.

Expected verdict: WIN.  Unit-stride load/store with no loop-carried
dependences is the canonical vectorisation sweet spot.  LEGO's
lego-to-x86-vector pipeline emits AVX-512 FMA instructions here.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 8192
TILE = 16


def kernel_scalar(a, X, Y):
    """NumPy reference: y[:] = a*x + y (in-place)."""
    Y[:] = a * X + Y


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(a: float, X: Buffer[N], Y: Buffer[N]):
    for i in tile_range:
        Y[i] = a * X[i] + Y[i]
