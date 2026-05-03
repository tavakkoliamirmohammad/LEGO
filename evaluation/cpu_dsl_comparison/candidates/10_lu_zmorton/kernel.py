"""10_lu_zmorton: LU-style reduction with Z-Morton gather access.

CASTLE candidate 02 — polybench-lu-zmorton.
Layout class: Z-Morton.
Prior verdicts: AMD WIN (1.9–3.09×), Intel WIN (1.2–1.91×).

LU decomposition accesses rows and columns with a Morton-swizzled layout.
This kernel approximates the inner elimination loop: for each pivot row k,
update row i via: A[i] -= A[k] * A[pivot(k,i)].

Uses the same 1-D Morton swizzle as gemm-zmorton to exercise the gather path.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1024
TILE = 16


def kernel_scalar(A, B, C):
    """Row update: C[i] -= A[morton(i)] * B[i]."""
    indices = np.arange(N, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N - 1)
    C[:] -= A[morton] * B


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = C[i] - A[morton] * B[i]
