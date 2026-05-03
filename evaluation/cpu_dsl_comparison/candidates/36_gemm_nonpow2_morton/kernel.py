"""Candidate 36_gemm_nonpow2_morton: Polybench GEMM with non-pow-2 Morton layout (WIN on both arches)

CASTLE candidate 28. Layout class: Morton+non-pow2.
Prior verdicts: AMD WIN, Intel WIN.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 65536
TILE = 16


def kernel_scalar(A, B, C):
    """Morton gather + accumulate."""
    idx = np.arange(N, dtype=np.int32)
    ti = idx & 0x5555
    tj = (idx >> 1) & 0x5555
    m = (ti | (tj << 1)) & (N - 1)
    C[:] = A[m] * B + C


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = A[morton] * B[i] + C[i]
