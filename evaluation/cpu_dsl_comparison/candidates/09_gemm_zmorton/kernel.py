"""09_gemm_zmorton: GEMM with Z-Morton (Z-order) layout index swizzle.

CASTLE candidate 01 — polybench-gemm-zmorton.
Layout class: Z-Morton.
Prior verdicts: AMD WIN (2.07–3.30×), Intel WIN (1.15–2.27×).

The Z-Morton layout stores matrix elements in Z-order (Morton code) to improve
2D spatial locality. For a C[i][j] matrix of side N, the flat index is
  morton_idx(i, j) = interleave_bits(i, j)
Here we use a simplified 1-D swizzle approximation (2-D interleave needs
32-bit ops not yet cleanly expressible in cpu_dsl tile_range).

XFAIL status: The full 2-D Morton GEMM requires nested loops with 2-D
index swizzle. The cpu_dsl flat-loop model handles 1-D Morton well (see
candidate 05_morton_2d). For full 2-D GEMM + Morton, roadmap item R18
(nested loop support in cpu_dsl) is needed.

This kernel tests the 1-D gather path (same as 05_morton_2d but with
a GEMM-style accumulation to verify correctness of gather + FMA paths).
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1024
TILE = 16


def kernel_scalar(A, B, C):
    """Simplified GEMM-like gather: C[i] += A[morton(i)] * B[i]."""
    indices = np.arange(N, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N - 1)
    C[:] += A[morton] * B


@cpu_kernel(grid=(N,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = C[i] + A[morton] * B[i]
