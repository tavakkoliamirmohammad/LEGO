"""Candidate 18_tblis_notranspose: simplified tiled matrix multiply (TBLIS pattern).

CASTLE candidate 10. Layout class: TBLIS.
Prior verdicts: AMD WIN, Intel LOSS.

The full TBLIS library uses runtime microkernel selection (blocked GEBP algorithm
with architecture-specific register tiles), which is not directly expressible in
cpu_dsl v1. This simplified version captures the core layout pattern:
- Matrix multiply with explicit K-dimension tiling (column-strip of B).
- Inner loop over K is a scalar reduction (no vectorization of k-axis).
- Outer loop over the flat M*N grid is the vectorizable tile_range.

The LEGO-expressible version: C[i] += A[i%M * K + k] * B[k * N + i//M]
where i encodes both (row, col) dimensions flatly.

Verdict: PARITY — this pattern is similar to 02_gemm_row_major; scalar JIT
matches vec_jit because the k-loop reduction guard prevents vectorization
of the reduction axis, and the outer loop is a simple 1D tile.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = 64
N = 64
K = 64

_MN = M * N
_MK = M * K
_KN = K * N


def kernel_scalar(A, B, C):
    """NumPy GEMM reference: C += A @ B."""
    # A is (M, K), B is (K, N), C is (M, N).
    C += A @ B


@cpu_kernel(grid=(_MN,), tile=(16,))
def kernel_cpu_dsl(A: Buffer[_MK], B: Buffer[_KN], C: Buffer[_MN]):
    """Flat tile over C elements; inner k-reduction is scalar."""
    for ij in tile_range:
        # ij encodes (i, j): i = ij // N, j = ij % N
        for k in range(K):
            C[ij] = C[ij] + A[(ij // N) * K + k] * B[k * N + (ij % N)]
