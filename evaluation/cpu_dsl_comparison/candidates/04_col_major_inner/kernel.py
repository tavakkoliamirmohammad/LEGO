"""04_col_major_inner: C[i,j] = A[j,i] (transpose copy; column-major inner loop).

Expected verdict: PARITY or LOSS.  The inner loop strides through column-major
storage (stride != 1), so vectorisation yields gather-style loads and the
bandwidth benefit is partial.  Representative of RFP-class candidates (15)
and block-cyclic candidates (25) which showed borderline results.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = 256
N = 256
_MN = M * N
TILE = 16


def kernel_scalar(A, C):
    """NumPy reference: C = A.T (out-of-place transpose)."""
    # A is stored row-major M×N; result C is M×N row-major = N×M col-major copy.
    np.copyto(C, A.T)


# Flat layout: A[j, i] = A_flat[j*M + i], C[i, j] = C_flat[i*N + j]
# v1 limitation: store target must also appear as a load source in the
# same kernel body to avoid a vector-store type mismatch in the lowering.
# We use C[j] = A[col*M+i] + C[j]*0.0 (semantically identical; makes C
# both a load and a store so the vectoriser matches types correctly).
@cpu_kernel(grid=(_MN,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[_MN], C: Buffer[_MN]):
    for j in tile_range:
        i = j // N
        col = j - i * N
        # Read-modify-write form: add C[j]*0.0 so C is both read and written.
        C[j] = A[col * M + i] + C[j] * 0.0
