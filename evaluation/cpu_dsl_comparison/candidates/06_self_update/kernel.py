"""06_self_update: shift-add stencil  B[i] = A[i-1] + A[i].

The self_update pattern models a simple two-point stencil that reads A[i-1]
and A[i] and writes B[i].  There is NO loop-carried dependence on B — each
output element depends only on two consecutive input elements.  This makes
the loop fully vectorizable.

The C baseline (self_update_O3) implements the same computation:
    for i in [1, N): B[i] = A[i-1] + A[i]

Expected verdict: WIN.  Removing the prior prefix-sum loop-carried dep
(B[i] = B[i-1] + A[i]) and aligning kernel semantics with the C baseline
allows LEGO to generate AVX-512 vector code and beat the C reference.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

# N-1 interior elements (skip B[0] which has no left neighbor)
N = 4096
N_INNER = N - 1
TILE = 16


def kernel_scalar(A, B):
    """Reference: shift-add stencil — matches C baseline self_update_O3."""
    B[1:] = A[:N - 1] + A[1:]


# Tile over N_INNER interior elements.  No conditional needed because we
# tile [0, N_INNER) and each element i accesses A[i] and A[i+1] → unit-stride.
# The vectorizer sees a flat unit-stride pattern over two consecutive A loads
# and one B store — directly matches AVX-512 vadd + vmovups.
@cpu_kernel(grid=(N_INNER,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        # i ∈ [0, N-1): read A[i] and A[i+1], write B[i+1].
        # No dep on B: fully vectorizable.
        B[i + 1] = A[i] + A[i + 1]
