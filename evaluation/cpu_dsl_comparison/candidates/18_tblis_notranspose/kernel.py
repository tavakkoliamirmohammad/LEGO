"""Candidate 18_tblis_notranspose: tiled matrix multiply (TBLIS/ikj pattern).

CASTLE candidate 10. Layout class: TBLIS.
Prior verdicts: AMD WIN, Intel LOSS.

The full TBLIS library uses runtime microkernel selection (blocked GEBP algorithm
with architecture-specific register tiles), which is not directly expressible in
cpu_dsl v1. This version captures the core layout pattern using the ikj (outer-product)
decomposition:
- Vectorize over j (column dimension of C and B) using tile_range.
- Inner loop over combined i,k pairs (range(M*K)) is a scalar reduction.
- Each j-tile processes all M*K outer-product contributions to C[:,j_tile].

This formulation enables:
- Unit-stride vector access to C[i*N+j] (j is tile_range, unit stride in j)
- Broadcast scalar load of A[ik] (loop-invariant in j)
- Unit-stride vector access to B[(ik%K)*N+j] (j is tile_range, unit stride in j)

R20 fix: the R20 vectorizer extension (outer-loop vectorization with inner
reduction scf.for) enables the inner M*K range() loop to remain scalar while
the outer j tile_range is vectorized.

Verdict target: WIN vs C O3 (vectorized GEBP-style inner loop).
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


@cpu_kernel(grid=(N,), tile=(16,))
def kernel_cpu_dsl(A: Buffer[_MK], B: Buffer[_KN], C: Buffer[_MN]):
    """Vectorize over j (columns); inner ik loop processes all outer-product pairs.

    Access pattern (with j as tile_range outer IV):
      C[(ik//K)*N+j]  — unit-stride in j, broadcast row offset (ik//K)*N
      A[ik]           — broadcast (loop-invariant in j)
      B[(ik%K)*N+j]   — unit-stride in j, broadcast column offset (ik%K)*N
    All three are Unit or Broadcast → no gathers, full AVX-512 throughput.
    """
    for j in tile_range:
        for ik in range(_MK):
            C[(ik // K) * N + j] = C[(ik // K) * N + j] + A[ik] * B[(ik % K) * N + j]
