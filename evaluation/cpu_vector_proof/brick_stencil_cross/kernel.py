"""Cross-brick stencil proof-point — v1 scaffold.

Demonstrates the @cpu_kernel vectorization path on a stencil-style kernel.

v1 STATUS: This kernel operates on a contiguous 1D array and exercises the
inner-axis vectorization path via a 3-point stencil. It does NOT perform
cross-brick reads because:

  1. Cross-brick reads require knowledge of the actual brick stride when
     computing the neighbor-block base address.  The current IR-shape
     scaffolding (Task 13) computes the second-block base from the boundary
     lane index, which is correct for the FileCheck IR shape test but
     produces wrong addresses for real cross-brick stencils.

  2. When roadmap entry R12 lands (brick-aware second-block base computation),
     this file will be updated to use the 3D 7-point stencil with halos.

See README.md for full context and the R12 roadmap entry.
"""
import numpy as np

from lego.backend.cpu_dsl import cpu_kernel, Buffer

# 1D 3-point stencil over N elements; boundary elements (0, N-1) are not
# written, so the grid covers only the interior: indices 1 .. N-2.
N = 1024
TILE = 16

# Grid covers the interior N-2 elements; the kernel shifts the tile-local
# index by 1 to map into the interior.  Buffer size is N so edge loads
# A[i-1] and A[i+1] are always in-bounds.
@cpu_kernel(grid=(N - 2,), tile=(TILE,))
def stencil_3pt(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        # i is the global index in [0, N-2); shift by 1 → interior index
        B[i + 1] = A[i] + A[i + 1] + A[i + 2]


def stencil_3pt_scalar(A, B):
    """NumPy scalar reference — interior elements only."""
    B[1:-1] = A[:-2] + A[1:-1] + A[2:]
