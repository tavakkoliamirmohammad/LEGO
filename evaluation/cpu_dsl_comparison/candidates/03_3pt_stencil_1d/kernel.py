"""03_3pt_stencil_1d: B[i] = A[i-1] + A[i] + A[i+1], interior only.

Expected verdict: WIN (unit-stride loads; overlapping windows vectorise well).

Note: v1 of the cpu_dsl pipeline has an open issue (R12) where
``B[i + k]`` stores with a constant offset trigger a memref.store type
mismatch during vector lowering.  If that error fires, the result is
captured as ERROR/LOSS and the candidate records the error message, so
the harness still runs clean.
"""
import numpy as np
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1024
TILE = 16


def kernel_scalar(A, B):
    """NumPy reference: interior 3-point average."""
    B[1:-1] = A[:-2] + A[1:-1] + A[2:]


# grid covers the N-2 interior elements; B write offset is i+1
# (R12 note: offset store B[i+1] may hit type-mismatch in v1)
@cpu_kernel(grid=(N - 2,), tile=(TILE,))
def kernel_cpu_dsl(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        B[i + 1] = A[i] + A[i + 1] + A[i + 2]
