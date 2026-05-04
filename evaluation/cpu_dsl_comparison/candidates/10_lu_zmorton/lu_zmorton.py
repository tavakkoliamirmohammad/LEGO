"""10_lu_zmorton: LU-style reduction with Z-Morton gather access.

CASTLE candidate 02 - polybench-lu-zmorton.
Layout class: Z-Morton.
Prior verdicts: AMD WIN (1.9-3.09x), Intel WIN (1.2-1.91x).

LU decomposition accesses rows and columns with a Morton-swizzled layout.
This kernel approximates the inner elimination loop: for each pivot row k,
update row i via: A[i] -= A[k] * A[pivot(k,i)].

Uses the same 1-D Morton swizzle as gemm-zmorton to exercise the gather path.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 16  # 65536 — matches measure.py's N_BENCH
TILE = 16


def _ref(A, B, C):
    """Inline NumPy scalar equivalent for LU-style Morton gather."""
    indices = np.arange(N, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N - 1)
    C[:] -= A[morton] * B


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-3,
    meta={"N": N, "layout_class": "Z-Morton", "prior_verdict": "WIN"},
)
@cpu_kernel(grid=(N,))
def lu_zmorton(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in range(N):
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = C[i] - A[morton] * B[i]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = lu_zmorton.measure(A, B, C)
    rec["verified"] = lu_zmorton.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
