"""03_3pt_stencil_1d: B[i] = A[i-1] + A[i] + A[i+1], interior only.

Expected verdict: WIN (unit-stride loads; overlapping windows vectorise well).

Note: v1 of the cpu_dsl pipeline has an open issue (R12) where ``B[i + k]``
stores with a constant offset trigger a memref.store type mismatch during
vector lowering. If that error fires, the result is captured as ERROR/LOSS
and the candidate records the error message, so the harness still runs clean.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1024
TILE = 16


def _ref(A, B):
    """NumPy reference: interior 3-point average."""
    B[1:-1] = A[:-2] + A[1:-1] + A[2:]


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N},
)
@cpu_kernel
def stencil_3pt(A: Buffer[N], B: Buffer[N]):
    for i in range(N - 2):
        B[i + 1] = A[i] + A[i + 1] + A[i + 2]


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = stencil_3pt.measure(A, B)
    rec["verified"] = stencil_3pt.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
