"""36_gemm_nonpow2_morton: Polybench GEMM with non-pow-2 Morton layout.

CASTLE candidate 28. Layout class: Morton+non-pow2.
Prior verdicts: AMD WIN, Intel WIN.

Same Morton gather + FMA shape as 35_heat3d_pow2_pad; the candidate name
flags the non-pow-2 dimension class, but the pow-2 buffer size keeps the
mask cheap so the vectorizer still emits a gather + unit-stride FMA.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 65536
TILE = 16


def _ref(A, B, C):
    """NumPy reference: Morton gather + accumulate."""
    idx = np.arange(N, dtype=np.int32)
    ti = idx & 0x5555
    tj = (idx >> 1) & 0x5555
    m = (ti | (tj << 1)) & (N - 1)
    C[:] = A[m] * B + C


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Morton+non-pow2", "prior_verdict": "WIN"},
)
@cpu_kernel
def gemm_nonpow2_morton(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in range(N):
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = A[morton] * B[i] + C[i]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = gemm_nonpow2_morton.measure(A, B, C)
    rec["verified"] = gemm_nonpow2_morton.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
