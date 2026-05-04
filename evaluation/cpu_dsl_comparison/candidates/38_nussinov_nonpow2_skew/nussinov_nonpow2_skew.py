"""38_nussinov_nonpow2_skew: NPDP Nussinov-style non-pow-2 skew tiling.

CASTLE candidate 31. Layout class: Skew+non-pow2.
Prior verdicts: AMD MIXED, Intel MIXED.

Approximation: models the strided memory access pattern characteristic of
Skew+non-pow2 tiling — ``B[i] = A[i*STRIDE] * 2.0`` with STRIDE=2. The
strided gather on the read side stresses the vectorizer's strided-load
emit path. R19 ROADMAP: known correctness issue in LegoVectorize Strided
(catch-all path vectorizes the index MulIOp before Strided emit reads it),
so the verify step here may surface as a tolerance mismatch.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 16384
STRIDE = 2
TILE = 16
# Buffer must be large enough to hold strided accesses.
N_BUF = N * STRIDE


def _ref(A, B):
    """NumPy reference: B[i] = A[i*STRIDE] * 2.0."""
    np.multiply(A[::STRIDE], 2.0, out=B)


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Skew+non-pow2", "prior_verdict": "MIXED"},
)
@cpu_kernel
def nussinov_nonpow2_skew(A: Buffer[N_BUF], B: Buffer[N]):
    for i in range(N):
        B[i] = A[i * 2] * 2.0


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N_BUF).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = nussinov_nonpow2_skew.measure(A, B)
    rec["verified"] = nussinov_nonpow2_skew.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
