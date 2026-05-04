"""23_symm_rfp: Polybench SYMM with RFP storage format (borderline PARITY/LOSS).

CASTLE candidate 15. Layout class: RFP.
Prior verdicts: AMD PARITY, Intel LOSS.

Approximation: models stride-2 memory access pattern characteristic of RFP.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 16384
STRIDE = 2
TILE = 16
N_BUF = N * STRIDE


def _ref(A, B):
    """Vectorized strided gather: B[i] = A[i * STRIDE] * 2.0."""
    np.multiply(A[::STRIDE], 2.0, out=B)


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "RFP", "prior_verdict": "LOSS"},
)
@cpu_kernel(grid=(N,))
def symm_rfp(A: Buffer[N_BUF], B: Buffer[N]):
    for i in range(N):
        B[i] = A[i * 2] * 2.0


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N_BUF).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = symm_rfp.measure(A, B)
    rec["verified"] = symm_rfp.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
