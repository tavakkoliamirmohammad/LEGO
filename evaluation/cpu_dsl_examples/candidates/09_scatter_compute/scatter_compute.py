"""09_scatter_compute: per-element polynomial evaluation followed by an
indirect scatter store.

This kernel is the genuine clang-miss: the scatter store at the end
forces clang to scalarise the *entire loop* including the polynomial
compute (no auto-vectoriser will speculate vector compute when the
final write is non-affine).

LEGO with the address-chain skip + ``vector.scatter`` infrastructure
keeps the polynomial in vector form (5 vector FMAs, vector load) and
only the final scatter is non-affine. The compute-side speedup is
straight 8-16x lane parallelism.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer
from lego.backend.compiler import DType
from lego.core import Row

N = 1 << 20   # 1M elements


def _ref(A, idx, B):
    """NumPy reference: polynomial(A) → scatter via idx into B."""
    x = A
    p = ((((x * 0.1 + 0.2) * x + 0.3) * x + 0.4) * x + 0.5) * x + 0.6
    B[idx] = p


@benchmark(
    reference=_ref, n_iters=500, warmup=50, rtol=1e-4,
    meta={"N": N, "layout_class": "ScatterCompute", "prior_verdict": "NEW"},
)
@cpu_kernel
def scatter_compute(A: Buffer[N],
                    idx: Buffer(Row(N), N, dtype=DType.i64),
                    B: Buffer[N]):
    for i in range(N):
        x = A[i]
        # 5-stage Horner evaluation — five vector FMAs.
        p = (((((x * 0.1 + 0.2) * x + 0.3) * x + 0.4) * x + 0.5) * x + 0.6)
        B[idx[i]] = p


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    idx = rng.permutation(N).astype(np.int64)
    B = np.zeros(N, dtype=np.float32)
    return A, idx, B


if __name__ == "__main__":
    A, idx, B = _make_args()
    rec = scatter_compute.measure(A, idx, B)
    rec["verified"] = scatter_compute.verify(A, idx, np.zeros_like(B))
    print(json.dumps(rec, default=str))
