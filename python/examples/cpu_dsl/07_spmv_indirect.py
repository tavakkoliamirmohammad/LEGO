"""07_spmv_indirect: B[idx[i]] = A[i] * 2.0 — indirect-store (scatter).

Targets clang's *real* blind spot: indirect stores. clang's
auto-vectoriser emits vector.gather for indirect *reads* but routinely
scalarises indirect *writes* — even when ``idx`` is provably a
permutation (no duplicates). Reasons: memory-model ordering concerns,
heuristic budget for scatter, and conservative aliasing analysis.

LEGO's vectoriser classifies the store as ``NonAffine`` and emits
``vector.scatter`` (lowers to AVX-512 ``vpscatterqd`` on x86 or to
gather-via-shuffle on ARM NEON). This is the genuine gap — we measure
it directly.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer
from lego.backend.compiler import DType
from lego.core import Row

N = 1 << 20   # 1M elements


def _ref(A, idx, B):
    """NumPy reference: B[idx] = A * 2.0  (scatter)."""
    B[idx] = A * 2.0


@benchmark(
    reference=_ref, n_iters=1000, warmup=100, rtol=1e-5,
    meta={"N": N, "layout_class": "Scatter", "prior_verdict": "NEW"},
)
@cpu_kernel
def spmv_indirect(A: Buffer[N],
                  idx: Buffer(Row(N), N, dtype=DType.i64),
                  B: Buffer[N]):
    for i in range(N):
        B[idx[i]] = A[i] * 2.0


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    idx = rng.permutation(N).astype(np.int64)
    B = np.zeros(N, dtype=np.float32)
    return A, idx, B


if __name__ == "__main__":
    A, idx, B = _make_args()
    rec = spmv_indirect.measure(A, idx, B)
    rec["verified"] = spmv_indirect.verify(A, idx, np.zeros_like(B))
    print(json.dumps(rec, default=str))
