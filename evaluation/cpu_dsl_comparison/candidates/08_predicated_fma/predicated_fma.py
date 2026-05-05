"""08_predicated_fma: predicated FMA — apply C[i] = A[i]*B[i] + C[i]
only where ``mask[i] > threshold``.

Exercises predicated computation: an inner update conditional on a
runtime value.  Verifies LLVM's auto-vectoriser produces a masked
sequence through the cpu_dsl pipeline with the same FP semantics
LEGO's lowering sets at vector→llvm.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20   # 1M elements
THRESHOLD = 0.0   # half the data passes (random std-normal)


def _ref(A, B, mask, C):
    """NumPy reference: C[mask > 0] += A[mask > 0] * B[mask > 0]."""
    sel = mask > THRESHOLD
    C[sel] = A[sel] * B[sel] + C[sel]


@benchmark(
    reference=_ref, n_iters=1000, warmup=100, rtol=1e-5,
    meta={"N": N, "layout_class": "Predicated", "prior_verdict": "NEW"},
)
@cpu_kernel
def predicated_fma(A: Buffer[N], B: Buffer[N], mask: Buffer[N], C: Buffer[N]):
    for i in range(N):
        if mask[i] > 0.0:
            C[i] = A[i] * B[i] + C[i]


def _make_args():
    rng = np.random.default_rng(0)
    A    = rng.standard_normal(N).astype(np.float32)
    B    = rng.standard_normal(N).astype(np.float32)
    mask = rng.standard_normal(N).astype(np.float32)
    C    = np.zeros(N, dtype=np.float32)
    return A, B, mask, C


if __name__ == "__main__":
    A, B, mask, C = _make_args()
    rec = predicated_fma.measure(A, B, mask, C)
    rec["verified"] = predicated_fma.verify(A, B, mask, np.zeros_like(C))
    print(json.dumps(rec, default=str))
