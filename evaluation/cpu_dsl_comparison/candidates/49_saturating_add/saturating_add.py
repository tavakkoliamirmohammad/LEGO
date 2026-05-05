"""49_saturating_add: clamp each element of A+B to [-1.0, 1.0] (saturating add).

Tier-3 parity check.  clang -O3 -march=native vectorises this cleanly with
``vminps`` / ``vmaxps`` (saturating arithmetic is a known compiler fast
path).  LEGO's existing pipeline should match clang here — we benchmark it
to *verify* parity rather than expect a win, completing the non-affine
coverage matrix.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20


def _ref(A, B, C):
    s = A + B
    C[:] = np.minimum(np.maximum(s, -1.0), 1.0)


@benchmark(
    reference=_ref, n_iters=500, warmup=50, rtol=1e-5,
    meta={"N": N, "layout_class": "Saturating", "prior_verdict": "NEW"},
)
@cpu_kernel
def saturating_add(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    lo = -1.0
    hi = 1.0
    for i in range(N):
        s = A[i] + B[i]
        C[i] = max(lo, min(hi, s))


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = saturating_add.measure(A, B, C)
    rec["verified"] = saturating_add.verify(A, B, np.zeros(N, dtype=np.float32))
    print(json.dumps(rec, default=str))
