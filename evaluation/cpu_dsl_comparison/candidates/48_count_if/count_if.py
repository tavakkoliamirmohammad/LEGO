"""48_count_if: predicated counter — count elements satisfying A[i] > 0.

Derives from the generalised filtered-reduce pass (loop-invariant value 1.0
+ predicate-gated addf).  Common pattern in classification / thresholding
workloads; clang -O3 vectorises this with cmpgtps + popcount + add.  LEGO
emits a vector mask + select(mask, 1, 0) + add reduction; somewhat slower
than clang's popcount on Zen 4 but in the same order of magnitude.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20


def _ref(A, out):
    out[0] = float((A > 0.0).sum())


@benchmark(
    reference=_ref, n_iters=500, warmup=50, rtol=1e-4,
    meta={"N": N, "layout_class": "PredicatedCount", "prior_verdict": "NEW"},
)
@cpu_kernel
def count_if(A: Buffer[N], out: Buffer[1]):
    cnt = 0.0
    for i in range(N):
        if A[i] > 0.0:
            cnt = cnt + 1.0
    out[0] = cnt


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    return A, out


if __name__ == "__main__":
    A, o = _make_args()
    rec = count_if.measure(A, o)
    rec["verified"] = count_if.verify(A, np.zeros(1, dtype=np.float32))
    print(json.dumps(rec, default=str))
