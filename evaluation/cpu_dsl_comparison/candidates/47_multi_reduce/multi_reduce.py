"""47_multi_reduce: sum + max + min computed in a single pass over A[].

Multi-output reduction with three iter_args (s, mx, mn).  clang -O3 -march=native
can vectorise *one* reduction (e.g. sum) cleanly, but stacking max+min on top
generally drops to a more conservative scalar/partial-vector form on Zen 4
because the two min/max ops compete with the sum's reduction tree for register
pressure.  LEGO's generalised filtered-reduce pass (no scf.if path) holds N
parallel vector accumulators in registers and emits one vector.reduction per
output at the end.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20   # 1M elements


def _ref(A, out_sum, out_max, out_min):
    out_sum[0] = float(A.sum())
    out_max[0] = float(A.max())
    out_min[0] = float(A.min())


@benchmark(
    reference=_ref, n_iters=500, warmup=50, rtol=1e-3,
    meta={"N": N, "layout_class": "MultiReduce", "prior_verdict": "NEW"},
)
@cpu_kernel
def multi_reduce(A: Buffer[N],
                 out_sum: Buffer[1], out_max: Buffer[1], out_min: Buffer[1]):
    s = 0.0
    mx = -1.0e30
    mn = 1.0e30
    for i in range(N):
        v = A[i]
        s = s + v
        mx = max(mx, v)
        mn = min(mn, v)
    out_sum[0] = s
    out_max[0] = mx
    out_min[0] = mn


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    out_sum = np.zeros(1, dtype=np.float32)
    out_max = np.zeros(1, dtype=np.float32)
    out_min = np.zeros(1, dtype=np.float32)
    return A, out_sum, out_max, out_min


if __name__ == "__main__":
    A, os, om, on = _make_args()
    rec = multi_reduce.measure(A, os, om, on)
    rec["verified"] = multi_reduce.verify(A, np.zeros(1, dtype=np.float32),
                                          np.zeros(1, dtype=np.float32),
                                          np.zeros(1, dtype=np.float32))
    print(json.dumps(rec, default=str))
