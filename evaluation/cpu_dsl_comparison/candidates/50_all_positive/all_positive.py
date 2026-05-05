"""50_all_positive: scan an array, set a flag to 0 if any element is non-positive.

Tier-2 ``all`` reduction.  Encoded as an int iter_arg ``flag`` initialised
to 1 and reset to 0 inside ``if A[i] <= 0``.  Equivalent to
``flag = flag AND (A[i] > 0)``.  The generalised fold pass currently
matches associative bin-op + iter-arg combines, *not* predicate-gated
constant resets — so we expect this kernel to fall through to the scalar
path and report PARITY (or close to it) against clang's branchy scan.
The point of the candidate is empirical coverage of the ``all/any`` shape
in the tier matrix.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20


def _ref(A, out):
    out[0] = 1 if np.all(A > 0.0) else 0


@benchmark(
    reference=_ref, n_iters=200, warmup=20, rtol=0.0,
    meta={"N": N, "layout_class": "AllAny", "prior_verdict": "NEW"},
)
@cpu_kernel
def all_positive(A: Buffer[N], out: Buffer[1]):
    flag = 1.0
    for i in range(N):
        if A[i] <= 0.0:
            flag = 0.0
    out[0] = flag


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.uniform(low=0.1, high=2.0, size=N).astype(np.float32)  # all positive
    out = np.zeros(1, dtype=np.float32)
    return A, out


if __name__ == "__main__":
    A, out = _make_args()
    rec = all_positive.measure(A, out)
    rec["verified"] = all_positive.verify(A, np.zeros(1, dtype=np.float32))
    print(json.dumps(rec, default=str))
