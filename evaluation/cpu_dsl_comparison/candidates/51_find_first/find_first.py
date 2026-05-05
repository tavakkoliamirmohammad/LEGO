"""51_find_first: branchless find-first-index where A[i] > threshold.

Tier-1 ``find-first`` reduction.  We sidestep the ``break`` requirement
by encoding the search as a min-of-IV reduction:

    idx = N             # sentinel "not found"
    for i in range(N):
        if A[i] > thr:
            if i < idx:
                idx = i

The generalised fold pass currently does *not* recognise the IV itself
as a combine value (it only accepts unit-stride loads or loop-invariant
constants).  So this kernel should fall through to scalar.  Future work
covered by the ``all the tiers`` follow-up: extend the fold matcher to
allow ``valueIn = induction_var`` for ``MinUI/MinSI`` combines, which
would also pick up ``argmin-without-paired-value`` patterns.

clang -O3 -march=native handles this with cmov-based reductions, so the
parity check tells us how far the scalar baseline is from optimal.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20


def _ref(A, out):
    # numpy version: argmax of (A > thr) gives first True; if none, returns 0,
    # so guard with all-False check.
    mask = A > 0.5
    if mask.any():
        out[0] = np.argmax(mask)
    else:
        out[0] = N


@benchmark(
    reference=_ref, n_iters=200, warmup=20, rtol=0.0,
    meta={"N": N, "layout_class": "FindFirst", "prior_verdict": "NEW"},
)
@cpu_kernel
def find_first(A: Buffer[N], out: Buffer[1]):
    # Use a float-typed accumulator so cpu_dsl can promote into the f32
    # output buffer.  We store ``i + 1`` while predicate holds for the
    # *first* time only, then convert back to a 0-based index in numpy.
    # cpu_dsl can't unify INDEX vs F32 in arithmetic, so we maintain a
    # float counter ``i_f`` alongside ``i`` and reduce in float space.
    idx_f = 1048576.0
    i_f = 0.0
    for i in range(N):
        if A[i] > 0.5:
            if i_f < idx_f:
                idx_f = i_f
        i_f = i_f + 1.0
    out[0] = idx_f


def _make_args():
    rng = np.random.default_rng(0)
    # Half above 0.5, half below — first hit somewhere in the first ~thousand.
    A = rng.uniform(low=0.0, high=1.0, size=N).astype(np.float32)
    out = np.zeros(1, dtype=np.float32)
    return A, out


if __name__ == "__main__":
    A, out = _make_args()
    rec = find_first.measure(A, out)
    rec["verified"] = find_first.verify(A, np.zeros(1, dtype=np.float32))
    print(json.dumps(rec, default=str))
