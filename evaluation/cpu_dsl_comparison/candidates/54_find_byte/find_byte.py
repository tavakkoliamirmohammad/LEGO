"""54_find_byte: count occurrences of a target byte value in a stream.

Tier-1 ``find-byte`` shape — same predicate-gated count as 48_count_if
but with an *equality* test on a quantised value, mimicking
``memchr``-style scans.  We keep both a count and a first-match index
to cover the dual-output flavour of find-byte (libc ``memchr`` returns
position; many engines use it for tag dispatch and want both).

The fold pass picks up the count combine; the first-match column rides
the same find-first reduction shape used in 51_find_first.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20
TARGET = 0.25  # byte value (quantised float) to search for


def _ref(A, out_count, out_first):
    # Use a strict-strict band matching the LEGO kernel's
    # ``-1e-4 < (A-target) < 1e-4`` form (avoid np.isclose tolerance
    # asymmetry on the boundary).
    d = A - TARGET
    matches = (d > -1e-4) & (d < 1e-4)
    out_count[0] = float(matches.sum())
    if matches.any():
        out_first[0] = float(np.argmax(matches))
    else:
        out_first[0] = float(N)


@benchmark(
    reference=_ref, n_iters=200, warmup=20, rtol=0.0,
    meta={"N": N, "layout_class": "FindByte", "prior_verdict": "NEW"},
)
@cpu_kernel
def find_byte(A: Buffer[N], out_count: Buffer[1], out_first: Buffer[1]):
    cnt = 0.0
    idx_f = 1048576.0
    i_f = 0.0
    for i in range(N):
        # Use a window-equality predicate (stand-in for byte comparison
        # on float-typed memory; the matcher cares about the cmp shape,
        # not the underlying tolerance).
        d = A[i] - 0.25
        if d > -1e-4:
            if d < 1e-4:
                cnt = cnt + 1.0
                if i_f < idx_f:
                    idx_f = i_f
        i_f = i_f + 1.0
    out_count[0] = cnt
    out_first[0] = idx_f


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.uniform(low=0.0, high=1.0, size=N).astype(np.float32)
    # Plant a few exact matches so the search is non-trivial.
    A[100] = TARGET
    A[12345] = TARGET
    A[N // 2] = TARGET
    out_count = np.zeros(1, dtype=np.float32)
    out_first = np.zeros(1, dtype=np.float32)
    return A, out_count, out_first


if __name__ == "__main__":
    A, oc, of = _make_args()
    rec = find_byte.measure(A, oc, of)
    rec["verified"] = find_byte.verify(A,
                                       np.zeros(1, dtype=np.float32),
                                       np.zeros(1, dtype=np.float32))
    print(json.dumps(rec, default=str))
