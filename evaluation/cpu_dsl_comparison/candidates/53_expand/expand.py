"""53_expand: stream-expand — copy A[i] into B[j++] under predicate.

Tier-2 ``expand/decompress`` pattern (inverse of compaction at the
shape level — both are gated copies into a monotonic output index).

    j = 0
    for i in range(N):
        if A[i] > 0.0:
            B[j] = A[i]
            j = j + 1
    out[0] = j

The cpu_dsl path doesn't currently surface a ``vector.compressstore``
recognition for scf-level ``j`` iter-args (the LegoVectorize compaction
pass works on the Lego dialect form, not on raw scf-affine).  So the
cpu_dsl form here is expected to fall through to scalar; clang -O3
emits scalar too because the dependent ``j++`` defeats unconditional
vectorisation.  Empirical parity verifies the path stays correct.
A future generalisation would lower this scf form to vector.compress*.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20


def _ref(A, B, out):
    mask = A > 0.0
    cnt = int(mask.sum())
    B[:cnt] = A[mask]
    out[0] = float(cnt)


@benchmark(
    reference=_ref, n_iters=200, warmup=20, rtol=0.0,
    meta={"N": N, "layout_class": "Expand", "prior_verdict": "NEW"},
)
@cpu_kernel
def expand(A: Buffer[N], B: Buffer[N], out: Buffer[1]):
    # ``j`` is used both as a memory subscript (needs INDEX) and as the
    # final count written into a float buffer.  We track both forms in
    # parallel — ``j`` for indexing, ``j_f`` for the f32 store at end.
    j = 0
    j_f = 0.0
    for i in range(N):
        if A[i] > 0.0:
            B[j] = A[i]
            j = j + 1
            j_f = j_f + 1.0
    out[0] = j_f


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    return A, B, out


if __name__ == "__main__":
    A, B, out = _make_args()
    rec = expand.measure(A, B, out)
    rec["verified"] = expand.verify(A, np.zeros(N, dtype=np.float32),
                                    np.zeros(1, dtype=np.float32))
    print(json.dumps(rec, default=str))
