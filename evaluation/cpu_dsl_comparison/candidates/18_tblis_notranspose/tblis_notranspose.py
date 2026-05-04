"""18_tblis_notranspose: TBLIS-style tiled GEMM (ikj outer-product pattern).

CASTLE candidate 10 (TBLIS layout class). Prior: AMD WIN, Intel LOSS.

The full TBLIS library uses runtime microkernel selection (blocked GEBP with
arch-specific register tiles); not directly expressible in cpu_dsl v1. This
captures the core layout pattern via ikj outer-product decomposition:

- Vectorize over j (column dim of C and B) — unit stride.
- Inner combined ik loop processes all M*K outer-product contributions.
- A[ik] is broadcast (loop-invariant in j); B[(ik%K)*N+j] and C[(ik//K)*N+j]
  are unit-stride in j.

R20 enabling: outer-loop vectorization with inner reduction scf.for keeps the
inner ik range() loop scalar while the outer j range(N) vectorizes.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = N = K = 64
_MN, _MK, _KN = M * N, M * K, K * N


def _ref(A, B, C):
    """NumPy reference: C += A @ B, with flat ravel layout."""
    C += (A.reshape(M, K) @ B.reshape(K, N)).ravel()


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-3,
    meta={"N": M, "layout_class": "TBLIS", "prior_verdict": "LOSS"},
)
@cpu_kernel(grid=(N,))
def tblis_notranspose(A: Buffer[_MK], B: Buffer[_KN], C: Buffer[_MN]):
    for j in range(N):
        for ik in range(_MK):
            C[(ik // K) * N + j] = C[(ik // K) * N + j] + A[ik] * B[(ik % K) * N + j]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(_MK).astype(np.float32)
    B = rng.standard_normal(_KN).astype(np.float32)
    C = np.zeros(_MN, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = tblis_notranspose.measure(A, B, C)
    rec["verified"] = tblis_notranspose.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
