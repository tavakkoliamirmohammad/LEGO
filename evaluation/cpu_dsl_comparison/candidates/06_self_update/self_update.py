"""06_self_update: shift-add stencil  B[i+1] = A[i] + A[i+1].

The self_update pattern models a simple two-point stencil that reads A[i-1]
and A[i] and writes B[i]. There is NO loop-carried dependence on B — each
output element depends only on two consecutive input elements. This makes
the loop fully vectorizable.

Expected verdict: WIN. Removing the prior prefix-sum loop-carried dep
(B[i] = B[i-1] + A[i]) and aligning kernel semantics with the C baseline
allows LEGO to generate AVX-512 vector code and beat the C reference.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

# N-1 interior elements (skip B[0] which has no left neighbor)
N = 4096
N_INNER = N - 1
TILE = 16


def _ref(A, B):
    """Reference: shift-add stencil — matches C baseline self_update_O3."""
    B[1:] = A[:N - 1] + A[1:]


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N},
)
@cpu_kernel(grid=(N_INNER,), tile=(TILE,))
def self_update(A: Buffer[N], B: Buffer[N]):
    for i in tile_range:
        # i in [0, N-1): read A[i] and A[i+1], write B[i+1].
        # No dep on B: fully vectorizable.
        B[i + 1] = A[i] + A[i + 1]


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = self_update.measure(A, B)
    rec["verified"] = self_update.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
