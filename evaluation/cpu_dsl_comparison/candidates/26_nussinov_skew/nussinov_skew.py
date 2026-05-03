"""26_nussinov_skew: NPDP Nussinov with skew tiling (unit-stride after skew transform).

CASTLE candidate 18. Layout class: Skew tile.
Prior verdicts: AMD WIN, Intel WIN.

Approximation: models stride-2 memory access pattern characteristic of
Skew tile.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

# kernel.py used N=4096; measure.py used N=4096 (same).  Keep N=4096 here.
N = 4096
STRIDE = 2
TILE = 16
N_BUF = N * STRIDE


def _ref(A, B):
    """Vectorized strided gather: B[i] = A[i * STRIDE] * 2.0."""
    np.multiply(A[::STRIDE], 2.0, out=B)


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Skew tile", "prior_verdict": "WIN"},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def nussinov_skew(A: Buffer[N_BUF], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i * 2] * 2.0


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N_BUF).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = nussinov_skew.measure(A, B)
    rec["verified"] = nussinov_skew.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
