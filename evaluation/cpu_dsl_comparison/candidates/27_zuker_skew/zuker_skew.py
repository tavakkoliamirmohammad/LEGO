"""27_zuker_skew: NPDP Zuker with skew tiling — stride-2 gather pattern.

CASTLE candidate 19. Layout class: Skew tile.
Prior verdicts: AMD WIN, Intel LOSS (L3 capacity).

Approximation: models stride-2 deinterleave-style memory access pattern
characteristic of the Zuker skew tiling approximation.
B[i] = A[i*2] * 2.0
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 4096
STRIDE = 2
TILE = 16
N_BUF = N * STRIDE


def _ref(A, B):
    """NumPy reference: B[i] = A[i*STRIDE] * 2.0 (stride-2 gather)."""
    for i in range(N):
        B[i] = A[i * STRIDE] * 2.0


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Skew tile", "prior_verdict": "LOSS"},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def zuker_skew(A: Buffer[N_BUF], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i * 2] * 2.0


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N_BUF).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = zuker_skew.measure(A, B)
    rec["verified"] = zuker_skew.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
