"""29_particlefilter_aosoA: Rodinia particle filter with AoSoA layout.

CASTLE candidate 21. Layout class: AoSoA.
Prior verdicts: AMD LOSS, Intel PARITY.

Approximation: models stride-4 memory access pattern characteristic of AoSoA
(Array-of-Structures-of-Arrays) — every 4th element of a flat buffer is read.
B[i] = A[i*4] * 2.0
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 16384
STRIDE = 4
TILE = 16
N_BUF = N * STRIDE


def _ref(A, B):
    """NumPy reference: B[i] = A[i*STRIDE] * 2.0 (stride-4 gather)."""
    for i in range(N):
        B[i] = A[i * STRIDE] * 2.0


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "AoSoA", "prior_verdict": "PARITY"},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def particlefilter_aosoA(A: Buffer[N_BUF], B: Buffer[N]):
    for i in tile_range:
        B[i] = A[i * 4] * 2.0


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N_BUF).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = particlefilter_aosoA.measure(A, B)
    rec["verified"] = particlefilter_aosoA.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
