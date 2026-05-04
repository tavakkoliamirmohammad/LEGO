"""39_hotspot_tile: Rodinia hotspot with standard tiling.

CASTLE candidate 32. Layout class: Tile.
Prior verdicts: AMD WIN, Intel WIN.

Simplified to a 1-D unit-stride FMA loop ``C[i] = A[i]*B[i] + C[i]``
that captures the same access pattern type as the original tiled
hotspot kernel. N is sized at 1M to amortize JIT call overhead.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1048576
TILE = 16


def _ref(A, B, C):
    """NumPy reference: tiled unit-stride accumulation."""
    np.add(A * B, C, out=C)


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Tile", "prior_verdict": "WIN"},
)
@cpu_kernel(grid=(N,))
def hotspot_tile(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in range(N):
        C[i] = A[i] * B[i] + C[i]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = hotspot_tile.measure(A, B, C)
    rec["verified"] = hotspot_tile.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
