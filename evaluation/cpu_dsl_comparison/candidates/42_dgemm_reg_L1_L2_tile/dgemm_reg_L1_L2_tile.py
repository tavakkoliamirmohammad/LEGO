"""42_dgemm_reg_L1_L2_tile: DGEMM with register + L1 + L2 tiling.

CASTLE candidate 35. Layout class: Reg+L1+L2 tile.
Prior verdicts: AMD WIN, Intel WIN.

Simplified to a 1-D unit-stride FMA loop ``C[i] = A[i]*B[i] + C[i]``
post-tiling. The original CASTLE candidate is a double-precision GEMM,
but cpu_dsl v1 hard-wires the kernel element type to f32, so this
harness preserves the access pattern (unit-stride FMA after tiling)
in single precision. N=1M to amortize JIT dispatch overhead.
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
    meta={"N": N, "layout_class": "Reg+L1+L2 tile", "prior_verdict": "WIN"},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def dgemm_reg_L1_L2_tile(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        C[i] = A[i] * B[i] + C[i]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = dgemm_reg_L1_L2_tile.measure(A, B, C)
    rec["verified"] = dgemm_reg_L1_L2_tile.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
