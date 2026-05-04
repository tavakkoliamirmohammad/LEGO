"""14_2mm_reg_L1_tile: 2MM proxy with register + L1 tiling (unit-stride).

CASTLE candidate 6. Layout class: Reg+L1 tile.
Prior verdicts: AMD WIN, Intel WIN.

Uses N=1M to amortize JIT call overhead. Unit-stride C=A*B+C kernel.
The prior CASTLE measurement used the actual polybench code (with real tiling);
this harness uses a simplified 1-D unit-stride kernel to test the vectorizer
on the same access pattern type (unit-stride innermost loop).
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
    n_iters=1000, warmup=100, rtol=1e-3,
    meta={"N": N, "layout_class": "Reg+L1 tile", "prior_verdict": "WIN"},
)
@cpu_kernel(grid=(N,))
def kernel_2mm_reg_L1_tile(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
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
    rec = kernel_2mm_reg_L1_tile.measure(A, B, C)
    rec["verified"] = kernel_2mm_reg_L1_tile.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
