"""09_gemm_zmorton: GEMM with Z-Morton (Z-order) layout index swizzle.

CASTLE candidate 01 — polybench-gemm-zmorton.
Layout class: Z-Morton.
Prior verdicts: AMD WIN (2.07–3.30x), Intel WIN (1.15–2.27x).

The Z-Morton layout stores matrix elements in Z-order (Morton code) to improve
2D spatial locality. This kernel tests the 1-D gather path with a GEMM-style
accumulation to verify correctness of gather + FMA paths.

Simplified GEMM-like gather: C[i] += A[morton(i)] * B[i].
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 16  # 64K — large enough to stress the gather path
TILE = 16


def _ref(A, B, C):
    """NumPy scalar equivalent: C += A[morton] * B."""
    indices = np.arange(N, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N - 1)
    C[:] += A[morton] * B


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Z-Morton", "prior_verdict": "WIN"},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def gemm_zmorton(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = C[i] + A[morton] * B[i]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = gemm_zmorton.measure(A, B, C)
    rec["verified"] = gemm_zmorton.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
