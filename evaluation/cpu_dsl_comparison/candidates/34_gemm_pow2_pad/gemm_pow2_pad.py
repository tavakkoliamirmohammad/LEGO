"""34_gemm_pow2_pad: Polybench GEMM with pow-2 padding (Morton gather).

CASTLE candidate 26. Layout class: Pow2 pad.
Prior verdicts: AMD WIN, Intel LOSS (L3 geometry difference).

Models the Z-Morton gather access pattern that arises with pow-2 padded GEMM:
C[i] = A[morton(i)] * B[i] + C[i]
where morton(i) interleaves the low bits of i.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 65536
TILE = 16


def _ref(A, B, C):
    """NumPy reference: Morton gather + accumulate."""
    idx = np.arange(N, dtype=np.int32)
    ti = idx & 0x5555
    tj = (idx >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N - 1)
    np.add(A[morton] * B, C, out=C)


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N, "layout_class": "Pow2 pad", "prior_verdict": "LOSS"},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def gemm_pow2_pad(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N - 1)
        C[i] = A[morton] * B[i] + C[i]


def _make_args():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C = np.zeros(N, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = gemm_pow2_pad.measure(A, B, C)
    rec["verified"] = gemm_pow2_pad.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
