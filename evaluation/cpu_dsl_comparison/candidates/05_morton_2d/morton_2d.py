"""05_morton_2d: read from a Z-Morton encoded 1-D buffer using inline bit ops.

Expected verdict: PARITY or WIN vs gcc on the gather path, since both must
perform irregular gather reads. GCC can't auto-vectorize Morton decode; LEGO
emits vector.gather via the NonAffine path.

The kernel computes a simplified 1-D Morton-style index transform:
  morton(i) = (spread even bits of i) | (spread odd bits of i << 1)
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 65536        # 256 * 256 flat array (matches the c_baseline's expected N)
TILE = 16


def _ref(A, B):
    """NumPy reference: 1-D Morton-style index scramble."""
    indices = np.arange(N, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = ti | (tj << 1)
    morton = morton & (N - 1)  # mask to in-bounds
    B[:] = A[morton]


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N},
)
@cpu_kernel(grid=(N,))
def morton_2d(A: Buffer[N], B: Buffer[N]):
    for i in range(N):
        # Real Morton decode using bitwise ops now supported in the DSL.
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = ti | (tj << 1)
        morton = morton & (N - 1)
        B[i] = A[morton]


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = morton_2d.measure(A, B)
    rec["verified"] = morton_2d.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
