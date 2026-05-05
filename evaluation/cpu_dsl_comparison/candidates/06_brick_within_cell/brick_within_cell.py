"""06_brick_within_cell: within-brick vectorisation.

Run at N=1M to amortize JIT startup overhead.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20    # 1M elements
BRICK = 8


def _ref(A, B):
    """NumPy scalar reference: B = A * 2.0 + 1.0."""
    B[:] = A * 2.0 + 1.0


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": N},
)
@cpu_kernel
def brick_within_cell(A: Buffer[N], B: Buffer[N]):
    for i in range(N):
        B[i] = A[i] * 2.0 + 1.0


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = brick_within_cell.measure(A, B)
    rec["verified"] = brick_within_cell.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
