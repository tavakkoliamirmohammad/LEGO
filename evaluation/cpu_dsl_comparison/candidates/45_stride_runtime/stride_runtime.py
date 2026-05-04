"""45_stride_nonpow2: B[i] = A[i * 7] * 2.0 — non-power-of-2 stride.

LEGO's existing R20 deinterleave path handles strides {2, 4, 8} via
``vector.deinterleave``; for other strides it emits ``vector.gather``.
Clang's auto-vectoriser also has heuristics tuned for power-of-2 strides;
non-pow-2 strides typically fall through to a scalar loop.

Targets clang's blind spot: stride-7 access is a real pattern (e.g.,
red-black ordering, packed structures with 7-field structs), and
neither auto-vectoriser handles it well.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 18         # 256K logical elements
STRIDE = 7          # non-power-of-2 stride
N_PHYS = N * STRIDE


def _ref(A, B):
    """NumPy reference: B[:] = A[::7] * 2.0."""
    B[:] = A[::STRIDE][:N] * 2.0


@benchmark(
    reference=_ref, n_iters=500, warmup=50, rtol=1e-5,
    meta={"N": N, "layout_class": "Stride7", "prior_verdict": "NEW"},
)
@cpu_kernel
def stride_runtime(A: Buffer[N_PHYS], B: Buffer[N]):
    for i in range(N):
        B[i] = A[i * STRIDE] * 2.0


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_PHYS).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = stride_runtime.measure(A, B)
    rec["verified"] = stride_runtime.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
