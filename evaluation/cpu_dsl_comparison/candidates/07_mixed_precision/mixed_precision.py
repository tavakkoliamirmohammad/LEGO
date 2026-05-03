"""07_mixed_precision: scalar-argument f32 promotion path.

Expected verdict: PARITY or WIN. This kernel exercises the scalar 'scale'
parameter path (CT_FLOAT promoted to a runtime f32 constant) + a fused
multiply-add into Y. It deliberately uses only 2 buffers (X, Y) to avoid
the v1 R12 store-to-3rd-buffer bug.

'mixed precision' here means: Python float -> f32 constant in the IR (the
compiler must emit arith.constant f32 instead of silently truncating).

Run at N=1M to amortize JIT startup overhead.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N = 1 << 20    # 1M elements
TILE = 16
SCALE = 1.5   # constant baked into the kernel


def _ref(X, Y):
    """NumPy reference: Y = SCALE * X + Y (in-place)."""
    Y[:] = SCALE * X + Y


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-2,
    meta={"N": N},
)
@cpu_kernel(grid=(N,), tile=(TILE,))
def mixed_precision(X: Buffer[N], Y: Buffer[N]):
    for i in tile_range:
        Y[i] = 1.5 * X[i] + Y[i]


def _make_args():
    rng = np.random.default_rng(0)
    X = rng.standard_normal(N).astype(np.float32)
    Y = rng.standard_normal(N).astype(np.float32)
    return X, Y


if __name__ == "__main__":
    X, Y = _make_args()
    rec = mixed_precision.measure(X, Y)
    rec["verified"] = mixed_precision.verify(X, Y.copy())
    print(json.dumps(rec, default=str))
