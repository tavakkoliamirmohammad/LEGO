"""01_saxpy: y[i] = a*x[i] + y[i].  Trivial unit-stride fused-multiply-add.

Expected verdict: WIN.  Unit-stride load/store with no loop-carried
dependences is the canonical vectorisation sweet spot.  LEGO's
lego-to-x86-vector pipeline emits AVX-512 FMA instructions here.

Run as a script to time a single kernel::

    python saxpy.py

Import to use measure() / verify()::

    from saxpy import saxpy
    rec = saxpy.measure(a, X, Y)   # returns JSON-serializable dict
    ok  = saxpy.verify(a, X, Y)    # correctness check
"""
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

N    = 1 << 20   # 1M elements
TILE = 16        # AVX-512 float32 lane count


def _saxpy_ref(a, X, Y):
    """NumPy reference: Y[:] = a*X + Y (in-place)."""
    Y[:] = a * X + Y


@benchmark(reference=_saxpy_ref, n_iters=1000, warmup=100)
@cpu_kernel(grid=(N,), tile=(TILE,))
def saxpy(a: float, X: Buffer[N], Y: Buffer[N]):
    """SAXPY: Y[i] = a*X[i] + Y[i] — unit-stride, vectorisable."""
    for i in tile_range:
        Y[i] = a * X[i] + Y[i]


def _make_args():
    rng = np.random.default_rng(0)
    a = np.float32(2.5)
    X = rng.standard_normal(N).astype(np.float32)
    Y = rng.standard_normal(N).astype(np.float32)
    return a, X, Y


if __name__ == "__main__":
    import json
    a, X, Y = _make_args()
    rec = saxpy.measure(a, X, Y)
    rec["verified"] = saxpy.verify(a, X, Y.copy())
    print(json.dumps(rec, default=str))
