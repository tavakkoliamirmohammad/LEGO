"""22_jacobi2d_brick: Polybench jacobi-2d with brick layout (simplified).

CASTLE candidate 14. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

Jacobi-2d: B[i,j] = 0.25 * (A[i-1,j] + A[i+1,j] + A[i,j-1] + A[i,j+1]).
Brick layout approximation: flat 1D buffer with stride-N y-neighbors and
stride-1 x-neighbors. No integer division in the kernel body.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY = 256, 256
N_FLAT = NX * NY
# Interior: skip first and last rows (for +/-NX neighbors)
_OFFSET = NX      # first safe row
_INNER  = (NX - 2) * NY
TILE = 16

_NY = NY


def _ref(A, B):
    """Jacobi-2d one step: mirrors the kernel offset arithmetic exactly."""
    o = _OFFSET
    n = _INNER
    B[o:o + n] = 0.25 * (
        A[o - _NY : o - _NY + n]   # -row
        + A[o + _NY : o + _NY + n] # +row
        + A[o - 1   : o - 1 + n]   # -col
        + A[o + 1   : o + 1 + n]   # +col
    )


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": _INNER, "layout_class": "Brick", "prior_verdict": "LOSS"},
)
@cpu_kernel(grid=(_INNER,), tile=(TILE,))
def jacobi2d_brick(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """Jacobi-2d -- flat 1D tiling over interior elements."""
    for n in tile_range:
        flat = n + _OFFSET
        B[flat] = (A[flat - _NY]    # -row neighbor (strided)
                   + A[flat + _NY]  # +row neighbor (strided)
                   + A[flat - 1]    # -col neighbor (unit stride)
                   + A[flat + 1]    # +col neighbor (unit stride)
                   ) * 0.25


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_FLAT).astype(np.float32)
    B = np.zeros(N_FLAT, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = jacobi2d_brick.measure(A, B)
    rec["verified"] = jacobi2d_brick.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
