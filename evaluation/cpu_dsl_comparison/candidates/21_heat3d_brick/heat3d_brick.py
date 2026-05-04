"""21_heat3d_brick: Polybench heat-3d with brick layout (simplified).

CASTLE candidate 13. Layout class: Brick.
Prior verdicts: AMD MIXED, Intel LOSS.

The heat-3d stencil computes B[x,y,z] = 0.5*(A[x,y,z] + A[x+/-1,y,z] + A[x,y+/-1,z] + A[x,y,z+/-1])
-- identical to the 7-point stencil. The brick layout variation is captured via
the same flat 1D offset-based approach (avoids integer division in the body).
The update is applied in-place over multiple time steps; here we capture one step.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY, NZ = 32, 32, 32
N_FLAT = NX * NY * NZ
_OFFSET = NX * NY
_INNER  = (NX - 2) * NY * NZ
TILE = 16

_NYNZ = NY * NZ
_NZ   = NZ


def _ref(A, B):
    """Heat-3d one step: mirrors the kernel offset arithmetic exactly."""
    o = _OFFSET
    n = _INNER
    B[o:o + n] = 0.5 * (
        A[o:o + n]
        + A[o - _NYNZ : o - _NYNZ + n]
        + A[o + _NYNZ : o + _NYNZ + n]
        + A[o - _NZ   : o - _NZ + n]
        + A[o + _NZ   : o + _NZ + n]
        + A[o - 1     : o - 1 + n]
        + A[o + 1     : o + 1 + n]
    )


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": _INNER, "layout_class": "Brick", "prior_verdict": "LOSS"},
)
@cpu_kernel
def heat3d_brick(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """Heat-3d one step -- flat 1D tiling with compile-time neighbor offsets."""
    for n in range(_INNER):
        flat = n + _OFFSET
        B[flat] = (A[flat]
                   + A[flat - _NYNZ] + A[flat + _NYNZ]   # +/-x
                   + A[flat - _NZ]   + A[flat + _NZ]      # +/-y
                   + A[flat - 1]     + A[flat + 1]         # +/-z
                   ) * 0.5


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_FLAT).astype(np.float32)
    B = np.zeros(N_FLAT, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = heat3d_brick.measure(A, B)
    rec["verified"] = heat3d_brick.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
