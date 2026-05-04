"""37_stencil_nonpow2_brick: 5-point 2D stencil on a non-pow-2 grid.

CASTLE candidate 29. Layout class: Brick+non-pow2.
Prior verdicts: AMD LOSS, Intel LOSS.

Full BrickLib not bundled. This version captures the non-power-of-2 brick
size effect: 30x30 2D grid (not a power of 2) flattened to 1D, with the
inner trip count ``_INNER = 28*30 = 840`` driving a 5-point cross stencil
``B[i] = 0.25*(A[i-NY] + A[i+NY] + A[i-1] + A[i+1])``.

The non-pow-2 ``_INNER`` exercises the vectorizer's tail-loop handling.
``_ref`` matches the kernel's exact flat-loop semantics (not a 2D
``b[1:-1, 1:-1]`` slice) so full-array verify compares cleanly.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

# Non-pow2 dimensions: 30x30 interior
NX, NY = 30, 30
N_FLAT = NX * NY
_OFFSET = NY                # skip first row
_INNER = (NX - 2) * NY      # 28 * 30 = 840
TILE = 8                    # smaller tile for non-pow2

_NY = NY


def _ref(A, B):
    """NumPy reference matching the kernel's flat-loop semantics exactly.

    For each n in [0, _INNER), with flat = n + _OFFSET, computes
    ``B[flat] = 0.25*(A[flat-_NY] + A[flat+_NY] + A[flat-1] + A[flat+1])``.
    """
    flats = np.arange(_OFFSET, _OFFSET + _INNER)
    B[flats] = 0.25 * (
        A[flats - _NY] + A[flats + _NY] + A[flats - 1] + A[flats + 1]
    )


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": _INNER, "layout_class": "Brick+non-pow2", "prior_verdict": "LOSS"},
)
@cpu_kernel(grid=(_INNER,))
def stencil_nonpow2_brick(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """5-point 2D stencil on non-pow2 grid — flat 1D tiling."""
    for n in range(_INNER):
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
    rec = stencil_nonpow2_brick.measure(A, B)
    rec["verified"] = stencil_nonpow2_brick.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
