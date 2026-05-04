"""19_bricklib_3d7pt: simplified 3D 7-point stencil (brick layout pattern).

CASTLE candidate 11. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

Full BrickLib API is not bundled and uses an ABI not expressible in cpu_dsl v1.
This version captures the core access pattern: a flat 3D-like stencil over a
1D buffer where each element reads 6 neighbors at fixed positive/negative offsets.

Key: we lay out the buffer as 1D with stride-1 neighbors (+/-1, +/-NX, +/-NX*NY).
This avoids integer division in the kernel body (all address arithmetic is
add/sub of compile-time constants, which is unit-stride or strided gather).

The vectorizable axis is the flat 1D range(_INNER) over the interior.
Boundary effects: we skip the first and last NX*NY rows so indices are always
within bounds for the +/-NX*NY neighbors.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY, NZ = 32, 32, 32
N_FLAT = NX * NY * NZ
# Interior slice: skip first and last layer to avoid OOB in +/-NX neighbor.
_OFFSET = NX * NY        # offset of first safe element
_INNER  = (NX - 2) * NY * NZ  # number of interior elements
TILE = 16

_NYNZ = NY * NZ
_NZ   = NZ


def _ref(A, B):
    """7-point stencil on 3D flat buffer (matches kernel offset arithmetic exactly).

    Mirrors the kernel: B[flat] = 0.5 * sum(A[flat + dz] for dz in 7 offsets)
    over flat in [_OFFSET, _OFFSET + _INNER). Indexing wraps "OOB" reads the
    same way the JIT kernel does (raw flat indices), so vec/scalar/numpy all
    agree bit-for-bit on the full output buffer.
    """
    o = _OFFSET
    n = _INNER
    B[o:o + n] = 0.5 * (
        A[o:o + n]
        + A[o - _NYNZ : o - _NYNZ + n]   # -x
        + A[o + _NYNZ : o + _NYNZ + n]   # +x
        + A[o - _NZ   : o - _NZ + n]     # -y
        + A[o + _NZ   : o + _NZ + n]     # +y
        + A[o - 1     : o - 1 + n]       # -z
        + A[o + 1     : o + 1 + n]       # +z
    )


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": _INNER, "layout_class": "Brick", "prior_verdict": "LOSS"},
)
@cpu_kernel
def bricklib_3d7pt(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """3D 7-point stencil -- flat 1D tiling over interior, offset-based addressing.

    All address arithmetic uses compile-time constants (unit-stride and strided
    gathers of +/-1, +/-NZ, +/-NX*NY). No integer division in the kernel body.
    """
    for n in range(_INNER):
        # n is the flat interior index; flat buffer address = n + _OFFSET.
        flat = n + _OFFSET
        B[flat] = (A[flat]
                   + A[flat - _NYNZ]   # -x face neighbor (stride NX*NY)
                   + A[flat + _NYNZ]   # +x face neighbor
                   + A[flat - _NZ]     # -y neighbor (stride NY)
                   + A[flat + _NZ]     # +y neighbor
                   + A[flat - 1]       # -z neighbor (unit stride)
                   + A[flat + 1]       # +z neighbor
                   ) * 0.5


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_FLAT).astype(np.float32)
    B = np.zeros(N_FLAT, dtype=np.float32)
    return A, B


if __name__ == "__main__":
    A, B = _make_args()
    rec = bricklib_3d7pt.measure(A, B)
    rec["verified"] = bricklib_3d7pt.verify(A, np.zeros_like(B))
    print(json.dumps(rec, default=str))
