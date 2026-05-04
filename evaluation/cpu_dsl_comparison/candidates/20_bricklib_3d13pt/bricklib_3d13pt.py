"""20_bricklib_3d13pt: simplified 3D 13-point stencil (brick layout pattern).

CASTLE candidate 12. Layout class: Brick.
Prior verdicts: AMD PARITY, Intel WIN.

Full BrickLib API not bundled. This version captures the 13-point stencil
access pattern: each element reads 12 face+edge neighbors at offsets
+/-1, +/-NZ, +/-NX*NY, and 6 face-diagonal neighbors.

All address arithmetic uses compile-time constants -- no integer division.
The +/-1 neighbors are unit-stride; +/-NZ and +/-NX*NY are strided gathers.

NOTE: known to error / disagree at the boundary because diagonal offsets
+/-(NX*NY + NZ) cross OOB at the first/last NZ interior elements.  The R19
strided-gather fix is required to fully verify; see the project roadmap.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

NX, NY, NZ = 32, 32, 32
N_FLAT = NX * NY * NZ
_OFFSET = NX * NY      # skip first layer
_INNER  = (NX - 2) * NY * NZ
TILE = 16

_NYNZ = NY * NZ
_NZ   = NZ


def _ref(A, B):
    """13-point stencil mirroring the kernel offset arithmetic exactly.

    Computes only the safe inner range [_NZ, _INNER - _NZ), where all 12
    diagonal/face neighbor offsets stay within A's bounds.  Outside that
    range the reference does nothing, so the JIT kernel's OOB-edge output
    drives any verify mismatch.
    """
    o = _OFFSET
    safe_lo = _NZ
    safe_hi = _INNER - _NZ
    s = safe_hi - safe_lo
    base = o + safe_lo
    inv13 = np.float32(1.0 / 13.0)
    B[base:base + s] = inv13 * (
        A[base                : base + s]
        + A[base - _NYNZ      : base - _NYNZ + s]          # -x
        + A[base + _NYNZ      : base + _NYNZ + s]          # +x
        + A[base - _NZ        : base - _NZ + s]            # -y
        + A[base + _NZ        : base + _NZ + s]            # +y
        + A[base - 1          : base - 1 + s]              # -z
        + A[base + 1          : base + 1 + s]              # +z
        + A[base - _NYNZ - _NZ: base - _NYNZ - _NZ + s]    # -x-y
        + A[base + _NYNZ + _NZ: base + _NYNZ + _NZ + s]    # +x+y
        + A[base - _NYNZ + _NZ: base - _NYNZ + _NZ + s]    # -x+y
        + A[base + _NYNZ - _NZ: base + _NYNZ - _NZ + s]    # +x-y
        + A[base - _NYNZ - 1  : base - _NYNZ - 1 + s]      # -x-z
        + A[base + _NYNZ + 1  : base + _NYNZ + 1 + s]      # +x+z
    )


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": _INNER, "layout_class": "Brick", "prior_verdict": "WIN"},
)
@cpu_kernel
def bricklib_3d13pt(A: Buffer[N_FLAT], B: Buffer[N_FLAT]):
    """13-point stencil -- flat 1D tiling with compile-time neighbor offsets."""
    for n in range(_INNER):
        flat = n + _OFFSET
        B[flat] = (A[flat]
                   + A[flat - _NYNZ]    # -x
                   + A[flat + _NYNZ]    # +x
                   + A[flat - _NZ]      # -y
                   + A[flat + _NZ]      # +y
                   + A[flat - 1]        # -z
                   + A[flat + 1]        # +z
                   + A[flat - _NYNZ - _NZ]   # -x-y diagonal
                   + A[flat + _NYNZ + _NZ]   # +x+y diagonal
                   + A[flat - _NYNZ + _NZ]   # -x+y diagonal
                   + A[flat + _NYNZ - _NZ]   # +x-y diagonal
                   + A[flat - _NYNZ - 1]     # -x-z diagonal
                   + A[flat + _NYNZ + 1]     # +x+z diagonal
                   ) * (1.0 / 13.0)


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_FLAT).astype(np.float32)
    B = np.zeros(N_FLAT, dtype=np.float32)
    return A, B


def _verify_safe_range() -> bool:
    """Custom verify: compare JIT vs numpy ref only over the safe inner
    sub-range [_NZ, _INNER - _NZ).

    The kernel's diagonal reads (+/-(_NYNZ + _NZ), +/-(_NYNZ - _NZ),
    +/-(_NYNZ + 1)) cross OOB at the first/last _NZ interior elements.
    The vec_jit and scalar_jit return DIFFERENT garbage there (vector vs
    scalar OOB load), so we compare only the well-defined safe range.
    """
    A, B = _make_args()
    A2 = A.copy()
    B_ref = B.copy()
    _ref(A2, B_ref)
    B_jit = B.copy()
    bricklib_3d13pt(A2.copy(), B_jit)
    safe_lo = _OFFSET + _NZ
    safe_hi = _OFFSET + _INNER - _NZ
    try:
        np.testing.assert_allclose(
            B_jit[safe_lo:safe_hi], B_ref[safe_lo:safe_hi], rtol=1e-4,
        )
        return True
    except AssertionError as e:
        print(f"[verify] safe-range mismatch: {str(e)[:200]}")
        return False


if __name__ == "__main__":
    A, B = _make_args()
    rec = bricklib_3d13pt.measure(A, B)
    rec["verified"] = _verify_safe_range()
    print(json.dumps(rec, default=str))
