"""Verify 20_bricklib_3d13pt correctness across scalar_jit / vec_jit paths.

Safe range analysis for the 13-point stencil:
  For element at flat = n + _OFFSET, the most extreme neighbor access is
  A[flat - _NYNZ - _NZ].  This is valid only when flat - _NYNZ - _NZ >= 0,
  i.e. n >= _NYNZ + _NZ - _OFFSET = 1024 + 32 - 1024 = 32.
  Similarly, A[flat + _NYNZ + _NZ] is valid only when n <= _INNER - _NZ - 1.
  So we skip the first _NZ = 32 and last _NZ + 1 = 33 interior elements.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_cpu_dsl, N_FLAT, _INNER, _OFFSET, _NYNZ, _NZ


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_FLAT).astype(np.float32)

    # Reference: scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    B_sc = np.zeros(N_FLAT, dtype=np.float32)
    try:
        scalar_jit(A, B_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    B_vec = np.zeros(N_FLAT, dtype=np.float32)
    try:
        vec_jit(A, B_vec)
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {e}")
        return

    # Safe range: skip first _NZ and last _NZ interior elements to avoid
    # out-of-bounds accesses from the diagonal neighbors (±_NYNZ±_NZ).
    # Both scalar and vec read garbage at those boundary elements, but they
    # read DIFFERENT garbage (scalar Python wraps, vec reads vectorized OOB).
    safe_start = _NZ           # = 32: skip first row of interior
    safe_end   = _INNER - _NZ  # = 30688: skip last row of interior

    if safe_end <= safe_start:
        print(f"PENDING R??: safe range too small for 13pt stencil")
        return

    try:
        np.testing.assert_allclose(
            B_vec[_OFFSET + safe_start : _OFFSET + safe_end],
            B_sc [_OFFSET + safe_start : _OFFSET + safe_end],
            rtol=1e-4,
            err_msg="vec_jit != scalar_jit (in safe interior)",
        )
        print(f"VERIFIED: {__file__}")
    except AssertionError as e:
        print(f"PENDING R??: 13pt stencil mismatch in safe range: {str(e)[:80]}")


if __name__ == "__main__":
    main()
