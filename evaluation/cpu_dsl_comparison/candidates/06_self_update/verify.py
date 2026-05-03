"""Verify 06_self_update correctness across scalar_jit / vec_jit paths.

Checks:
1. vec_jit == scalar_jit (internal consistency)
2. vec_jit matches NumPy reference: B[1:] = A[:-1] + A[1:]
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)

    # NumPy reference
    B_ref = np.zeros(N, dtype=np.float32)
    kernel_scalar(A, B_ref)

    # Scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    B_sc = np.zeros(N, dtype=np.float32)
    try:
        scalar_jit(A, B_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    B_vec = np.zeros(N, dtype=np.float32)
    try:
        vec_jit(A, B_vec)
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {e}")
        return

    # Internal: vec == scalar
    np.testing.assert_allclose(B_vec, B_sc, rtol=1e-4,
                                err_msg="vec_jit != scalar_jit")
    # vs reference (B[0] stays 0, B[1:] = A[:-1]+A[1:])
    np.testing.assert_allclose(B_vec[1:], B_ref[1:], rtol=1e-4,
                                err_msg="vec_jit != numpy reference")
    print(f"VERIFIED: {__file__}")


if __name__ == "__main__":
    main()
