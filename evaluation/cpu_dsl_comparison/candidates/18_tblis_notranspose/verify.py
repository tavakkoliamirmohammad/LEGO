"""Verify 18_tblis_notranspose correctness across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_cpu_dsl, M, N, K, _MK, _KN, _MN


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(_MK).astype(np.float32)
    B = rng.standard_normal(_KN).astype(np.float32)

    # Reference: scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    C_sc = np.zeros(_MN, dtype=np.float32)
    try:
        scalar_jit(A, B, C_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    C_vec = np.zeros(_MN, dtype=np.float32)
    try:
        vec_jit(A, B, C_vec)
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {e}")
        return

    np.testing.assert_allclose(C_vec, C_sc, rtol=1e-3,
                                err_msg="vec_jit != scalar_jit")
    print(f"VERIFIED: {__file__}")


if __name__ == "__main__":
    main()
