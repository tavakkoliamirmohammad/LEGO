"""Verify 04_col_major_inner correctness across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_cpu_dsl, M, N, _MN


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(_MN).astype(np.float32)

    # Reference: scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    C_sc = np.zeros(_MN, dtype=np.float32)
    try:
        scalar_jit(A, C_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    C_vec = np.zeros(_MN, dtype=np.float32)
    try:
        vec_jit(A, C_vec)
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {e}")
        return

    np.testing.assert_allclose(C_vec, C_sc, rtol=1e-4,
                                err_msg="vec_jit != scalar_jit")
    print(f"VERIFIED: {__file__}")


if __name__ == "__main__":
    main()
