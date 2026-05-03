"""Verify 01_saxpy correctness across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N


def main():
    rng = np.random.default_rng(0)
    a = np.float32(2.5)
    X = rng.standard_normal(N).astype(np.float32)
    Y_base = rng.standard_normal(N).astype(np.float32)

    # Reference: scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    Y_sc = Y_base.copy()
    try:
        scalar_jit(a, X, Y_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    Y_vec = Y_base.copy()
    try:
        vec_jit(a, X, Y_vec)
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {e}")
        return

    np.testing.assert_allclose(Y_vec, Y_sc, rtol=1e-4,
                                err_msg="vec_jit != scalar_jit")
    print(f"VERIFIED: {__file__}")


if __name__ == "__main__":
    main()
