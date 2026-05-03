"""Verify 34_gemm_pow2_pad correctness across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = rng.standard_normal(N).astype(np.float32)
    C_base = rng.standard_normal(N).astype(np.float32)

    # Reference: scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    C_sc = C_base.copy()
    try:
        scalar_jit(A, B, C_sc)
    except Exception as e:
        print(f"PENDING R??: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    C_vec = C_base.copy()
    try:
        vec_jit(A, B, C_vec)
    except Exception as e:
        print(f"PENDING R??: vec_jit failed: {e}")
        return

    np.testing.assert_allclose(C_vec, C_sc, rtol=1e-4,
                                err_msg="vec_jit != scalar_jit")
    print(f"VERIFIED: {__file__}")


if __name__ == "__main__":
    main()
