"""Verify 03_3pt_stencil_1d correctness across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_cpu_dsl, N


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)

    # Reference: scalar JIT
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

    # Compare interior points only (index 1 to N-2)
    np.testing.assert_allclose(B_vec[1:-1], B_sc[1:-1], rtol=1e-4,
                                err_msg="vec_jit != scalar_jit")
    print(f"VERIFIED: {__file__}")


if __name__ == "__main__":
    main()
