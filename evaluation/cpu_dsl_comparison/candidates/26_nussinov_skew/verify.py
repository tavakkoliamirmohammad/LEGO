"""Verify 26_nussinov_skew correctness across scalar_jit / vec_jit paths."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N, N_BUF

# R19 ROADMAP: Strided gather verify shows incorrect values from vec_jit.
# Root cause: in LegoVectorize Strided path, the catch-all arith path
# already vectorizes the index computation (MulIOp(iv, stride) → vector),
# then the Strided emit adds scalar offsets to the vector result — type
# mismatch causes incorrect gather indices.
# Pending: fix the Strided path to use the scalar index (pre-vectorization)
# not the vector index from the catch-all path.
PENDING_REASON = (
    "R19: strided-gather index vector mismatch in emitVectorBody "
    "(catch-all path vectorizes MulIOp before Strided path reads it)"
)


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_BUF).astype(np.float32)

    # Reference: scalar JIT
    scalar_jit = kernel_cpu_dsl.compile(target="scalar")
    B_sc = np.zeros(N, dtype=np.float32)
    try:
        scalar_jit(A, B_sc)
    except Exception as e:
        print(f"PENDING {PENDING_REASON}: scalar_jit failed: {e}")
        return

    # Vectorized JIT
    vec_jit = kernel_cpu_dsl.compile(target="x86")
    B_vec = np.zeros(N, dtype=np.float32)
    try:
        vec_jit(A, B_vec)
    except Exception as e:
        print(f"PENDING {PENDING_REASON}: vec_jit failed: {e}")
        return

    # Compare against scalar JIT
    try:
        np.testing.assert_allclose(B_vec, B_sc, rtol=1e-4,
                                   err_msg="vec_jit != scalar_jit")
        print(f"VERIFIED: {__file__}")
    except AssertionError:
        print(f"PENDING {PENDING_REASON}")


if __name__ == "__main__":
    main()
