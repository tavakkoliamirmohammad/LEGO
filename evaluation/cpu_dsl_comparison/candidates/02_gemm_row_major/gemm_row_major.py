"""02_gemm_row_major: C[i,j] += sum_k A[i,k]*B[k,j], row-major.

Uses LEGO's layout-aware ``Buffer[M, K]`` form so the kernel writes
``A[i, k]`` instead of hand-computed flat indices.  ``Buffer[M, K]``
defaults to a ``Row(M, K)`` layout; the multi-dim subscript lowers
through ``apply`` to ``i*K + k`` automatically.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = 64
N = 64
K = 64


def _ref(A, B, C):
    """NumPy reference: C += A @ B."""
    C += A @ B


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-3,
    meta={"M": M, "N": N, "K": K},
)
@cpu_kernel
def gemm_row_major(A: Buffer[M, K], B: Buffer[K, N], C: Buffer[M, N]):
    for i in range(M):
        for k in range(K):
            for j in range(N):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = gemm_row_major.measure(A, B, C)
    rec["verified"] = gemm_row_major.verify(
        A, B, np.zeros((M, N), dtype=np.float32))
    print(json.dumps(rec, default=str))
