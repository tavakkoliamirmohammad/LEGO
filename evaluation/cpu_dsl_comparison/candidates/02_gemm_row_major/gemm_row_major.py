"""02_gemm_row_major: C[i,j] += sum_k A[i,k]*B[k,j], row-major blocked.

Expected verdict: WIN.  Tile over i (rows), with j (columns) as the innermost
unit-stride loop. The j-loop is unit-stride in B[k*N+j] and C[i*N+j] —
vectorizable. Avoids integer division on the induction variable that would
trigger NonAffine gather emission.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = 64
N = 64
K = 64
TILE_M = 8
_MK = M * K
_KN = K * N
_MN = M * N
_N = N   # column count visible in outer scope


def _ref(A, B, C):
    """NumPy reference: C += A @ B (accumulate into C); flat ravel layout."""
    C += (A.reshape(M, K) @ B.reshape(K, N)).ravel()


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-3,
    meta={"N": M},
)
@cpu_kernel(grid=(M,), tile=(TILE_M,))
def gemm_row_major(A: Buffer[_MK], B: Buffer[_KN], C: Buffer[_MN]):
    for i in tile_range:
        for k in range(K):
            for j in range(_N):
                C[i * _N + j] = C[i * _N + j] + A[i * K + k] * B[k * _N + j]


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(_MK).astype(np.float32)
    B = rng.standard_normal(_KN).astype(np.float32)
    C = np.zeros(_MN, dtype=np.float32)
    return A, B, C


if __name__ == "__main__":
    A, B, C = _make_args()
    rec = gemm_row_major.measure(A, B, C)
    rec["verified"] = gemm_row_major.verify(A, B, np.zeros_like(C))
    print(json.dumps(rec, default=str))
