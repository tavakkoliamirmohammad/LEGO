"""04_col_major_inner: C[i,j] = A[j,i] (transpose copy; column-major inner loop).

Expected verdict: PARITY or LOSS.  The inner loop strides through column-major
storage (stride != 1), so vectorisation yields gather-style loads and the
bandwidth benefit is partial.  Representative of RFP-class candidates and
block-cyclic candidates which showed borderline results.

v1 limitation: store target must also appear as a load source in the same
kernel body to avoid a vector-store type mismatch in the lowering. The
read-modify-write form C[j] = A[col*M+i] + C[j]*0.0 makes C both a load and a
store so the vectoriser matches types correctly.
"""
import json
import numpy as np
from lego.testing import benchmark
from lego.backend.cpu_dsl import cpu_kernel, Buffer

M = 256
N = 256
_MN = M * N
TILE = 16


def _ref(A, C):
    """NumPy reference: C = A.T (out-of-place transpose), flat layout."""
    A_2d = A.reshape(M, N)
    C_2d = C.reshape(M, N)
    np.copyto(C_2d, A_2d.T)


@benchmark(
    reference=_ref,
    n_iters=1000, warmup=100, rtol=1e-4,
    meta={"N": M},
)
@cpu_kernel
def col_major_inner(A: Buffer[_MN], C: Buffer[_MN]):
    for j in range(_MN):
        i = j // N
        col = j - i * N
        # Read-modify-write form: add C[j]*0.0 so C is both read and written.
        C[j] = A[col * M + i] + C[j] * 0.0


def _make_args():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(_MN).astype(np.float32)
    C = np.zeros(_MN, dtype=np.float32)
    return A, C


if __name__ == "__main__":
    A, C = _make_args()
    rec = col_major_inner.measure(A, C)
    rec["verified"] = col_major_inner.verify(A, np.zeros_like(C))
    print(json.dumps(rec, default=str))
