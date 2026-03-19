"""Numba CUDA tiled matmul — LEGO layouts for shared-memory blocking.

Each thread block computes a TPB x TPB tile of the output matrix C.
LEGO's TileBy layout computes the flat offsets used to scatter the result
into C (stored as a 1D array), mirroring the Triton pointer-arithmetic
pattern.  Verified against PyTorch (torch.matmul).

Requires: numba, torch, CUDA GPU.
"""

import math
import numpy as np
import torch
from numba import cuda, float32
from lego.frontends.numba_jit import jit as lego_jit
from lego.core import OrderBy, Row

TPB = 16


@lego_jit
@cuda.jit
def matmul_kernel(A, B, C_flat, M, N, K):
    """Shared-memory blocked matmul writing C via LEGO flat offsets."""

    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TPB + ty
    col = cuda.blockIdx.x * TPB + tx

    tmp = float32(0.)
    for tile in range(0, K, TPB):
        # Load A tile into shared memory
        if row < M and tile + tx < K:
            sA[ty, tx] = A[row, tile + tx]
        else:
            sA[ty, tx] = 0
        # Load B tile into shared memory
        if col < N and tile + ty < K:
            sB[ty, tx] = B[tile + ty, col]
        else:
            sB[ty, tx] = 0
        cuda.syncthreads()

        for k in range(TPB):
            tmp += sA[ty, k] * sB[k, tx]
        cuda.syncthreads()

    # LEGO: compute flat output offset (row-major)
    L_C = OrderBy(Row(M, N)).TileBy([M, N])
    c_off = L_C[row, col]

    if row < M and col < N:
        C_flat[c_off] = tmp


def numba_matmul(A_np, B_np):
    M, K = A_np.shape
    _, N = B_np.shape

    d_A = cuda.to_device(A_np)
    d_B = cuda.to_device(B_np)
    d_C = cuda.device_array(M * N, dtype=A_np.dtype)

    threads = (TPB, TPB)
    blocks = (math.ceil(N / TPB), math.ceil(M / TPB))
    matmul_kernel[blocks, threads](d_A, d_B, d_C, M, N, K)

    return d_C.copy_to_host().reshape(M, N)


if __name__ == "__main__":
    torch.manual_seed(0)
    for size in [64, 128, 256]:
        M, N, K = size, size, size
        a_torch = torch.randn(M, K)
        b_torch = torch.randn(K, N)
        expected = (a_torch @ b_torch).numpy()

        a_np = a_torch.numpy()
        b_np = b_torch.numpy()
        c_np = numba_matmul(a_np, b_np)

        ok = np.allclose(c_np, expected, atol=1e-3)
        print(f"{M}x{K} @ {K}x{N}  match={ok}")
        assert ok, f"Mismatch at size={size}"
    print("PASS: Numba CUDA LEGO matmul matches PyTorch")
