"""Numba CUDA vecadd — LEGO hello world. Requires numba + CUDA GPU."""

import numpy as np
import torch
from numba import cuda
from lego.frontends.numba_jit import jit as lego_jit
from lego.core import OrderBy, Row


@lego_jit
@cuda.jit
def vecadd_kernel(x, y, z, N):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    L = OrderBy(Row(N)).TileBy([N])
    idx = L[i]
    if idx < N:
        z[idx] = x[idx] + y[idx]


def vecadd(x_np, y_np):
    N = x_np.shape[0]
    d_x = cuda.to_device(x_np)
    d_y = cuda.to_device(y_np)
    d_z = cuda.device_array(N, dtype=x_np.dtype)
    threads = 256
    blocks = (N + threads - 1) // threads
    vecadd_kernel[blocks, threads](d_x, d_y, d_z, N)
    return d_z.copy_to_host()


if __name__ == "__main__":
    torch.manual_seed(0)
    for N in [1024, 8192, 2**16]:
        x_torch = torch.randn(N)
        y_torch = torch.randn(N)
        expected = (x_torch + y_torch).numpy()

        x_np = x_torch.numpy()
        y_np = y_torch.numpy()
        z_np = vecadd(x_np, y_np)

        ok = np.allclose(z_np, expected, atol=1e-5)
        print(f"N={N:>6d}  match={ok}")
        assert ok, f"Mismatch at N={N}"
    print("PASS: Numba CUDA vecadd matches PyTorch")
