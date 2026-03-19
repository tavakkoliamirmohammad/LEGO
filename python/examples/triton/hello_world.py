"""Triton vecadd — LEGO hello world. Requires GPU."""

import torch
import triton
import triton.language as tl
from lego.frontends.triton_jit import jit as lego_jit
from lego.core import OrderBy, Row


@lego_jit
@triton.jit
def vecadd_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    L = OrderBy(Row(N)).TileBy([N / BLOCK_SIZE], [BLOCK_SIZE])
    offs = L[pid, :]
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


def vecadd(x, y):
    N = x.numel()
    z = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vecadd_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)
    return z


if __name__ == "__main__":
    torch.manual_seed(0)
    for N in [1024, 8192, 2**16]:
        x = torch.randn(N, device="cuda")
        y = torch.randn(N, device="cuda")
        z = vecadd(x, y)
        expected = x + y
        ok = torch.allclose(z, expected, atol=1e-5)
        print(f"N={N:>6d}  match={ok}")
        assert ok, f"Mismatch at N={N}"
    print("PASS: Triton vecadd matches PyTorch")
