"""Minimal Triton JIT hello world — requires GPU."""
import torch, triton, triton.language as tl
from lego.frontends.triton_jit import jit as lego_jit
from lego.core import OrderBy, Row

@lego_jit
@triton.jit
def vecadd(x_ptr, y_ptr, z_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    L = OrderBy(Row(N)).TileBy([N // BLOCK], [BLOCK])
    offs = L[pid, :]
    mask = offs < N
    tl.store(z_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)

x = torch.randn(1024, device='cuda'); y = torch.randn(1024, device='cuda')
z = torch.empty_like(x)
vecadd[(4,)](x, y, z, 1024, BLOCK=256)
print(f"Match: {torch.allclose(z, x + y)}")
