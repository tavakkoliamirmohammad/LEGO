"""Vector addition using LEGO block_ptr mode.

Generates tl.make_block_ptr() calls instead of pointer arithmetic,
enabling TMA acceleration on Hopper+ GPUs. The same LEGO layout
expression works for both modes -- only the decorator flag changes.
"""

import torch
import triton
import triton.language as tl
from lego.core import *
from lego.frontends.triton_jit import jit as lego_jit


# -- LEGO block_ptr kernel -------------------------------------------------
# use_block_ptr=True: generates tl.make_block_ptr / tl.advance
@lego_jit(use_block_ptr=True)
@triton.jit
def vecadd_block_ptr_kernel(x_ptr, y_ptr, z_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    # Same LEGO layout as the pointer-arithmetic version.
    # With use_block_ptr=True, `x_ptrs = x_ptr + L[pid, :]` generates
    # tl.make_block_ptr(base=x_ptr, shape=..., strides=..., offsets=...,
    #                   block_shape=..., order=...)
    L = OrderBy(Row(n_elements)).TileBy([n_elements / BLOCK_SIZE], [BLOCK_SIZE])

    x_ptrs = x_ptr + L[pid, :]
    y_ptrs = y_ptr + L[pid, :]

    x = tl.load(x_ptrs)
    y = tl.load(y_ptrs)
    z = x + y

    z_ptrs = z_ptr + L[pid, :]
    tl.store(z_ptrs, z)


# -- LEGO pointer-arithmetic kernel (for comparison) -----------------------
@lego_jit
@triton.jit
def vecadd_ptr_kernel(x_ptr, y_ptr, z_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    L = OrderBy(Row(n_elements)).TileBy([n_elements / BLOCK_SIZE], [BLOCK_SIZE])
    offsets = L[pid, :]
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)


# -- launch helpers ---------------------------------------------------------

def vecadd_block_ptr(x, y):
    n_elements = x.numel()
    z = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vecadd_block_ptr_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z


def vecadd_ptr(x, y):
    n_elements = x.numel()
    z = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vecadd_ptr_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    # Use a size divisible by BLOCK_SIZE for block_ptr (no boundary_check)
    size = 1024 * 64
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')

    z_block_ptr = vecadd_block_ptr(x, y)
    z_ptr = vecadd_ptr(x, y)
    z_torch = x + y

    print("=== Correctness ===")
    if torch.allclose(z_block_ptr, z_torch, atol=1e-5):
        print("  LEGO block_ptr vecadd: PASS")
    else:
        diff = (z_block_ptr - z_torch).abs().max().item()
        print(f"  LEGO block_ptr vecadd: FAIL (max diff={diff})")

    if torch.allclose(z_ptr, z_torch, atol=1e-5):
        print("  LEGO pointer   vecadd: PASS")
    else:
        diff = (z_ptr - z_torch).abs().max().item()
        print(f"  LEGO pointer   vecadd: FAIL (max diff={diff})")

    # -- benchmark ----------------------------------------------------------
    print("\n=== Benchmark ===")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[1024 * (2 ** i) for i in range(4, 16)],
            line_arg='provider',
            line_vals=['block_ptr', 'pointer', 'torch'],
            line_names=["LEGO block_ptr", "LEGO pointer", "Torch"],
            styles=[('red', '-'), ('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name="vecadd-block-ptr-performance",
            args={},
        ))
    def benchmark(N, provider):
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.randn(N, device='cuda', dtype=torch.float32)

        if provider == 'block_ptr':
            ms = triton.testing.do_bench(lambda: vecadd_block_ptr(x, y))
        elif provider == 'pointer':
            ms = triton.testing.do_bench(lambda: vecadd_ptr(x, y))
        elif provider == 'torch':
            ms = triton.testing.do_bench(lambda: x + y)

        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)
