import torch
import triton
import triton.language as tl
from triton.runtime import driver
import lego
from lego.lego import *
from lego import jit as lego_jit

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

@lego_jit
@triton.jit
def vecadd_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1D Tiled layout: Map (tile_idx, element_idx) to linear offset
    L = OrderBy(Row(n_elements)).TileBy([n_elements / BLOCK_SIZE, BLOCK_SIZE])
    
    pid = tl.program_id(0)
    range_n = tl.arange(0, BLOCK_SIZE)
    
    # Get offsets for this program's tile
    offsets = L[pid, range_n]
    
    # Masking for elements beyond n_elements
    mask = (pid * BLOCK_SIZE + range_n) < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

@triton.jit
def triton_vecadd_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)

def vecadd(x, y):
    n_elements = x.numel()
    z = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    vecadd_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z

def triton_vecadd(x, y):
    n_elements = x.numel()
    z = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    triton_vecadd_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return z

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 2**16
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    z_lego = vecadd(x, y)
    z_triton = triton_vecadd(x, y)
    z_torch = x + y
    
    if torch.allclose(z_lego, z_torch, atol=1e-5):
        print("✅ LEGO VecAdd match")
    else:
        print("❌ LEGO VecAdd mismatch")
        
    if torch.allclose(z_triton, z_torch, atol=1e-5):
        print("✅ Triton VecAdd match")
    else:
        print("❌ Triton VecAdd mismatch")

    # Benchmarking
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[2 ** i for i in range(12, 24)],
            line_arg='provider',
            line_vals=['lego', 'triton', 'torch'],
            line_names=["LEGO", "Triton", "Torch"],
            styles=[('red', '-'), ('blue', '-'), ('green', '-')],
            ylabel="GB/s",
            plot_name="vecadd-performance",
            args={},
        ))
    def benchmark(N, provider):
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.randn(N, device='cuda', dtype=torch.float32)
        
        if provider == 'lego':
            ms = triton.testing.do_bench(lambda: vecadd(x, y))
        elif provider == 'triton':
            ms = triton.testing.do_bench(lambda: triton_vecadd(x, y))
        elif provider == 'torch':
            ms = triton.testing.do_bench(lambda: x + y)
            
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)
