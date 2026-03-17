import torch
import triton
import triton.language as tl
from triton.runtime import driver
import lego
from lego.core import *
from lego.frontends.triton_jit import jit as lego_jit

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')

def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret

@lego_jit
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # Persistent program logic
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # Define LEGO layouts for input and output:
    # 1. Row(n_rows, n_cols): represents a 2D row-major layout.
    # 2. OrderBy(...): specifies that the elements should be ordered according to the Row layout.
    # 3. TileBy([n_rows, n_cols]): tiles the entire matrix into a single large tile 
    #    (or more precisely, it just treats the whole matrix as one tile here because 
    #    the tiling matches the dimensions).
    L_in = OrderBy(Row(n_rows, n_cols)).TileBy([n_rows, n_cols])
    L_out = OrderBy(Row(n_rows, n_cols)).TileBy([n_rows, n_cols])

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_offset = L_in[row_idx, col_offsets] # Access elements at row_idx and col_offsets
        input_ptrs = input_ptr + input_offset

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_offset = L_out[row_idx, col_offsets]
        output_ptrs = output_ptr + output_offset
        tl.store(output_ptrs, softmax_output, mask=mask)

@triton.jit
def triton_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
kernels = {}
triton_kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)

    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            NUM_GPRS = NUM_REGS * 2 if is_cdna() else NUM_REGS
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(int(num_programs), n_rows)
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    return y

def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel, num_programs = triton_kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = triton_softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            NUM_GPRS = NUM_REGS * 2 if is_cdna() else NUM_REGS
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        triton_kernels[BLOCK_SIZE] = (kernel, num_programs)
    num_programs = min(int(num_programs), n_rows)
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    if torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0):
        print("✅ Softmax match")
    else:
        print("❌ Softmax mismatch")

    # Benchmarking
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[2 ** i for i in range(9, 14)],
            line_arg='provider',
            line_vals=['triton', 'torch', 'lego'],
            line_names=["Triton", "Torch", "LEGO"],
            styles=[('blue', '-'), ('green', '-'), ('red', '-')],
            ylabel="GB/s",
            plot_name="softmax-performance",
            args={'M': 4096},
        ))
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        if provider == 'lego':
            torch_out = torch.softmax(x, axis=-1)
            triton_out = softmax(x)
            if not torch.allclose(torch_out, triton_out, atol=1e-2):
                raise ValueError(f"LEGO Softmax mismatch! N={N}")
        
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        elif provider == 'lego':
            ms = triton.testing.do_bench(lambda: softmax(x))
        elif provider == 'triton':
            ms = triton.testing.do_bench(lambda: triton_softmax(x))
        
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)
