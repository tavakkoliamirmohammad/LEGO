import torch
import triton
import triton.language as tl
from triton.runtime import driver
import lego
from lego.core import *
from lego.frontends.triton_jit import jit as lego_jit

def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret

@lego_jit
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)

    L_in = OrderBy(Row(n_rows, n_cols)).TileBy([n_rows, n_cols])
    L_out = OrderBy(Row(n_rows, n_cols)).TileBy([n_rows, n_cols])

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_offset = L_in[row_idx, col_offsets]
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
def triton_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
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

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows, )](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    triton_softmax_kernel[(n_rows, )](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device='cuda')
    y_lego = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    if torch.allclose(y_lego, y_torch, atol=1e-2, rtol=0):
        print("✅ Softmax match")
    else:
        print("❌ Softmax mismatch")

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
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
        elif provider == 'lego':
            ms = triton.testing.do_bench(lambda: softmax(x))
        elif provider == 'triton':
            ms = triton.testing.do_bench(lambda: triton_softmax(x))
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

    benchmark.run(show_plots=False, print_data=True)
