import torch
import triton
import triton.language as tl
import sympy as sp
import lego
from lego.lego import *
from lego import jit as lego_jit

# Symbols matching kernel args
n_rows = sp.Symbol('n_rows', integer=True, positive=True)
n_cols = sp.Symbol('n_cols', integer=True, positive=True)
BLOCK_SIZE = sp.Symbol('BLOCK_SIZE', integer=True, positive=True)

# Layouts from benchmarks/triton/softmax_sympy.py
# 2D layout tiled by [n_rows, n_cols]
L_in = OrderBy(Row(n_rows, n_cols)).TileBy([n_rows, n_cols])
L_out = OrderBy(Row(n_rows, n_cols)).TileBy([n_rows, n_cols])

@lego_jit
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Result of L_in[row_idx, 0:BLOCK_SIZE] is [1, BLOCK_SIZE]
        input_offset = L_in[row_idx, 0:BLOCK_SIZE]
        input_ptrs = input_ptr + input_offset

        # Mask must also be 2D shape (1, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE)[None, :] < n_cols

        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # axis=1 for 2D tensor
        row_minus_max = row - tl.max(row, axis=1)
        # Note that exponentiation in Triton is fast but approximate
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=1)
        softmax_output = numerator / denominator

        output_offset = L_out[row_idx, 0:BLOCK_SIZE]
        output_ptrs = output_ptr + output_offset
        
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 2
    y = torch.empty_like(x)
    num_programs = min(128, n_rows) 
    softmax_kernel[(num_programs, )](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y

def test_softmax(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, device='cuda')
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-2, rtol=0)
    print(f"✅ Softmax match for M={M}, N={N}")

if __name__ == "__main__":
    test_softmax(1823, 781)
