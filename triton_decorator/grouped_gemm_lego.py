import torch
import triton
import triton.language as tl
import lego
from lego.lego import *
from lego import jit as lego_jit
from sympy import Max, Min

@lego_jit
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def lego_grouped_matmul_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        
        # Layouts defined locally using loaded dimensions
        L_A = OrderBy(Row(gm, gk)).TileBy([gm/BLOCK_SIZE_M, gk/BLOCK_SIZE_K], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        L_B = OrderBy(Row(gk, gn)).TileBy([gk/BLOCK_SIZE_K, gn/BLOCK_SIZE_N], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        L_C = OrderBy(Row(gm, gn)).TileBy([gm/BLOCK_SIZE_M, gn/BLOCK_SIZE_N], [BLOCK_SIZE_M, BLOCK_SIZE_N])

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                a_ptrs = a_ptr + L_A[tile_m_idx, kk, :, :]
                b_ptrs = b_ptr + L_B[kk, tile_n_idx, :, :]
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
            
            c_ptrs = c_ptr + L_C[tile_m_idx, tile_n_idx, :, :]
            tl.store(c_ptrs, accumulator.to(tl.float16))

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles

def lego_group_gemm_fn(group_A, group_B):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B)
    group_size = len(group_A)
    A_addrs, B_addrs, C_addrs, g_sizes, g_lds, group_C = [], [], [], [], [], []
    for i in range(group_size):
        A, B = group_A[i], group_B[i]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr()); B_addrs.append(B.data_ptr()); C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]; g_lds += [A.stride(0), B.stride(0), C.stride(0)]
    d_a_ptrs = torch.tensor(A_addrs, device=device); d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device); d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    
    # Grid and launch using autotuned parameters
    grid = lambda META: (META['NUM_SM'], )
    lego_grouped_matmul_kernel[grid](
        d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
    )
    return group_C

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(7, 13)],
        line_arg='provider',
        line_vals=['torch', 'lego'],
        line_names=["Torch", "LEGO"],
        styles=[('blue', '-'), ('red', '-')],
        ylabel="runtime(ms)",
        plot_name="group-gemm-performance",
        args={},
    ))
def benchmark(N, provider):
    group_size = 4
    group_A, group_B = [], []
    for i in range(group_size):
        group_A.append(torch.rand((N, N), device="cuda", dtype=torch.float16))
        group_B.append(torch.rand((N, N), device="cuda", dtype=torch.float16))

    if provider == 'lego':
        ms = triton.testing.do_bench(lambda: lego_group_gemm_fn(group_A, group_B))
    else:
        ms = triton.testing.do_bench(lambda: [torch.matmul(a, b) for a, b in zip(group_A, group_B)])
    return ms

if __name__ == "__main__":
    group_m, group_n, group_k = [1024, 512, 256, 128], [1024, 512, 256, 128], [1024, 512, 256, 128]
    group_A, group_B = [], []
    for i in range(len(group_m)):
        group_A.append(torch.rand((group_m[i], group_k[i]), device="cuda", dtype=torch.float16))
        group_B.append(torch.rand((group_k[i], group_n[i]), device="cuda", dtype=torch.float16))
    tri_out = lego_group_gemm_fn(group_A, group_B)
    ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
    for i in range(len(group_m)):
        assert torch.allclose(ref_out[i], tri_out[i], atol=2e-2, rtol=1e-2)
    print("✅ Grouped GEMM match")
    benchmark.run(show_plots=False, print_data=True)
