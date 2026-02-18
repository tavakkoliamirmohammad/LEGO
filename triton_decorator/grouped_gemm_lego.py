import torch
import triton
import triton.language as tl
import sympy as sp
import lego
from lego.lego import *
from lego import jit as lego_jit

@lego_jit
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
    gm_sym = Symbol('gm')
    gn_sym = Symbol('gn')
    gk_sym = Symbol('gk')
    
    L_A = OrderBy(Row(gm_sym, gk_sym)).TileBy([gm_sym/BLOCK_SIZE_M, gk_sym/BLOCK_SIZE_K], [BLOCK_SIZE_M, BLOCK_SIZE_K])
    L_B = OrderBy(Row(gk_sym, gn_sym)).TileBy([gk_sym/BLOCK_SIZE_K, gn_sym/BLOCK_SIZE_N], [BLOCK_SIZE_K, BLOCK_SIZE_N])
    L_C = OrderBy(Row(gm_sym, gn_sym)).TileBy([gm_sym/BLOCK_SIZE_M, gn_sym/BLOCK_SIZE_N], [BLOCK_SIZE_M, BLOCK_SIZE_N])

    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = tl.where(True, num_m_tiles * num_n_tiles, 0)
        
        start = tl.where(True, tl.maximum(tile_idx, last_problem_end), 0)
        stop = tl.where(True, last_problem_end + num_tiles, 0)
        
        for current_tile_idx in range(start, stop, NUM_SM):
            # Guard 'k' to ensure it's kept in source for the inner loop's 'range'
            k = tl.where(True, gk, 0)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            
            tile_idx_in_gemm = current_tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                offset_a = L_A[tile_m_idx, kk, 0:BLOCK_SIZE_M, 0:BLOCK_SIZE_K]
                offset_b = L_B[kk, tile_n_idx, 0:BLOCK_SIZE_K, 0:BLOCK_SIZE_N]
                a = tl.load(a_ptr + offset_a)
                b = tl.load(b_ptr + offset_b)
                accumulator += tl.dot(a, b)
            c = accumulator.to(tl.float16)

            offset_c = L_C[tile_m_idx, tile_n_idx, 0:BLOCK_SIZE_M, 0:BLOCK_SIZE_N]
            tl.store(c_ptr + offset_c, c)

        num_iters = tl.cdiv(stop - start, NUM_SM)
        tile_idx = start + num_iters * NUM_SM
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
    NUM_SM = 84 
    grid = (NUM_SM, )
    lego_grouped_matmul_kernel[grid](
        d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
        NUM_SM=NUM_SM, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )
    return group_C

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
