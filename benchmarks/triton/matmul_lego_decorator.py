import torch
import triton
import triton.language as tl
import sympy as sp
from sympy import Symbol
import lego
from lego.lego import *
from lego import jit as lego_jit

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_cuda_autotune_config():
    return [
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return []


# Decorator order matters! Bottom-up application means:
# 1. @lego.jit applies FIRST to raw function (transforms LEGO -> Triton code)
# 2. @triton.jit applies SECOND to the transformed function  
@lego_jit
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GM: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    """Matrix multiplication kernel using LEGO layout algebra."""
    
    # Get thread block ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    
    # -----------------------------------------------------------
    # LEGO Layout Definitions - These are evaluated at decoration time!
    # -----------------------------------------------------------
    
    # Create symbolic variables
    s_pid = Symbol('pid')
    s_num_pid_m = Symbol('num_pid_m')
    s_num_pid_n = Symbol('num_pid_n')
    s_GM = Symbol('GM')
    s_M = Symbol('M')
    s_N = Symbol('N')
    s_K = Symbol('K')
    s_BM = Symbol('BM')
    s_BN = Symbol('BN')
    s_BK = Symbol('BK')
    s_k = Symbol('k')
    
    # PID Layout - maps thread blocks to output tiles
    L_pid = OrderBy(
        Col(sp.Max(s_num_pid_m//s_GM, 1), 1),
        Col(sp.Min(s_num_pid_m, s_GM), s_num_pid_n)
    ).TileBy([s_num_pid_m, s_num_pid_n])
    
    pid_m, pid_n = L_pid.inv(s_pid)
    
    # Matrix A Layout - Row-major [M, K] tiled by [BM, BK]
    L_A = OrderBy(Row(s_M, s_K)).TileBy([s_M/s_BM, s_K/s_BK], [s_BM, s_BK])
    offset_a = L_A[pid_m, s_k, :, :]
    
    # Matrix B Layout - Row-major [K, N] tiled by [BK, BN]
    L_B = OrderBy(Row(s_K, s_N)).TileBy([s_K/s_BK, s_N/s_BN], [s_BK, s_BN])
    offset_b = L_B[s_k, pid_n, :, :]
    
    # Matrix C Layout - Row-major [M, N] tiled by [BM, BN]
    L_C = OrderBy(Row(s_M, s_N)).TileBy([s_M/s_BM, s_N/s_BN], [s_BM, s_BN])
    offset_c = L_C[pid_m, pid_n, :, :]
    
    # -----------------------------------------------------------
    # Runtime Kernel Body - After LEGO transformation, the above symbols
    # will be replaced with the generated runtime expressions
    # -----------------------------------------------------------
    
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BK)):
        # Load A and B using LEGO-generated offsets
        a_ptrs = a_ptr + offset_a
        b_ptrs = b_ptr + offset_b
        
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        accumulator = tl.dot(a, b, accumulator)
    
    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)
    
    c = accumulator.to(tl.float16)
    
    # Store C using LEGO-generated offset
    c_ptrs = c_ptr + offset_c
    tl.store(c_ptrs, c)


def matmul(a, b, activation=""):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Use simple fixed configuration since we don't have autotune  
    BM, BN, BK, GM = 64, 64, 32, 4
    
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), )
    
    matmul_kernel[grid](a, b, c, M, N, K, BM, BN, BK, GM, ACTIVATION=activation)
    return c


if __name__ == "__main__":
    print("Running LEGO Triton Matmul with Decorator Syntax...")
    
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        print(f"Max diff: {torch.max(torch.abs(triton_output - torch_output))}")
