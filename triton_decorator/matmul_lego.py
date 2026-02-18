import torch
import triton
import triton.language as tl
import sympy as sp
import lego
from lego.lego import *
from lego import jit as lego_jit

# Symbols for layouts
M_sym = sp.Symbol('M', integer=True, positive=True)
N_sym = sp.Symbol('N', integer=True, positive=True)
K_sym = sp.Symbol('K', integer=True, positive=True)
BM_sym = sp.Symbol('BM', integer=True, positive=True)
BN_sym = sp.Symbol('BN', integer=True, positive=True)
BK_sym = sp.Symbol('BK', integer=True, positive=True)
GM_sym = sp.Symbol('GM', integer=True, positive=True)
pid_sym = sp.Symbol('pid', integer=True, positive=True)
num_pid_m_sym = sp.Symbol('num_pid_m', integer=True, positive=True)
num_pid_n_sym = sp.Symbol('num_pid_n', integer=True, positive=True)

# Layout for grouped L2 optimization
L_group = OrderBy(Col(sp.Max(num_pid_m_sym // GM_sym, 1), 1),
                  Col(sp.Min(num_pid_m_sym, GM_sym), num_pid_n_sym)).TileBy([num_pid_m_sym, num_pid_n_sym])

# Layouts for A, B, C (assuming Row-major for simplicity, can be extended for Transpose)
L_A = OrderBy(Row(M_sym, K_sym)).TileBy([M_sym/BM_sym, K_sym/BK_sym], [BM_sym, BK_sym])
L_B = OrderBy(Row(K_sym, N_sym)).TileBy([K_sym/BK_sym, N_sym/BN_sym], [BK_sym, BN_sym])
L_C = OrderBy(Row(M_sym, N_sym)).TileBy([M_sym/BM_sym, N_sym/BN_sym], [BM_sym, BN_sym])

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@lego_jit
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        # Meta-parameters
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GM: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    
    # pid_m, pid_n = {{ pid_m }}, {{ pid_n }}
    pid_m, pid_n = L_group.inv(pid)

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # a_ptrs = a_ptr + {{ offset_aptrs }}
        a_ptrs = a_ptr + L_A[pid_m, k, :, :]
        # b_ptrs = b_ptr + {{ offset_bptrs }}
        b_ptrs = b_ptr + L_B[k, pid_n, :, :]
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
    
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    
    c = accumulator.to(tl.float16)
    
    # c_ptrs = c_ptr + {{offset_cptrs}}
    c_ptrs = c_ptr + L_C[pid_m, pid_n, :, :]
    
    tl.store(c_ptrs, c)

def matmul(a, b, activation=""):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # Simple grid lambda as in benchmark
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']), )
    
    # Launch with some default/tunable meta-parameters
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BM=128, BN=256, BK=64, GM=8,
        ACTIVATION=activation,
        num_warps=8, num_stages=3
    )
    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    print("✅ Matmul match")
