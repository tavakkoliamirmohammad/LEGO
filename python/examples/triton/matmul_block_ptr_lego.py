"""Matrix multiplication using LEGO block_ptr mode.

Generates tl.make_block_ptr() + tl.advance() instead of pointer arithmetic,
enabling TMA acceleration on Hopper+ GPUs. The kernel code is identical to
the pointer-arithmetic version -- only the decorator flag changes.
"""

import torch
import triton
import triton.language as tl
from lego.core import *
from lego.frontends.triton_jit import jit as lego_jit
from sympy import Max, Min


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'BM': 128, 'BN': 256, 'BK': 16, 'GM': 1, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GM': 1, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    return get_cuda_autotune_config() if is_cuda() else get_hip_autotune_config()


# -- LEGO block_ptr matmul -------------------------------------------------
# use_block_ptr=True:
#   - `a_ptrs = a_ptr + L_A[pid_m, k, :, :]` generates tl.make_block_ptr(...)
#   - Loop-dependent block_ptrs are auto-hoisted before the loop
#   - tl.advance() is inserted at the end of the loop body

@lego_jit(use_block_ptr=True)
@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_block_ptr_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    # PID layout for L2 cache optimization (group-major ordering).
    # This is NOT converted to block_ptr (multi-perm OrderBy), which is correct:
    # PID remapping is index arithmetic, not a memory descriptor.
    L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                    Col(Min(num_pid_m, GM), num_pid_n)).TileBy([num_pid_m, num_pid_n])
    pid_m, pid_n = L_pid.inv(pid)

    # Data layouts -- identical to the pointer-arithmetic matmul_lego.py.
    # With use_block_ptr=True, the rewriter generates:
    #   a_ptrs = tl.make_block_ptr(base=a_ptr, shape=(M,K), strides=(K,1),
    #                              offsets=(pid_m*BM, 0), block_shape=(BM,BK), order=(1,0))
    # and hoists it before the loop, inserting tl.advance(a_ptrs, (0, BK)) at the end.
    L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
    L_B = OrderBy(Row(K, N)).TileBy([K / BK, N / BN], [BK, BN])
    L_C = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a_ptrs = a_ptr + L_A[pid_m, k, :, :]
        b_ptrs = b_ptr + L_B[k, pid_n, :, :]
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + L_C[pid_m, pid_n, :, :]
    tl.store(c_ptrs, c)


# -- LEGO pointer-arithmetic matmul (for comparison) -----------------------

@lego_jit
@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_ptr_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GM: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)

    L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                    Col(Min(num_pid_m, GM), num_pid_n)).TileBy([num_pid_m, num_pid_n])
    pid_m, pid_n = L_pid.inv(pid)

    L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
    L_B = OrderBy(Row(K, N)).TileBy([K / BK, N / BN], [BK, BN])
    L_C = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a_ptrs = a_ptr + L_A[pid_m, k, :, :]
        b_ptrs = b_ptr + L_B[k, pid_n, :, :]
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + L_C[pid_m, pid_n, :, :]
    tl.store(c_ptrs, c)


# -- launch helpers ---------------------------------------------------------

def matmul_block_ptr(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
    matmul_block_ptr_kernel[grid](a, b, c, M, N, K)
    return c


def matmul_ptr(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
    matmul_ptr_kernel[grid](a, b, c, M, N, K)
    return c


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)

    c_block_ptr = matmul_block_ptr(a, b)
    c_ptr = matmul_ptr(a, b)
    c_torch = torch.matmul(a, b)

    print("=== Correctness ===")
    if torch.allclose(c_block_ptr, c_torch, atol=1e-2, rtol=0):
        print("  LEGO block_ptr matmul: PASS")
    else:
        diff = (c_block_ptr - c_torch).abs().max().item()
        print(f"  LEGO block_ptr matmul: FAIL (max diff={diff})")

    if torch.allclose(c_ptr, c_torch, atol=1e-2, rtol=0):
        print("  LEGO pointer   matmul: PASS")
    else:
        diff = (c_ptr - c_torch).abs().max().item()
        print(f"  LEGO pointer   matmul: FAIL (max diff={diff})")

    # -- benchmark ----------------------------------------------------------
    print("\n=== Benchmark ===")
    ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2 ** i for i in range(7, 13)],
            line_arg="provider",
            line_vals=[ref_lib.lower(), "block_ptr", "pointer"],
            line_names=[ref_lib, "LEGO block_ptr", "LEGO pointer"],
            styles=[("green", "-"), ("red", "-"), ("blue", "-")],
            ylabel="TFLOPS",
            plot_name="matmul-block-ptr-performance-fp16",
            args={},
        )
    ])
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b), quantiles=quantiles)
        elif provider == 'block_ptr':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul_block_ptr(a, b), quantiles=quantiles)
        elif provider == 'pointer':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul_ptr(a, b), quantiles=quantiles)

        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=False, print_data=True)
