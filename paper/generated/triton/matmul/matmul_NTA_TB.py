
import torch
import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def get_cuda_autotune_config():
    return [
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GM': 8}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=5, num_warps=2),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 128, 'GM': 8}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=4, num_warps=4),
    ]

def get_hip_autotune_config():
    return [
        triton.Config({'BM': 128, 'BN': 256, 'BK': 16, 'GM': 1, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 16, 'GM': 4, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GM': 1, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32, 'GM': 1, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr, ACTIVATION: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid % min(GM, num_pid_m) + pid // (num_pid_n * min(GM, num_pid_m)) % max(1, num_pid_m // GM) * min(GM, num_pid_m)
    pid_n = pid % (num_pid_n * min(GM, num_pid_m)) // min(GM, num_pid_m)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a_ptrs = a_ptr + (BK * k + K * (BM * pid_m + tl.arange(0, BM)[:, None]) + tl.arange(0, BK)[None, :])
        b_ptrs = b_ptr + (BK * k + K * (BN * pid_n + tl.arange(0, BN)[None, :]) + tl.arange(0, BK)[:, None])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
    if ACTIVATION == 'leaky_relu':
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (BN * pid_n + N * (BM * pid_m + tl.arange(0, BM)[:, None]) + tl.arange(0, BN)[None, :])
    tl.store(c_ptrs, c)


def matmul(a, b, activation=""):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']), )
    matmul_kernel[grid](a, b, c, M, N, K, ACTIVATION=activation)
    return c


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GM: tl.constexpr,
        ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    GM_effective = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GM_effective)
    pid_n = (pid % num_pid_in_group) // GM_effective

    offs_am = (pid_m * BM + tl.arange(0, BM))
    offs_bn = (pid_n * BN + tl.arange(0, BN))
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def triton_matmul(a, b, activation=""):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']), )
    triton_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16).T
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    pass
else:
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16).T
    a = a.to(torch.float8_e5m2)
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))

    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        pass
    else:
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        print("❌ Triton and Torch differ")

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],
            x_vals=[2 ** i for i in range(7, 14)],
            line_arg="provider",
            line_vals=["triton", "lego"] if fp8_inputs else [
                ref_lib.lower(), "triton", "lego"],  
            line_names=["Triton", "LEGO"] if fp8_inputs else [
                ref_lib, "Triton", "LEGO"], 
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",  
            plot_name="matmul-performance-" + ("fp16" if not fp8_inputs else "fp8"),
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16).T
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.to(torch.float8_e5m2)

    if provider == 'lego' and not (TORCH_HAS_FP8 and fp8_inputs):
        torch_out = torch.matmul(a, b)
        triton_out = matmul(a, b)
        if not torch.allclose(torch_out, triton_out, atol=1e-1, rtol=1e-1):
            raise ValueError(f"Triton matmul outputs do not match Torch matmul! (M = {M}, N = {N}, K = {K})")

    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == 'lego':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_matmul(a, b), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=False, print_data=True)

