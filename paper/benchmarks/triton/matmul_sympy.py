import sympy as sp
import jinja2
from lego.lego_python import *

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set transpose flags')
    parser.add_argument('--TA', action='store_true',
                        help='Set transpose_a to True')
    parser.add_argument('--TB', action='store_true',
                        help='Set transpose_b to True')
    parser.add_argument('--NTA', action='store_true',
                        help='Set transpose_a to False')
    parser.add_argument('--NTB', action='store_true',
                        help='Set transpose_b to False')
    return parser.parse_args()


transpose_a = False
transpose_b = False

args = parse_arguments()

# Update values based on arguments
if args.TA:
    transpose_a = True
if args.TB:
    transpose_b = True
if args.NTA:
    transpose_a = False
if args.NTB:
    transpose_b = False


def layout_A(M, K, BM, BK, k):
    a_layout = OrderBy(Col(M, K) if transpose_a else Row(M, K)).TileBy(
        [M/BM, K/BK], [BM, BK])
    return a_layout['pid_m', k, :, :]


def layout_B(N, K, BN, BK, k):
    b_layout = OrderBy(
        Col(K, N) if transpose_b else Row(K, N)).TileBy([K/BK, N/BN], [BK, BN])
    return b_layout[k, 'pid_n', :, :]


def layout_C(M, N, BM, BN):
    # row_layout = Layout(1).ordered_by(Row(M)).grouped_by([M/BM], [BM])
    # col_layout = Layout(1).ordered_by(Row(N)).grouped_by([N/BN], [BN])
    c_layout = OrderBy(
        Row(M, N)).TileBy([M/BM, N/BN], [BM, BN])
    return c_layout['pid_m', 'pid_n', :, :]


def layout_group_pid(num_tile_m, num_tile_n, GM, pid):
    # L = OrderBy(Col(num_tile_m//GM, 1),
    #             Col(GM, num_tile_n)).TileBy([num_tile_m, num_tile_n])
    L = OrderBy(Col(sp.Max(num_tile_m//GM, 1), 1),
                Col(sp.Min(num_tile_m, GM), num_tile_n)).TileBy([num_tile_m, num_tile_n])

    # L = OrderBy(Row(num_tile_m, num_tile_n)).TileBy([num_tile_m, num_tile_n])
    # return L.inv(pid, [sp.Gt(GM, 0, evaluate=False), sp.Eq(num_pid_m % GM, 0, evaluate=False)])
    return L.inv(pid)



M, N, K, BM, BN, BK, num_pid_m, num_pid_n, GM, pid, k = sp.symbols(
    'M N K BM BN BK num_pid_m num_pid_n GM pid k', integer=True, positive=True)


pid_m, pid_n = layout_group_pid(num_pid_m, num_pid_n, GM, pid)
# print(pid_m)
# print(pid_n)
offset_aptrs = layout_A(M, K, BM, BK, k)
offset_bptrs = layout_B(N, K, BN, BK, k)
offset_cptrs = layout_C(M, N, BM, BN)

# Define a Jinja template for the Triton kernel.
kernel_template = jinja2.Template(
    """
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
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64, 'GM': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 32, 'GM': 8}, num_stages=4,
                      num_warps=4),
        # triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GM': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BM': 128, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=4,
        #               num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GM': 8}, num_stages=4,
                      num_warps=4),
        # triton.Config({'BM': 128, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=4,
        #               num_warps=4),
        triton.Config({'BM': 64, 'BN': 32, 'BK': 32, 'GM': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 32, 'GM': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        # triton.Config({'BM': 128, 'BN': 256, 'BK': 128, 'GM': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BM': 256, 'BN': 128, 'BK': 128, 'GM': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BM': 256, 'BN': 64, 'BK': 128, 'GM': 8}, num_stages=4,
        #               num_warps=4),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 128, 'GM': 8}, num_stages=4,
                      num_warps=4),
        # triton.Config({'BM': 128, 'BN': 128, 'BK': 128, 'GM': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BM': 128, 'BN': 64, 'BK': 64, 'GM': 8}, num_stages=4,
        #               num_warps=4),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64, 'GM': 8}, num_stages=4,
                      num_warps=4),
        # triton.Config({'BM': 128, 'BN': 32, 'BK': 64, 'GM': 8}, num_stages=4,
        #               num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BM': 128, 'BN': 256,
                'BK': 16, 'GM': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BM': 256, 'BN': 256,
                'BK': 16, 'GM': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BM': 128, 'BN': 128,
                'BK': 32, 'GM': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BM': 64, 'BN': 128, 'BK': 32,
                'GM': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BM': 64, 'BN': 64, 'BK': 32,
                'GM': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BM`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        # Meta-parameters
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  #
        GM: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = {{ pid_m }}
    pid_n = {{ pid_n }}

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a_ptrs = a_ptr + {{ offset_aptrs }}
        b_ptrs = b_ptr + {{ offset_bptrs }}
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + {{offset_cptrs}}
    tl.store(c_ptrs, c)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(
        M, META['BM']) * triton.cdiv(N, META['BN']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        ACTIVATION=activation  #
    )
    return c


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  #
        GM: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GM * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GM
    GM = min(num_pid_m - first_pid_m, GM)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % GM)
    pid_n = (pid % num_pid_in_group) // GM

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BM, BK] pointers
    # `b_ptrs` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BM + tl.arange(0, BM))
    offs_bn = (pid_n * BN + tl.arange(0, BN))
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BM, BN]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BN + tl.arange(0, BN)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c)



def triton_matmul(a, b, activation=""):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']), )
    triton_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16){{transpose_a}}
b = torch.randn((512, 512), device='cuda', dtype=torch.float16){{transpose_b}}
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    # print("\u2705 Triton and Torch match")
    pass
else:
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    print("\u274c Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16){{transpose_a}}
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16){{transpose_b}}
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    # b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))

    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        # print("\u2705 Triton and Torch match")
        pass
    else:
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        print("\u274c Triton and Torch differ")

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            # Argument names to use as an x-axis for the plot
            x_names=["M", "N", "K"],
            # Different possible values for `x_name`
            x_vals=[2 ** i for i in range(7, 14)],
            # Argument name whose value corresponds to a different line in the plot
            line_arg="provider",
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton", "lego"] if fp8_inputs else [
                ref_lib.lower(), "triton", "lego"],  # Label name for the lines
            line_names=["Triton", "LEGO"] if fp8_inputs else [
                ref_lib, "Triton", "LEGO"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            # Name for the plot, used also as a file name for saving the plot.
            ("fp16" if not fp8_inputs else "fp8"),
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16){{transpose_a}}
    b = torch.randn((K, N), device='cuda', dtype=torch.float16){{transpose_b}}
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        # b = b.T
        b = b.to(torch.float8_e5m2)

    # If using Triton, check correctness against torch.matmul.
    if provider == 'lego' and not (TORCH_HAS_FP8 and fp8_inputs):
        torch_out = torch.matmul(a, b)
        triton_out = matmul(a, b)
        if not torch.allclose(torch_out, triton_out, atol=1e-1, rtol=1e-1):
            print(torch_out)
            print(triton_out)
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
"""
)


# Render the kernel code, injecting the offset calculation code generated by Sympy.
params = {
    'pid_m': pid_m,
    'pid_n': pid_n,
    'offset_aptrs': offset_aptrs,
    'offset_bptrs': offset_bptrs,
    'offset_cptrs': offset_cptrs,
}

printer = LEGOPythonCodePrinter(allow_unknown_functions=True)
render_params = {key: printer.doprint(sp.simplify(value))
                 for key, value in params.items()}

render_params["transpose_a"] = ".T" if transpose_a else ""
render_params["transpose_b"] = ".T" if transpose_b else ""
kernel_code = kernel_template.render(**render_params)
print(kernel_code)
