"""Standard Triton matmul (tutorial structure) as a baseline; fixed configs or autotune."""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs); b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)


def triton_matmul(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=3):
    M, K = A.shape; _, N = B.shape
    grid = (M // BLOCK_M, N // BLOCK_N)
    _matmul_kernel[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages)
    return C


def time_fn(fn, iters=50, warmup=10, batches=3):
    """Median over batches of `iters` back-to-back calls (same method as CastleKernel.time)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(batches):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        per.append(s.elapsed_time(e) / iters)
    per.sort()
    return per[len(per) // 2]


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_stages=st, num_warps=nw)
    for bm, bn, bk, st, nw in [(128, 256, 32, 3, 8), (256, 128, 32, 3, 8), (128, 128, 32, 4, 4), (128, 128, 64, 3, 4),
                               (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (128, 64, 64, 3, 4), (64, 64, 32, 4, 4),
                               (64, 64, 64, 3, 4), (128, 128, 32, 3, 8)]
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _matmul_kernel_tuned(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs); b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak; b_ptrs += BLOCK_K * stride_bk
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)


def triton_matmul_tuned(A, B, C):
    M, K = A.shape; _, N = B.shape
    grid = lambda meta: (M // meta["BLOCK_M"], N // meta["BLOCK_N"])
    _matmul_kernel_tuned[grid](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))
    return C
