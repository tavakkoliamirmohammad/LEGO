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


def _sm_clock_mhz():
    """Current SM clock via NVML (None if NVML is unavailable); recorded next to every timing."""
    try:
        import pynvml
        if not hasattr(_sm_clock_mhz, "h"):
            pynvml.nvmlInit(); _sm_clock_mhz.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return pynvml.nvmlDeviceGetClockInfo(_sm_clock_mhz.h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None


LAST = {"clock_mhz": None, "batches_ms": None}


def time_fn(fn, iters=20, warmup=5, batches=5, min_warmup_ms=50.0, min_batch_ms=20.0, cooldown_s=0.3):
    """Best of `batches` short batches of back-to-back calls after an idle gap; the method every
    kernel in the suite is timed with, on every system.

    Power-capped GPUs change clock state within a fraction of a second of sustained tensor-core
    load, and the state depends on what ran just before, so long batches compare kernels at
    different clocks. Instead: idle for `cooldown_s` so every measurement starts from the same
    state, warm up for at least `warmup` calls and `min_warmup_ms`, then run `batches` batches of
    at least `iters` calls and `min_batch_ms` each (a burst well under the power controller's
    reaction time). The minimum batch mean is reported; LAST records the SM clock after each batch.
    """
    import time as _time
    def timed(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)
    torch.cuda.synchronize(); _time.sleep(cooldown_s)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    est = timed(max(1, iters)) / max(1, iters)
    timed(max(1, int(min_warmup_ms / max(est, 1e-3))))
    n = max(iters, int(min_batch_ms / max(est, 1e-3)))
    per, clocks = [], []
    for _ in range(batches):
        per.append(timed(n) / n); clocks.append(_sm_clock_mhz())
    LAST["clock_mhz"], LAST["batches_ms"] = clocks, per
    return min(per)


def clock_str():
    c = [x for x in (LAST["clock_mhz"] or []) if x]
    return f"{min(c)}-{max(c)}" if c else "n/a"


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
