"""cuBLAS through torch.matmul for the GEMM sweep shapes, runnable in any torch environment (used to
check a second CUDA/cuBLAS version on the same GPU; same timing method as the sweeps)."""
import argparse, os, sys, time, torch


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


def time_fn(fn, iters=20, warmup=5, batches=5, min_warmup_ms=30.0, min_batch_ms=20.0, cooldown_s=1.0):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096")
    ap.add_argument("--shapes", default="4096x6144x4096;4096x14336x4096;4096x4096x14336;4096x11008x4096;4096x4096x11008;4096x18944x3584;4096x3584x18944")
    ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    p = torch.cuda.get_device_properties(0)
    lines = [f"<!-- provenance -->\nDevice: {p.name} (cc {p.major}.{p.minor}, {p.multi_processor_count} SMs)\ntimestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')} | torch: {torch.__version__} | cuda: {torch.version.cuda} | cublas (torch build): {torch.backends.cuda.matmul.__class__.__name__}", ""]
    probs = [(n, n, n) for n in [int(x) for x in a.sizes.split(",") if x]] + [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(";") if s]
    for M, N, K in probs:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16); B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ms = time_fn(lambda: torch.matmul(A, B)); tf = 2.0 * M * N * K / ms / 1e9
        line = f"{M}x{N}x{K}: cuBLAS {ms:.3f} ms ({tf:.1f} TF)"; print(line, flush=True); lines.append(line)
    tag = f"{p.name.lower().replace(' ', '_')}_sm{p.major}{p.minor}"
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "results", tag, f"cublas_cuda{torch.version.cuda.replace('.', '')}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "w").write("\n".join(lines) + "\n"); print("wrote", out)


if __name__ == "__main__":
    main()
