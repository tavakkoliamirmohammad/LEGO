"""cuBLAS through torch.matmul for the GEMM sweep shapes, runnable in any torch environment (used to
check a second CUDA/cuBLAS version on the same GPU; same timing method as the sweeps)."""
import argparse, os, sys, time, torch


def time_fn(fn, iters=20, warmup=5, batches=3, min_warmup_ms=150.0, min_batch_ms=100.0):
    """Same method as bench/triton_matmul.time_fn (inline: this file runs without the LEGO package)."""
    def timed(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    est = timed(max(1, iters)) / max(1, iters)
    timed(max(1, int(min_warmup_ms / max(est, 1e-3))))
    n = max(iters, int(min_batch_ms / max(est, 1e-3)))
    per = sorted(timed(n) / n for _ in range(batches))
    return per[len(per) // 2]


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
