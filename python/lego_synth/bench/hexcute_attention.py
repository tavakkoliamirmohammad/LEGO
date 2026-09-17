"""Hexcute attention baseline (the artifact's `benchmark_attention.py` FlashAttention-2 kernel, tuned over
its own tiled-MMA / pipeline space).  Runs inside the Hexcute environment created by
`bench/hexcute_setup.sh`; it never imports the LEGO package, so it stays independent of our build.

Same logical problem as attn_sweep.py (B=1, H=BH heads, N tokens, head dim D, f16, non-causal — the
artifact kernel has no causal mode) in Hexcute's native [B, S, H, D] layout; same timing method
(median of 3 batches of 20 back-to-back launches); correctness against PyTorch's reference.
"""
import argparse
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

HEXCUTE_ROOT = os.environ.get("HEXCUTE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "hexcute-bench", "workspace", "hidet"))
sys.path.insert(0, os.path.join(HEXCUTE_ROOT, "examples", "cute"))
import hidet  # noqa: E402
import benchmark_attention as ba  # noqa: E402  (registers the tiled-MMA pairs for head sizes 32..256 step 32)


def register_extra_head_sizes(head_sizes):
    """The artifact registers tiled-MMA pairs for head sizes 32, 64, 96, 128, ...; add the same
    families for the head sizes it lacks (80, 112) so Hexcute gets the same chance we give ourselves."""
    from hidet.ir.cute.layout import TensorLayout, Level
    from hidet.ir.cute.algorithm import MmaAtom, TiledMma
    for head_size in head_sizes:
        for warp in [4, 8]:
            for repeat_m in [1, 2]:
                for repeat_n in [2, 4, 8]:
                    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
                    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
                    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
                    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
                    lvl = Level("warp", "thread_block", (warp, 1), TensorLayout((warp, 1)), (repeat_m, repeat_n))
                    tiled_mma_qk = TiledMma(mma_atom, [lvl])
                    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
                    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
                    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
                    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
                    lvl = Level("warp", "thread_block", (warp, 1), TensorLayout((warp, 1)), (repeat_m, head_size // 16))
                    tiled_mma_o = TiledMma(mma_atom, [lvl])
                    ba._tiled_mma_pairs.append((tiled_mma_qk, tiled_mma_o))


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


def header():
    p = torch.cuda.get_device_properties(0)
    try:
        commit = subprocess.run(["git", "-C", HEXCUTE_ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "?"
    return (f"<!-- provenance -->\nDevice: {p.name} (cc {p.major}.{p.minor}, {p.multi_processor_count} SMs)\n"
            f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')} | hidet(hexcute): {hidet.__version__} @ {commit} | torch: {torch.__version__} | cuda: {torch.version.cuda}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", type=int, default=32)
    ap.add_argument("--seq", default="1024,2048,4096")
    ap.add_argument("--dims", default="64,80,96,112,128")
    ap.add_argument("--sm", type=int, default=86, help="Hexcute code path (compute capability it generates for); 86 = Ampere-class mma.sync")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "..", "results", "hexcute_cache"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    hidet.option.cache_dir(os.path.abspath(args.cache_dir))
    hidet.option.search_space(2)
    dims = [int(x) for x in args.dims.split(",")]
    register_extra_head_sizes([d for d in dims if d % 32 != 0])
    p = torch.cuda.get_device_properties(0)
    tag = f"{p.name.lower().replace(' ', '_')}_sm{p.major}{p.minor}"
    lines = [header(), f"seed={args.seed} bh={args.bh} sm_ver={args.sm} search_space=2 (all tiled-MMA pairs x pingpong/pipeline stages 2-4)", ""]

    def out(msg):
        print(msg, flush=True); lines.append(msg)

    for N in [int(x) for x in args.seq.split(",")]:
        for D in dims:
            flops = 4.0 * args.bh * N * N * D
            q = torch.randn(1, N, args.bh, D, device="cuda", dtype=torch.float16)      # Hexcute's [B, S, H, D]
            k = torch.randn_like(q); v = torch.randn_like(q); o = torch.empty_like(q)
            ref = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2)
            t0 = time.time()
            try:
                artifacts = ba.FlashAttention(1, N, args.bh, D, N, args.bh, args.sm).build()
            except Exception as e:
                out(f"N={N} D={D} BH={args.bh}: Hexcute build FAILED: {str(e)[:160]}"); continue
            build_s = time.time() - t0
            hq, hk, hv, ho = (hidet.from_torch(t) for t in (q, k, v, o))
            best = None; n_ok = 0
            for i, (mod, func) in enumerate(artifacts):
                try:
                    func(hq, hk, hv, ho); torch.cuda.synchronize()
                    ok = torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2)
                    ms = time_fn(lambda: func(hq, hk, hv, ho))
                except Exception as e:
                    out(f"  N={N} D={D} cand={i}: FAILED {str(e)[:100]}"); continue
                kw = mod._tuning_kwargs
                stages = kw.get("stages", "pingpong")
                out(f"  N={N} D={D} cand={i} stages={stages}: {ms:.3f} ms {flops/ms/1e9:.1f} TFLOP/s correct={ok}")
                n_ok += ok
                if ok and (best is None or ms < best[1]):
                    best = (i, ms, stages)
            out(f"N={N} D={D} BH={args.bh}: Hexcute best {best} ({len(artifacts)} candidates, {n_ok} correct, build {build_s:.0f} s)"
                + (f" {flops/best[1]/1e9:.1f} TFLOP/s" if best else ""))
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "results", tag, "hexcute_attn.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
