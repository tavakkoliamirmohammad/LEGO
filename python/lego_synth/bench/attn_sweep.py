"""Attention forward: CASTLE (synthesized Q/K/V shared layouts) vs PyTorch SDPA backends vs the Triton
tutorial kernel (fixed config and autotuned).

Head dimensions 64/80/96/112/128; PyTorch's flash and cuDNN backends pad odd head dimensions
internally; Triton's tutorial kernel requires a power-of-two head dimension, so for 80/96/112 it runs
padded to 128 with zero-filled boundary loads (bench/triton_attention.py).  Every kernel writes f16.

Config grammar: <BLOCK_M>x<BLOCK_N>s<stages>[r32][q0]  (r32: two m-tiles per warp; q0: Q fragments
re-read from shared memory instead of registers).
"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from lego_synth.problems import problem_for
from lego_synth.search import synthesize
from lego_synth.machine import for_device
from lego_synth.kernels.target import detect_target, metadata_header
from lego_synth.export import to_lego_tile
from lego_synth.kernels.attn_mma import build_attention
from lego_synth.kernels.launch import CastleKernel
from lego_synth.bench.triton_matmul import time_fn, LAST, clock_str
from lego_synth.bench.triton_attention import triton_attention, triton_attention_tuned


_layout_cache = {}


def kv_layout(block_n, d):
    if (block_n, d) not in _layout_cache:
        _layout_cache[(block_n, d)] = to_lego_tile(synthesize(problem_for("staging", block_n, d, 2, for_device()), "full").layout)
    return _layout_cache[(block_n, d)]


def parse_cfg(cfg):
    """'128x64s2r32q0' -> (BLOCK_M, BLOCK_N, stages, rows_per_warp, q_in_regs)."""
    import re
    m = re.fullmatch(r"(\d+)x(\d+)s(\d+)(r32)?(q0)?", cfg)
    if not m:
        raise ValueError(f"bad config {cfg!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), 32 if m.group(4) else 16, not m.group(5)


def triton_time(q, k, v, causal, tuned):
    try:
        fn = (lambda: triton_attention_tuned(q, k, v, causal=causal)) if tuned else (lambda: triton_attention(q, k, v, causal=causal))
        return time_fn(fn, iters=20, warmup=5)
    except Exception:
        return None


def sdpa_time(q, k, v, backend, causal=False):
    """PyTorch's fused backends need 4-D (B, H, N, D) inputs."""
    q4, k4, v4 = (t.unsqueeze(0) for t in (q, k, v))
    try:
        with sdpa_kernel(backend):
            return time_fn(lambda: F.scaled_dot_product_attention(q4, k4, v4, is_causal=causal), iters=20, warmup=5)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", type=int, default=32)
    ap.add_argument("--seq", default="1024,2048,4096")
    ap.add_argument("--dims", default="64,80,96,112,128")
    ap.add_argument("--configs", default="64x32s2,64x64s2,128x32s2r32,128x64s2r32,128x64s3r32,128x32s2r32q0,128x64s2r32q0,64x32s2r32")
    ap.add_argument("--out_dtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--out", default=None, help="result file; default results/<device-tag>/attn[_causal].md")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3, help="interleaved timing rounds over all systems for the summary line")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    target = detect_target()
    lines = [metadata_header(target), f"seed={args.seed} bh={args.bh} configs={args.configs} causal={args.causal} out_dtype={args.out_dtype} rounds={args.rounds}", ""]

    def out(msg):
        print(msg, flush=True); lines.append(msg)

    for N in [int(x) for x in args.seq.split(",")]:
        for D in [int(x) for x in args.dims.split(",")]:
            q = torch.randn(args.bh, N, D, device="cuda", dtype=torch.float16)
            k = torch.randn_like(q); v = torch.randn_like(q)
            o = torch.empty(args.bh, N, D, device="cuda", dtype=torch.float16 if args.out_dtype == "f16" else torch.float32)
            flops = 4.0 * args.bh * N * N * D * (0.5 if args.causal else 1.0)
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=args.causal)
            try:
                triton_attention_tuned(q, k, v, causal=args.causal); torch.cuda.synchronize()   # autotune once, outside the timed rounds
            except Exception:
                pass
            best = None; best_kernel = None
            for cfg in args.configs.split(","):
                bm, bn, st, rpw, qreg = parse_cfg(cfg)
                if (bm * D + st * 2 * bn * D) * 2 > target.smem_per_block_optin:
                    out(f"  N={N} D={D} cfg={cfg}: skipped (shared memory above the device opt-in limit)"); continue
                if args.causal and (bm % bn or (bm // bn) % st):
                    out(f"  N={N} D={D} cfg={cfg}: skipped (causal needs BLOCK_M a multiple of BLOCK_N x stages)"); continue
                try:
                    L = kv_layout(bn, D)
                    kern = CastleKernel.from_builder(build_attention(args.bh, N, D, L, L, BLOCK_M=bm, BLOCK_N=bn, stages=st, causal=args.causal,
                                                                    rows_per_warp=rpw, q_in_regs=qreg, out_dtype=args.out_dtype, sQ_layout=kv_layout(bm, D)))
                    kern.launch(q, k, v, o); torch.cuda.synchronize()
                    ok = torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2)
                    ms = kern.time(q, k, v, o, iters=20, warmup=5)
                except Exception as e:
                    out(f"  N={N} D={D} cfg={cfg}: FAILED {str(e)[:100]}"); continue
                out(f"  N={N} D={D} cfg={cfg}: {ms:.3f} ms {flops/ms/1e9:.1f} TFLOP/s correct={ok} regs={kern._fn.num_regs} spill={kern._fn.local_size_bytes}B clk={clock_str()}")
                if ok and (best is None or ms < best[1]):
                    best = (cfg, ms); best_kernel = kern
            # Interleaved rounds over all systems (see gemm_sweep): each keeps its best of `rounds`.
            systems = {"flash": lambda: sdpa_time(q, k, v, SDPBackend.FLASH_ATTENTION, args.causal),
                       "cudnn": lambda: sdpa_time(q, k, v, SDPBackend.CUDNN_ATTENTION, args.causal),
                       "efficient": lambda: sdpa_time(q, k, v, SDPBackend.EFFICIENT_ATTENTION, args.causal),
                       "triton": lambda: triton_time(q, k, v, args.causal, tuned=False),
                       "triton_tuned": lambda: triton_time(q, k, v, args.causal, tuned=True)}
            if best_kernel is not None:
                systems["castle"] = lambda: best_kernel.time(q, k, v, o)
            times = {s_: [] for s_ in systems}; clocks = {s_: [] for s_ in systems}
            for _ in range(args.rounds):
                for name, fn in systems.items():
                    t = fn(); clocks[name].append(clock_str())
                    if t is not None:
                        times[name].append(t)
            pick = lambda name: min(times[name]) if times[name] else None
            fl, cu, ef, tr, tr_tuned = (pick(x) for x in ("flash", "cudnn", "efficient", "triton", "triton_tuned"))
            if best_kernel is not None:
                best = (best[0], pick("castle"))
            fmt = lambda t: f"{t:.3f} ms" if t else "n/a"
            out(f"N={N} D={D} BH={args.bh}{' causal' if args.causal else ''}: SDPA flash {fmt(fl)} | cuDNN {fmt(cu)} | efficient {fmt(ef)} | Triton fixed {fmt(tr)} | Triton autotuned {fmt(tr_tuned)} | CASTLE best {best} | rounds={args.rounds} clocks MHz {clocks}")
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "results", target.tag, "attn_causal.md" if args.causal else "attn.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
