"""Attention forward: CASTLE (synthesized K/V shared layouts) vs PyTorch SDPA backends vs Triton tutorial.

Head dimensions 64/80/96/112/128; PyTorch's flash and cuDNN backends pad odd head dimensions
internally; Triton's tutorial kernel requires a power-of-two head dimension, so for 80/96/112
its number is the padded-to-128 kernel (reported as such).
"""
import argparse
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from lego_synth.problems import problem_for
from lego_synth.search import synthesize
from lego_synth.machine import ampere
from lego_synth.export import to_lego_tile
from lego_synth.kernels.attn_mma import build_attention
from lego_synth.kernels.launch import CastleKernel
from lego_synth.bench.triton_matmul import time_fn


def kv_layout(block_n, d):
    return to_lego_tile(synthesize(problem_for("staging", block_n, d, 2, ampere()), "full").layout)


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
    ap.add_argument("--configs", default="64x32s2,64x64s2,128x32s2,128x64s2,64x64s3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()
    lines = []

    def out(msg):
        print(msg, flush=True); lines.append(msg)

    for N in [int(x) for x in args.seq.split(",")]:
        for D in [int(x) for x in args.dims.split(",")]:
            q = torch.randn(args.bh, N, D, device="cuda", dtype=torch.float16)
            k = torch.randn_like(q); v = torch.randn_like(q); o = torch.empty(args.bh, N, D, device="cuda", dtype=torch.float32)
            flops = 4.0 * args.bh * N * N * D * (0.5 if args.causal else 1.0)
            fl = sdpa_time(q, k, v, SDPBackend.FLASH_ATTENTION, args.causal); cu = sdpa_time(q, k, v, SDPBackend.CUDNN_ATTENTION, args.causal); ef = sdpa_time(q, k, v, SDPBackend.EFFICIENT_ATTENTION, args.causal)
            best = None
            for cfg in args.configs.split(","):
                bm, rest = cfg.split("x"); bn, st = rest.split("s"); bm, bn, st = int(bm), int(bn), int(st)
                try:
                    L = kv_layout(bn, D)
                    kern = CastleKernel.from_builder(build_attention(args.bh, N, D, L, L, BLOCK_M=bm, BLOCK_N=bn, stages=st, causal=args.causal))
                    kern.launch(q, k, v, o); torch.cuda.synchronize()
                    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=args.causal)
                    ok = torch.allclose(o, ref, atol=2e-2, rtol=2e-2)
                    ms = kern.time(q, k, v, o, iters=20, warmup=5)
                except Exception as e:
                    out(f"  N={N} D={D} cfg={cfg}: FAILED {str(e)[:100]}"); continue
                out(f"  N={N} D={D} cfg={cfg}: {ms:.3f} ms {flops/ms/1e9:.1f} TFLOP/s correct={ok}")
                if ok and (best is None or ms < best[1]):
                    best = (cfg, ms)
            fmt = lambda t: f"{t:.3f} ms" if t else "n/a"
            out(f"N={N} D={D} BH={args.bh}{' causal' if args.causal else ''}: SDPA flash {fmt(fl)} | cuDNN {fmt(cu)} | efficient {fmt(ef)} | CASTLE best {best}")
    if args.out:
        open(args.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
