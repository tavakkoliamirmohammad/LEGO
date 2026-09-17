"""Same GEMM kernel, three shared-memory layouts: naive row-major, +8-element padding, synthesized swizzle."""
import argparse
import os
import torch

from lego_synth.layouts import RowMajor
from lego_synth.problems import problem_for
from lego_synth.search import synthesize
from lego_synth.machine import for_device
from lego_synth.kernels.target import detect_target, metadata_header
from lego_synth.eval_np import conflicts
from lego_synth.export import to_lego_tile
from lego_synth.kernels.gemm_mma import build_gemm
from lego_synth.kernels.launch import CastleKernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="4096x4096x4096"); ap.add_argument("--cfg", default="128x128x32w4x2s3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    target = detect_target()
    M, N, K = [int(x) for x in a.shape.split("x")]
    dims, w = a.cfg.split("w"); BM, BN, BK = [int(x) for x in dims.split("x")]; wm, wn = w.split("s")[0].split("x"); st = int(w.split("s")[1])
    wm, wn = int(wm), int(wn)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16); B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32); ref = A.float() @ B.float(); flops = 2.0 * M * N * K
    variants = {
        "naive": (RowMajor(BM, BK), RowMajor(BK, BN)),
        "pad8": (RowMajor(BM, BK, pad=8), RowMajor(BK, BN, pad=8)),
        "synthesized": (synthesize(problem_for("staging", BM, BK, 2, for_device()), "full").layout,
                        synthesize(problem_for("staging", BK, BN, 2, for_device()), "full").layout),
    }
    lines = [metadata_header(target), f"### GEMM layout ablation, {a.shape}, cfg {a.cfg}, {target.name}", "",
             "| shared layouts (A, B) | predicted ldmatrix conflicts per tile (A+B) | footprint (A+B, KB/stage) | ms | TFLOP/s | correct |", "|---|---|---|---|---|---|"]
    for name, (LA, LB) in variants.items():
        pa = problem_for("staging", BM, BK, 2, for_device()); pb = problem_for("staging", BK, BN, 2, for_device())
        pred = sum(conflicts(LA, p, 2, for_device()) for p in pa.patterns) + sum(conflicts(LB, p, 2, for_device()) for p in pb.patterns)
        k = CastleKernel.from_builder(build_gemm(M, N, K, BM, BN, BK, to_lego_tile(LA), to_lego_tile(LB), warps_m=wm, warps_n=wn, stages=st))
        k.launch(A, B, C); torch.cuda.synchronize(); ok = torch.allclose(C, ref, atol=1e-1, rtol=1e-2)
        ms = k.time(A, B, C, iters=20, warmup=5)
        lines.append(f"| {name}: {LA}, {LB} | {pred} | {(LA.footprint + LB.footprint) * 2 / 1024:.0f} | {ms:.3f} | {flops/ms/1e9:.1f} | {ok} |")
    text = "\n".join(lines); print(text)
    out_path = a.out or os.path.join(os.path.dirname(__file__), "..", "results", target.tag, "gemm_layout_ablation.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write(text + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
