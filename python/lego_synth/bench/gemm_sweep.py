"""GEMM: CASTLE (synthesized shared layouts) vs cuBLAS vs Triton on square f16 problems."""
import argparse
import torch

from lego_synth.search import Problem, synthesize
from lego_synth.problems import problem_for
from lego_synth.machine import ampere
from lego_synth.export import to_lego_tile
from lego_synth.kernels.gemm_mma import build_gemm
from lego_synth.kernels.launch import CastleKernel
from lego_synth.bench.triton_matmul import triton_matmul, triton_matmul_tuned, time_fn


def synth_tile(rows, cols):
    """Conflict-free layout for a b16 tile written by cp.async and read by ldmatrix (full tier)."""
    sol = synthesize(problem_for("staging", rows, cols, 2, ampere()), "full")
    return sol.layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096", help="square sizes, or '' to skip")
    ap.add_argument("--shapes", default="", help="semicolon-separated MxNxK LLM shapes, e.g. 4096x6144x4096;4096x14336x4096")
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", default="64x64x32w2x2,128x128x32w2x2,128x64x32w2x2,128x128x32w4x2,128x256x32w2x4,256x128x32w4x2")
    ap.add_argument("--stages", default="2,3,4")
    args = ap.parse_args()
    problems = [(n, n, n) for n in [int(x) for x in args.sizes.split(",") if x]]
    problems += [tuple(int(v) for v in sh.split("x")) for sh in args.shapes.split(";") if sh]
    lines = []

    def out(msg):
        print(msg, flush=True); lines.append(msg)

    for (M, N, K) in problems:
        n = f"{M}x{N}x{K}"
        A = torch.randn(M, K, device="cuda", dtype=torch.float16); B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float32)
        flops = 2.0 * M * N * K
        ref = time_fn(lambda: torch.matmul(A, B))
        tri = time_fn(lambda: triton_matmul(A, B, C))
        tri_tuned = time_fn(lambda: triton_matmul_tuned(A, B, C))
        best = None
        for cfg0 in args.configs.split(","):
          for st in [int(x) for x in args.stages.split(",")]:
            cfg = f"{cfg0}s{st}"
            dims, w = cfg0.split("w"); BM, BN, BK = [int(x) for x in dims.split("x")]; wm, wn = [int(x) for x in w.split("x")]
            try:
                k = CastleKernel.from_builder(build_gemm(M, N, K, BM, BN, BK, to_lego_tile(synth_tile(BM, BK)), to_lego_tile(synth_tile(BK, BN)), warps_m=wm, warps_n=wn, stages=st))
                k.launch(A, B, C); torch.cuda.synchronize()
                ok = torch.allclose(C, A.float() @ B.float(), atol=1e-1, rtol=1e-2)
                ms = k.time(A, B, C, iters=30, warmup=5)
            except Exception as e:
                out(f"  {n} cfg={cfg}: FAILED {str(e)[:120]}"); continue
            out(f"  {n} cfg={cfg}: {ms:.3f} ms {flops/ms/1e9:.1f} TFLOP/s correct={ok}")
            if ok and (best is None or ms < best[1]):
                best = (cfg, ms)
        out(f"{n}: cuBLAS {ref:.3f} ms ({flops/ref/1e9:.1f} TF) | Triton fixed {tri:.3f} ms ({flops/tri/1e9:.1f} TF) | Triton autotuned {tri_tuned:.3f} ms ({flops/tri_tuned/1e9:.1f} TF) | CASTLE best {best}")
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
