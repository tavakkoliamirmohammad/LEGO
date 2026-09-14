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
    ap.add_argument("--sizes", default="1024,2048,4096")
    ap.add_argument("--configs", default="64x64x32w2x2,128x128x32w2x2,128x64x32w2x2,128x128x32w4x2")
    args = ap.parse_args()
    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        M = N = K = n
        A = torch.randn(M, K, device="cuda", dtype=torch.float16); B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float32)
        flops = 2.0 * M * N * K
        ref = time_fn(lambda: torch.matmul(A, B))
        tri = time_fn(lambda: triton_matmul(A, B, C))
        tri_tuned = time_fn(lambda: triton_matmul_tuned(A, B, C))
        best = None
        for cfg in args.configs.split(","):
            dims, w = cfg.split("w"); BM, BN, BK = [int(x) for x in dims.split("x")]; wm, wn = [int(x) for x in w.split("x")]
            try:
                k = CastleKernel.from_builder(build_gemm(M, N, K, BM, BN, BK, to_lego_tile(synth_tile(BM, BK)), to_lego_tile(synth_tile(BK, BN)), warps_m=wm, warps_n=wn))
                k.launch(A, B, C); torch.cuda.synchronize()
                ok = torch.allclose(C, A.float() @ B.float(), atol=1e-1, rtol=1e-2)
                ms = k.time(A, B, C, iters=30, warmup=5)
            except Exception as e:
                print(f"  n={n} cfg={cfg}: FAILED {str(e)[:120]}"); continue
            print(f"  n={n} cfg={cfg}: {ms:.3f} ms {flops/ms/1e9:.1f} TFLOP/s correct={ok}")
            if ok and (best is None or ms < best[1]):
                best = (cfg, ms)
        print(f"n={n}: cuBLAS {ref:.3f} ms ({flops/ref/1e9:.1f} TF) | Triton fixed {tri:.3f} ms ({flops/tri/1e9:.1f} TF) | Triton autotuned {tri_tuned:.3f} ms ({flops/tri_tuned/1e9:.1f} TF) | CASTLE best {best}")


if __name__ == "__main__":
    main()
