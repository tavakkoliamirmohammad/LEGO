"""GEMM: CASTLE (synthesized shared layouts) vs cuBLAS vs Triton (fixed and autotuned) on square and LLM
projection shapes, f16 in, f32 accumulate, f16 out (cuBLAS's output type for f16 inputs)."""
import argparse
import os
import torch

from lego_synth.search import Problem, synthesize
from lego_synth.problems import problem_for
from lego_synth.machine import for_device
from lego_synth.kernels.target import detect_target, metadata_header
from lego_synth.export import to_lego_tile
from lego_synth.kernels.gemm_mma import build_gemm
from lego_synth.kernels.launch import CastleKernel
from lego_synth.bench.triton_matmul import triton_matmul, triton_matmul_tuned, time_fn, LAST, clock_str


def synth_tile(rows, cols):
    """Conflict-free layout for a b16 tile written by cp.async and read by ldmatrix (full tier)."""
    sol = synthesize(problem_for("staging", rows, cols, 2, for_device()), "full")
    return sol.layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,2048,4096", help="square sizes, or '' to skip")
    ap.add_argument("--shapes", default="", help="semicolon-separated MxNxK LLM shapes, e.g. 4096x6144x4096;4096x14336x4096")
    ap.add_argument("--out", default=None, help="result file; default results/<device-tag>/gemm.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default="128x128x32w2x2,128x128x32w4x2,128x256x32w2x4,256x128x32w4x2,128x64x32w2x2,64x128x32w2x2")
    ap.add_argument("--stages", default="2,3")
    ap.add_argument("--group_m", default="0,8")
    ap.add_argument("--out_dtype", default="f16", choices=["f16", "f32"])
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    target = detect_target()
    lines = [metadata_header(target), f"seed={args.seed} configs={args.configs} stages={args.stages} group_m={args.group_m} out_dtype={args.out_dtype}", ""]
    problems = [(n, n, n) for n in [int(x) for x in args.sizes.split(",") if x]]
    problems += [tuple(int(v) for v in sh.split("x")) for sh in args.shapes.split(";") if sh]
    def out(msg):
        print(msg, flush=True); lines.append(msg)

    for (M, N, K) in problems:
        n = f"{M}x{N}x{K}"
        A = torch.randn(M, K, device="cuda", dtype=torch.float16); B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16 if args.out_dtype == "f16" else torch.float32)
        C32 = torch.empty(M, N, device="cuda", dtype=torch.float32)     # the Triton tutorial kernel stores f32
        flops = 2.0 * M * N * K
        clocks = {}
        ref = time_fn(lambda: torch.matmul(A, B)); clocks["cublas"] = clock_str()
        tri = time_fn(lambda: triton_matmul(A, B, C32)); clocks["triton"] = clock_str()
        tri_tuned = time_fn(lambda: triton_matmul_tuned(A, B, C32)); clocks["triton_tuned"] = clock_str()
        expected = A.float() @ B.float()
        best = None
        for cfg0 in args.configs.split(","):
          for st in [int(x) for x in args.stages.split(",")]:
           for gm in [int(x) for x in args.group_m.split(",")]:
            cfg = f"{cfg0}s{st}g{gm}"
            dims, w = cfg0.split("w"); BM, BN, BK = [int(x) for x in dims.split("x")]; wm, wn = [int(x) for x in w.split("x")]
            smem_bytes = st * (BM * BK + BK * BN) * 2
            if smem_bytes > target.smem_per_block_optin:   # >48 KB uses dynamic shared memory up to the device's opt-in limit
                out(f"  {n} cfg={cfg}: skipped (shared {smem_bytes // 1024} KB > {target.smem_per_block_optin // 1024} KB opt-in)"); continue
            try:
                k = CastleKernel.from_builder(build_gemm(M, N, K, BM, BN, BK, to_lego_tile(synth_tile(BM, BK)), to_lego_tile(synth_tile(BK, BN)), warps_m=wm, warps_n=wn, stages=st, group_m=gm, out_dtype=args.out_dtype))
                k.launch(A, B, C); torch.cuda.synchronize()
                ok = torch.allclose(C.float(), expected, atol=1e-1, rtol=1e-2)
                ms = k.time(A, B, C, iters=30, warmup=5)
            except Exception as e:
                out(f"  {n} cfg={cfg}: FAILED {str(e)[:120]}"); continue
            out(f"  {n} cfg={cfg}: {ms:.3f} ms {flops/ms/1e9:.1f} TFLOP/s correct={ok} regs={k._fn.num_regs} clk={clock_str()}")
            if ok and (best is None or ms < best[1]):
                best = (cfg, ms); clocks["castle"] = clock_str()
        out(f"{n}: cuBLAS {ref:.3f} ms ({flops/ref/1e9:.1f} TF) | Triton fixed {tri:.3f} ms ({flops/tri/1e9:.1f} TF) | Triton autotuned {tri_tuned:.3f} ms ({flops/tri_tuned/1e9:.1f} TF) | CASTLE best {best} | clocks MHz {clocks}")
    out_path = args.out or os.path.join(os.path.dirname(__file__), "..", "results", target.tag, "gemm.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
