"""Measure baseline vs cpu_dsl for 02_gemm_row_major.

Isolation harness: scalar-JIT vs vectorized-JIT (apples-to-apples).
GEMM is not a large-N candidate (the kernel is N-dimensional nested loops);
we keep M=N=K=64 since larger sizes time out at 120s.

Timing methodology: bench() wraps the kernel in a single JIT call with N_ITERS
iterations to eliminate Python-loop overhead (~4-7µs/call) that otherwise
dominates the ~0.13ms kernel time and biases results.
"""
import json
import math
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, M, N, K, _MK, _KN, _MN

N_ITERS = 500


def main():
    rng = np.random.default_rng(0)
    A_2d = rng.standard_normal((M, K)).astype(np.float32)
    B_2d = rng.standard_normal((K, N)).astype(np.float32)

    # NumPy baseline (accumulate into preallocated C).
    C_base_2d = np.zeros((M, N), dtype=np.float32)
    warmup, timed = 100, 1000
    for _ in range(warmup):
        kernel_scalar(A_2d, B_2d, C_base_2d)
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        kernel_scalar(A_2d, B_2d, C_base_2d)
    t_numpy = (time.perf_counter_ns() - t0) / timed / 1e6

    A_flat = np.ascontiguousarray(A_2d.ravel())
    B_flat = np.ascontiguousarray(B_2d.ravel())
    C_flat = np.zeros(_MN, dtype=np.float32)

    # Scalar JIT baseline — use bench() to eliminate invoke overhead.
    t_scalar = float("nan")
    try:
        kernel_cpu_dsl.bench(A_flat, B_flat, C_flat, n_iters=10, target="scalar")  # warmup compile
        t_scalar = kernel_cpu_dsl.bench(A_flat, B_flat, C_flat, n_iters=N_ITERS, target="scalar")
    except Exception:
        pass

    # Vectorized JIT — bench() for same reason.
    t_vec = float("nan")
    notes = ""
    try:
        kernel_cpu_dsl.bench(A_flat, B_flat, C_flat, n_iters=10, target="x86")    # warmup compile
        t_vec = kernel_cpu_dsl.bench(A_flat, B_flat, C_flat, n_iters=N_ITERS, target="x86")
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a / b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float("nan")

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")

    print(json.dumps({
        "name": "02_gemm_row_major",
        "N": M,
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar, 6) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 6) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sr(t_numpy, t_vec),
        "verdict": verdict,
        "notes": notes or "3-level GEMM; j innermost unit-stride; vectorizes over j",
    }))


if __name__ == "__main__":
    main()
