"""Measure baseline vs cpu_dsl for 06_self_update (shift-add stencil).

Isolation harness: scalar-JIT vs vectorized-JIT (apples-to-apples).

Timing methodology: bench() wraps the kernel in a single JIT call with N_ITERS
iterations to eliminate Python-loop overhead (~4µs/call) that otherwise
dominates the ~0.3µs kernel time at N=4K and biases results toward LOSS.

Expected: vec_isolated ≈ 5-8× — no loop-carried dep, LEGO emits AVX-512 vadd.
"""
import json
import math
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N, N_INNER

N_ITERS = 10000


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)

    # NumPy baseline (shift-add via vectorized slice ops).
    warmup, timed = 100, 1000
    kernel_scalar(A, B)  # warmup
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        kernel_scalar(A, B)
    t_numpy = (time.perf_counter_ns() - t0) / timed / 1e6

    # Scalar JIT baseline — use bench() to measure pure kernel time.
    t_scalar = float("nan")
    try:
        kernel_cpu_dsl.bench(A, B, n_iters=100, target="scalar")   # warmup compile
        t_scalar = kernel_cpu_dsl.bench(A, B, n_iters=N_ITERS, target="scalar")
    except Exception:
        pass

    # Vectorized JIT — use bench() for the same reason.
    t_vec = float("nan")
    notes = ""
    try:
        kernel_cpu_dsl.bench(A, B, n_iters=100, target="x86")      # warmup compile
        t_vec = kernel_cpu_dsl.bench(A, B, n_iters=N_ITERS, target="x86")
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a / b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float("nan")

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")

    print(json.dumps({
        "name": "06_self_update",
        "N": N,
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar, 6) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 6) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sr(t_numpy, t_vec),
        "verdict": verdict,
        "notes": notes or "shift-add stencil B[i+1]=A[i]+A[i+1]; no loop-carried dep; fully vectorizable",
    }))


if __name__ == "__main__":
    main()
