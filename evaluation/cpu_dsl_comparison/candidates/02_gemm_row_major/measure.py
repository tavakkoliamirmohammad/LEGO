"""Measure baseline vs cpu_dsl for 02_gemm_row_major.

Isolation harness: scalar-JIT vs vectorized-JIT (apples-to-apples).
GEMM is not a large-N candidate (the kernel is N-dimensional nested loops);
we keep M=N=K=64 since larger sizes time out at 120s.
"""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, M, N, K, _MK, _KN, _MN


def _measure(fn, warmup=3, timed=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return float(np.median(times)) / 1e6


def main():
    rng = np.random.default_rng(0)
    A_2d = rng.standard_normal((M, K)).astype(np.float32)
    B_2d = rng.standard_normal((K, N)).astype(np.float32)

    # NumPy baseline.
    C_base_2d = np.zeros((M, N), dtype=np.float32)
    t_numpy = _measure(lambda: kernel_scalar(A_2d, B_2d, C_base_2d))

    A_flat = A_2d.ravel()
    B_flat = B_2d.ravel()

    # Scalar JIT baseline (apples-to-apples).
    t_scalar_jit = float('nan')
    try:
        scalar_jit = kernel_cpu_dsl.compile(target='scalar')
        C_scalar = np.zeros(_MN, dtype=np.float32)
        t_scalar_jit = _measure(lambda: scalar_jit(A_flat, B_flat, C_scalar))
    except Exception as e:
        t_scalar_jit = float('nan')

    # Vectorized JIT.
    t_vec_jit = float('nan')
    notes = ""
    try:
        vec_jit = kernel_cpu_dsl.compile(target='x86')
        C_dsl = np.zeros(_MN, dtype=np.float32)
        t_vec_jit = _measure(lambda: vec_jit(A_flat, B_flat, C_dsl))
    except Exception as e:
        t_vec_jit = float('nan')
        notes = str(e)

    def _safe_ratio(a, b):
        if a == a and b == b and b > 0:
            return round(a / b, 4)
        return float('nan')

    speedup_isolated = _safe_ratio(t_scalar_jit, t_vec_jit)
    speedup_vs_numpy = _safe_ratio(t_numpy, t_vec_jit)

    verdict = "ERROR" if t_vec_jit != t_vec_jit and notes else (
        "WIN" if speedup_isolated > 1.05 else
        "PARITY" if speedup_isolated > 0.95 else "LOSS"
    ) if t_vec_jit == t_vec_jit else "ERROR"

    rec = {
        "name": "02_gemm_row_major",
        "N": M,
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar_jit, 4) if t_scalar_jit == t_scalar_jit else t_scalar_jit,
        "vec_jit_ms": round(t_vec_jit, 4) if t_vec_jit == t_vec_jit else t_vec_jit,
        "speedup_isolated_jit": speedup_isolated,
        "speedup_vs_numpy": speedup_vs_numpy,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
