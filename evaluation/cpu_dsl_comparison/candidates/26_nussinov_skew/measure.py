"""Measure 26_nussinov_skew.

CASTLE candidate 18. Layout class: Skew tile.
Prior verdicts: AMD WIN, Intel WIN.

Skew tile: models stride-2 memory access pattern (deinterleave-style),
characteristic of the Nussinov skew tiling approximation.
Benchmarks the stride-2 kernel from kernel.py against the stride2_16k C baseline.
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_cpu_dsl, N, N_BUF, TILE

N_BENCH = N        # 4096 elements (logical), buffer N_BUF = 8192


def _measure(fn, warmup=200, timed=3000):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def main():
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal(N_BUF).astype(np.float32)
    B_np = np.zeros(N_BENCH, dtype=np.float32)

    # NumPy baseline: strided gather (A[::2] * 2.0 → B)
    t_numpy = _measure(lambda: np.multiply(A_np[::2], 2.0, out=B_np))

    t_scalar = float("nan")
    B_sc = np.zeros(N_BENCH, dtype=np.float32)
    try:
        t_scalar = kernel_cpu_dsl.bench_self_timed(A_np, B_sc, n_iters=3000, n_warmup=100, target="scalar")
    except Exception:
        pass

    t_vec = float("nan")
    B_v = np.zeros(N_BENCH, dtype=np.float32)
    notes = ""
    try:
        t_vec = kernel_cpu_dsl.bench_self_timed(A_np, B_v, n_iters=3000, n_warmup=100, target="x86")
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a/b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float("nan")

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")
    print(json.dumps({
        "name": "26_nussinov_skew",
        "N": N_BENCH,
        "layout_class": "Skew tile",
        "prior_verdict": "WIN",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar, 4) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 4) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sr(t_numpy, t_vec),
        "verdict": verdict,
        "notes": notes or "stride-2 gather: B[i] = A[i*2]*2.0 at N=4096",
    }))


if __name__ == "__main__":
    main()
