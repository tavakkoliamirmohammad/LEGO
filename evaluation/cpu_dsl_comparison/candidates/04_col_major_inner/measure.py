"""Measure baseline vs cpu_dsl for 04_col_major_inner."""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, M, N, _MN


def _measure(fn, warmup=5, timed=30):
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
    A_2d = rng.standard_normal((M, N)).astype(np.float32)

    C_base = np.empty((M, N), dtype=np.float32)
    t_base = _measure(lambda: kernel_scalar(A_2d, C_base))

    A_flat = np.ascontiguousarray(A_2d).ravel()
    C_dsl = np.zeros(_MN, dtype=np.float32)

    try:
        compiled = kernel_cpu_dsl.compile()
        t_dsl = _measure(lambda: compiled(A_flat, C_dsl))
        speedup = t_base / t_dsl
        verdict = "WIN" if speedup > 1.05 else "PARITY" if speedup > 0.95 else "LOSS"
        notes = ""
    except Exception as e:
        t_dsl = float("nan")
        speedup = float("nan")
        verdict = "ERROR"
        notes = str(e)

    rec = {
        "name": "04_col_major_inner",
        "baseline_ms": round(t_base, 4),
        "cpu_dsl_ms": round(t_dsl, 4) if t_dsl == t_dsl else t_dsl,
        "speedup": round(speedup, 4) if speedup == speedup else speedup,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
