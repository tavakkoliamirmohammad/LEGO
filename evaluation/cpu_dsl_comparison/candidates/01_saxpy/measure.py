"""Measure baseline vs cpu_dsl for 01_saxpy."""
import json
import sys
import time
import numpy as np

# Allow running from any directory by adjusting the path for sibling imports.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N


def _measure(fn, warmup=5, timed=30):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(timed):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return float(np.median(times)) / 1e6  # ms


def main():
    rng = np.random.default_rng(0)
    a = np.float32(2.5)
    X = rng.standard_normal(N).astype(np.float32)
    Y_ref = rng.standard_normal(N).astype(np.float32)

    Y_base = Y_ref.copy()
    t_base = _measure(lambda: kernel_scalar(a, X, Y_base))

    Y_dsl = Y_ref.copy()
    try:
        compiled = kernel_cpu_dsl.compile()
        t_dsl = _measure(lambda: compiled(a, X, Y_dsl))
        # Correctness: use Y_ref state before any mutation, compare fresh.
        Y_check_base = Y_ref.copy()
        Y_check_dsl = Y_ref.copy()
        kernel_scalar(a, X, Y_check_base)
        compiled(a, X, Y_check_dsl)
        np.testing.assert_allclose(Y_check_dsl, Y_check_base, rtol=1e-4,
                                   err_msg="saxpy correctness mismatch")
        speedup = t_base / t_dsl
        verdict = "WIN" if speedup > 1.05 else "PARITY" if speedup > 0.95 else "LOSS"
        notes = ""
    except Exception as e:
        t_dsl = float("nan")
        speedup = float("nan")
        verdict = "ERROR"
        notes = str(e)

    rec = {
        "name": "01_saxpy",
        "baseline_ms": round(t_base, 4),
        "cpu_dsl_ms": round(t_dsl, 4) if t_dsl == t_dsl else t_dsl,
        "speedup": round(speedup, 4) if speedup == speedup else speedup,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
