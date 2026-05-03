"""Measure baseline vs cpu_dsl for 07_mixed_precision."""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N, SCALE


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
    X = rng.standard_normal(N).astype(np.float32)
    Y_ref = rng.standard_normal(N).astype(np.float32)

    Y_base = Y_ref.copy()
    t_base = _measure(lambda: kernel_scalar(X, Y_base))

    Y_dsl = Y_ref.copy()
    try:
        compiled = kernel_cpu_dsl.compile()
        t_dsl = _measure(lambda: compiled(X, Y_dsl))
        # Correctness
        Y_check_base = Y_ref.copy()
        Y_check_dsl = Y_ref.copy()
        kernel_scalar(X, Y_check_base)
        compiled(X, Y_check_dsl)
        np.testing.assert_allclose(Y_check_dsl, Y_check_base, rtol=1e-4,
                                   err_msg="mixed_precision correctness mismatch")
        speedup = t_base / t_dsl
        verdict = "WIN" if speedup > 1.05 else "PARITY" if speedup > 0.95 else "LOSS"
        notes = "scale=1.5 baked as f32 constant; 2-buffer form avoids v1 R12 store bug"
    except Exception as e:
        t_dsl = float("nan")
        speedup = float("nan")
        verdict = "ERROR"
        notes = str(e)

    rec = {
        "name": "07_mixed_precision",
        "baseline_ms": round(t_base, 4),
        "cpu_dsl_ms": round(t_dsl, 4) if t_dsl == t_dsl else t_dsl,
        "speedup": round(speedup, 4) if speedup == speedup else speedup,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
