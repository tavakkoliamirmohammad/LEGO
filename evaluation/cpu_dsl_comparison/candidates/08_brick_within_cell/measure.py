"""Measure baseline vs cpu_dsl for 08_brick_within_cell.

Mirrors evaluation/cpu_vector_proof/brick_within_cell/measure.py but
conforms to the standard harness JSON schema used by run_all.py.
"""
import json
import sys
import time
import numpy as np

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
    return float(np.median(times)) / 1e6


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B_ref = np.zeros(N, dtype=np.float32)

    B_base = B_ref.copy()
    t_base = _measure(lambda: kernel_scalar(A, B_base))

    B_dsl = B_ref.copy()
    try:
        compiled = kernel_cpu_dsl.compile()
        t_dsl = _measure(lambda: compiled(A, B_dsl))
        # Correctness
        B_check_base = B_ref.copy()
        B_check_dsl = B_ref.copy()
        kernel_scalar(A, B_check_base)
        compiled(A, B_check_dsl)
        np.testing.assert_allclose(B_check_dsl, B_check_base, rtol=1e-4,
                                   err_msg="brick_within_cell correctness mismatch")
        speedup = t_base / t_dsl
        verdict = "WIN" if speedup > 1.05 else "PARITY" if speedup > 0.95 else "LOSS"
        notes = ("NumPy BLAS loop is heavily optimised; beating it requires "
                 "FMA intrinsic quality (R12 roadmap).")
    except Exception as e:
        t_dsl = float("nan")
        speedup = float("nan")
        verdict = "ERROR"
        notes = str(e)

    rec = {
        "name": "08_brick_within_cell",
        "baseline_ms": round(t_base, 4),
        "cpu_dsl_ms": round(t_dsl, 4) if t_dsl == t_dsl else t_dsl,
        "speedup": round(speedup, 4) if speedup == speedup else speedup,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
