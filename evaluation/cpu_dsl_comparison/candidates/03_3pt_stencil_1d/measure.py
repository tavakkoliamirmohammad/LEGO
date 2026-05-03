"""Measure baseline vs cpu_dsl for 03_3pt_stencil_1d.

Isolation harness: scalar-JIT vs vectorized-JIT (apples-to-apples).

Timing methodology: bench() wraps the kernel in a single JIT call with N_ITERS
iterations to eliminate Python-loop overhead (~0.77µs/call) that otherwise
dominates the ~0.1µs kernel time at N=1K and biases results toward LOSS.
"""
import json
import math
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N

N_ITERS = 3000


def _measure(fn, warmup=100, timed=1000):
    """Amortized measurement: compile-once, call many times in a hot loop."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    t_total = time.perf_counter_ns() - t0
    return (t_total / timed) / 1e6  # average ms per call


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N).astype(np.float32)
    B_ref = np.zeros(N, dtype=np.float32)

    # NumPy baseline.
    B_base = B_ref.copy()
    t_numpy = _measure(lambda: kernel_scalar(A, B_base))

    # Scalar JIT baseline (apples-to-apples) — use bench() to avoid
    # Python-loop overhead that would bias small-N results toward LOSS.
    t_scalar_jit = float('nan')
    try:
        kernel_cpu_dsl.bench(A, B_ref, n_iters=100, target='scalar')  # warmup
        t_scalar_jit = kernel_cpu_dsl.bench(A, B_ref, n_iters=N_ITERS, target='scalar')
    except Exception:
        t_scalar_jit = float('nan')

    # Vectorized JIT — bench() for unbiased timing.
    t_vec_jit = float('nan')
    notes = ""
    try:
        # Correctness check with compile() before bench timing.
        vec_jit = kernel_cpu_dsl.compile(target='x86')
        B_check_base = np.zeros(N, dtype=np.float32)
        B_check_dsl = np.zeros(N, dtype=np.float32)
        kernel_scalar(A, B_check_base)
        vec_jit(A, B_check_dsl)
        np.testing.assert_allclose(B_check_dsl[1:-1], B_check_base[1:-1],
                                   rtol=1e-4,
                                   err_msg="3pt stencil correctness mismatch")
        # Now measure with bench().
        kernel_cpu_dsl.bench(A, B_ref, n_iters=100, target='x86')  # warmup
        t_vec_jit = kernel_cpu_dsl.bench(A, B_ref, n_iters=N_ITERS, target='x86')
    except Exception as e:
        t_vec_jit = float('nan')
        notes = str(e)

    def _safe_ratio(a, b):
        if a == a and b == b and b > 0:
            return round(a / b, 4)
        return float('nan')

    speedup_isolated = _safe_ratio(t_scalar_jit, t_vec_jit)
    speedup_vs_numpy = _safe_ratio(t_numpy, t_vec_jit)

    verdict = "ERROR" if t_vec_jit != t_vec_jit else (
        "WIN" if speedup_isolated > 1.05 else
        "PARITY" if speedup_isolated > 0.95 else "LOSS"
    )

    rec = {
        "name": "03_3pt_stencil_1d",
        "N": N,
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
