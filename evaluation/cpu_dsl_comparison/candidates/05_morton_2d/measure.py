"""Measure baseline vs cpu_dsl for 05_morton_2d.

Isolation harness: scalar-JIT vs vectorized-JIT (apples-to-apples).
Real Morton decode via bitwise ops (&, >>, |) in the DSL.
The cpu_dsl version should emit vector.gather via the NonAffine path.
"""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N


def _measure(fn, warmup=100, timed=1000):
    """Amortized measurement: compile-once, call many times in a hot loop.

    JIT compilation happens once BEFORE this function is called (fn is
    already a compiled callable). Only the kernel call cost is measured.
    100 warm-up iterations flush instruction caches and branch predictors.
    The full timed block is timed as ONE wall-clock interval and divided,
    eliminating per-call timer overhead (~20ns / call on modern CPUs).
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    t_total = time.perf_counter_ns() - t0
    return (t_total / timed) / 1e6  # average ms per call


def main():
    rng = np.random.default_rng(0)
    A_buf = rng.standard_normal(N).astype(np.float32)
    B_base = np.zeros(N, dtype=np.float32)

    # NumPy baseline (kernel_scalar uses A, B naming).
    t_numpy = _measure(lambda: kernel_scalar(A_buf, B_base))

    # Scalar JIT baseline — bench_self_timed uses clock_gettime inside the JIT'd code.
    t_scalar_jit = float('nan')
    out_scalar = np.zeros(N, dtype=np.float32)
    notes = ""
    try:
        t_scalar_jit = kernel_cpu_dsl.bench_self_timed(A_buf, out_scalar, n_iters=1000, n_warmup=100, target='scalar')
    except Exception as e:
        t_scalar_jit = float('nan')
        notes = f"scalar compile error: {e}"

    # Vectorized JIT — bench_self_timed for MLIR-level timing.
    t_vec_jit = float('nan')
    out_dsl = np.zeros(N, dtype=np.float32)
    try:
        # Correctness check first.
        vec_jit = kernel_cpu_dsl.compile(target='x86')
        vec_jit(A_buf, out_dsl)
        if not np.allclose(B_base, out_dsl, atol=1e-4):
            notes = "WARNING: DSL output differs from NumPy reference"
        else:
            notes = notes or "real Morton gather via bitwise DSL ops"
        t_vec_jit = kernel_cpu_dsl.bench_self_timed(A_buf, out_dsl, n_iters=1000, n_warmup=100, target='x86')
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
        "name": "05_morton_2d",
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
