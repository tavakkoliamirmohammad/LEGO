"""Measure baseline vs cpu_dsl for 01_saxpy.

Isolation harness: compares scalar-JIT vs vectorized-JIT (apples-to-apples)
in addition to the NumPy BLAS baseline. Uses large N to amortize JIT startup.
"""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

# Large N to amortize ~100µs JIT call overhead and expose vectorization benefit.
N_BENCH = 1 << 20   # 1M elements
TILE = 16


@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _kernel_bench(a: float, X: Buffer[N_BENCH], Y: Buffer[N_BENCH]):
    for i in tile_range:
        Y[i] = a * X[i] + Y[i]


# Also import the small-N kernel for correctness check.
from kernel import kernel_scalar, kernel_cpu_dsl, N as N_SMALL


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
    a = np.float32(2.5)

    # --- correctness check at small N ---
    X_s = rng.standard_normal(N_SMALL).astype(np.float32)
    Y_ref = rng.standard_normal(N_SMALL).astype(np.float32)
    Y_check_base = Y_ref.copy()
    kernel_scalar(a, X_s, Y_check_base)

    # --- large-N timing arrays ---
    X_big = rng.standard_normal(N_BENCH).astype(np.float32)
    Y_big_ref = rng.standard_normal(N_BENCH).astype(np.float32)

    # NumPy baseline (large N)
    Y_numpy = Y_big_ref.copy()
    t_numpy = _measure(lambda: kernel_scalar(a, X_big, Y_numpy))

    # Scalar JIT baseline (apples-to-apples).
    t_scalar_jit = float('nan')
    try:
        scalar_jit = _kernel_bench.compile(target='scalar')
        Y_scalar = Y_big_ref.copy()
        t_scalar_jit = _measure(lambda: scalar_jit(a, X_big, Y_scalar))
    except Exception as e:
        t_scalar_jit = float('nan')
        _scalar_err = str(e)

    # Vectorized JIT.
    t_vec_jit = float('nan')
    notes = ""
    try:
        vec_jit = _kernel_bench.compile(target='x86')
        Y_vec = Y_big_ref.copy()
        t_vec_jit = _measure(lambda: vec_jit(a, X_big, Y_vec))

        # Correctness at small N.
        Y_check_dsl = Y_ref.copy()
        vec_jit_small = kernel_cpu_dsl.compile(target='x86')
        vec_jit_small(a, X_s, Y_check_dsl)
        np.testing.assert_allclose(Y_check_dsl, Y_check_base, rtol=1e-4,
                                   err_msg="saxpy correctness mismatch")
    except Exception as e:
        t_vec_jit = float('nan')
        notes = str(e)

    def _safe_ratio(a, b):
        if a == a and b == b and b > 0:
            return round(a / b, 4)
        return float('nan')

    speedup_isolated = _safe_ratio(t_scalar_jit, t_vec_jit)
    speedup_vs_numpy = _safe_ratio(t_numpy, t_vec_jit)

    verdict = "ERROR" if notes and t_vec_jit != t_vec_jit else (
        "WIN" if speedup_isolated > 1.05 else
        "PARITY" if speedup_isolated > 0.95 else "LOSS"
    )

    rec = {
        "name": "01_saxpy",
        "N": N_BENCH,
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
