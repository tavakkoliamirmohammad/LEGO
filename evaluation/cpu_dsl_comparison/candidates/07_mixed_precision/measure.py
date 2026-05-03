"""Measure baseline vs cpu_dsl for 07_mixed_precision.

Isolation harness: scalar-JIT vs vectorized-JIT at large N.
"""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

# Large N to amortize JIT startup overhead.
N_BENCH = 1 << 20   # 1M elements
TILE = 16
_SCALE = 1.5


@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _kernel_bench(X: Buffer[N_BENCH], Y: Buffer[N_BENCH]):
    for i in tile_range:
        Y[i] = 1.5 * X[i] + Y[i]


from kernel import kernel_scalar, kernel_cpu_dsl, N as N_SMALL, SCALE


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

    # Correctness check at small N.
    X_s = rng.standard_normal(N_SMALL).astype(np.float32)
    Y_ref = rng.standard_normal(N_SMALL).astype(np.float32)
    Y_check_base = Y_ref.copy()
    kernel_scalar(X_s, Y_check_base)

    # Large-N timing arrays.
    X_big = rng.standard_normal(N_BENCH).astype(np.float32)
    Y_big_ref = rng.standard_normal(N_BENCH).astype(np.float32)

    # NumPy baseline.
    Y_numpy = Y_big_ref.copy()
    t_numpy = _measure(lambda: kernel_scalar(X_big, Y_numpy))

    # Scalar JIT baseline — bench_self_timed uses clock_gettime inside the JIT'd code.
    t_scalar_jit = float('nan')
    Y_scalar = Y_big_ref.copy()
    try:
        t_scalar_jit = _kernel_bench.bench_self_timed(X_big, Y_scalar, n_iters=1000, n_warmup=100, target='scalar')
    except Exception as e:
        t_scalar_jit = float('nan')

    # Vectorized JIT — bench_self_timed for MLIR-level timing.
    t_vec_jit = float('nan')
    Y_vec = Y_big_ref.copy()
    notes = "scale=1.5 baked as f32 constant; 2-buffer form avoids v1 R12 store bug"
    try:
        # Correctness at small N first.
        Y_check_dsl = Y_ref.copy()
        vec_jit_small = kernel_cpu_dsl.compile(target='x86')
        vec_jit_small(X_s, Y_check_dsl)
        np.testing.assert_allclose(Y_check_dsl, Y_check_base, rtol=1e-4,
                                   err_msg="mixed_precision correctness mismatch")
        t_vec_jit = _kernel_bench.bench_self_timed(X_big, Y_vec, n_iters=1000, n_warmup=100, target='x86')
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
        "name": "07_mixed_precision",
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
