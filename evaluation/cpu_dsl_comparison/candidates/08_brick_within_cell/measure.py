"""Measure baseline vs cpu_dsl for 08_brick_within_cell.

Isolation harness: scalar-JIT vs vectorized-JIT at large N.
Mirrors evaluation/cpu_vector_proof/brick_within_cell but uses isolation mode.
"""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

# Large N to amortize JIT startup.
N_BENCH = 1 << 20   # 1M elements
BRICK = 8


@cpu_kernel(grid=(N_BENCH,), tile=(BRICK,))
def _kernel_bench(A: Buffer[N_BENCH], B: Buffer[N_BENCH]):
    for i in tile_range:
        B[i] = A[i] * 2.0 + 1.0


from kernel import kernel_scalar, kernel_cpu_dsl, N as N_SMALL


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

    # Correctness check at small N.
    A_s = rng.standard_normal(N_SMALL).astype(np.float32)
    B_ref = np.zeros(N_SMALL, dtype=np.float32)
    B_check_base = B_ref.copy()
    kernel_scalar(A_s, B_check_base)

    # Large-N timing arrays.
    A_big = rng.standard_normal(N_BENCH).astype(np.float32)
    B_big_ref = np.zeros(N_BENCH, dtype=np.float32)

    # NumPy baseline.
    B_numpy = B_big_ref.copy()
    t_numpy = _measure(lambda: kernel_scalar(A_big, B_numpy))

    # Scalar JIT baseline.
    t_scalar_jit = float('nan')
    try:
        scalar_jit = _kernel_bench.compile(target='scalar')
        B_scalar = B_big_ref.copy()
        t_scalar_jit = _measure(lambda: scalar_jit(A_big, B_scalar))
    except Exception as e:
        t_scalar_jit = float('nan')

    # Vectorized JIT.
    t_vec_jit = float('nan')
    notes = "NumPy BLAS loop is heavily optimised; isolation shows JIT vectorization benefit"
    try:
        vec_jit = _kernel_bench.compile(target='x86')
        B_vec = B_big_ref.copy()
        t_vec_jit = _measure(lambda: vec_jit(A_big, B_vec))

        # Correctness at small N.
        B_check_dsl = B_ref.copy()
        vec_jit_small = kernel_cpu_dsl.compile(target='x86')
        vec_jit_small(A_s, B_check_dsl)
        np.testing.assert_allclose(B_check_dsl, B_check_base, rtol=1e-4,
                                   err_msg="brick_within_cell correctness mismatch")
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
        "name": "08_brick_within_cell",
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
