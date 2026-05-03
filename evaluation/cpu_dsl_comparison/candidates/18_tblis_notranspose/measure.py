"""Measure 18_tblis_notranspose: simplified tiled GEMM (TBLIS layout pattern).

CASTLE candidate 10. Layout class: TBLIS.
Prior verdicts: AMD WIN, Intel LOSS.

This simplified version captures the TBLIS access pattern (tiled GEMM with
explicit K-blocking) without the TBLIS microkernel runtime selection layer.
The k-reduction axis is scalar (R18 reduction guard prevents vectorization);
the outer M*N tile_range is the vectorizable axis.
"""
import json
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, M, N, K, _MN, _MK, _KN


def _measure(fn, warmup=50, timed=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return (time.perf_counter_ns() - t0) / timed / 1e6


def main():
    rng = np.random.default_rng(42)
    A_2d = rng.standard_normal((M, K)).astype(np.float32)
    B_2d = rng.standard_normal((K, N)).astype(np.float32)
    C_base = np.zeros((M, N), dtype=np.float32)

    t_numpy = _measure(lambda: kernel_scalar(A_2d, B_2d, C_base))

    A_flat = A_2d.ravel()
    B_flat = B_2d.ravel()

    t_scalar_jit = float('nan')
    try:
        scalar_jit = kernel_cpu_dsl.compile(target='scalar')
        C_scalar = np.zeros(_MN, dtype=np.float32)
        t_scalar_jit = _measure(lambda: scalar_jit(A_flat, B_flat, C_scalar))
    except Exception:
        pass

    t_vec_jit = float('nan')
    notes = ""
    try:
        vec_jit = kernel_cpu_dsl.compile(target='x86')
        C_vec = np.zeros(_MN, dtype=np.float32)
        t_vec_jit = _measure(lambda: vec_jit(A_flat, B_flat, C_vec))
        # Quick correctness check: run once and compare to numpy
        C_check = np.zeros(_MN, dtype=np.float32)
        vec_jit(A_flat, B_flat, C_check)
        C_ref = A_2d @ B_2d
        np.testing.assert_allclose(C_check.reshape(M, N), C_ref, rtol=1e-3)
    except Exception as e:
        t_vec_jit = float('nan')
        notes = str(e)

    def _sr(a, b):
        if a == a and b == b and b > 0:
            return round(a / b, 4)
        return float('nan')

    speedup_iso = _sr(t_scalar_jit, t_vec_jit)
    speedup_np = _sr(t_numpy, t_vec_jit)
    verdict = "ERROR" if (t_vec_jit != t_vec_jit and notes) else (
        "WIN" if speedup_iso > 1.05 else
        "PARITY" if speedup_iso > 0.95 else "LOSS"
    ) if t_vec_jit == t_vec_jit else "PARITY"

    print(json.dumps({
        "name": "18_tblis_notranspose",
        "N": M,
        "layout_class": "TBLIS",
        "prior_verdict": "LOSS",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar_jit, 4) if t_scalar_jit == t_scalar_jit else t_scalar_jit,
        "vec_jit_ms": round(t_vec_jit, 4) if t_vec_jit == t_vec_jit else t_vec_jit,
        "speedup_isolated_jit": speedup_iso,
        "speedup_vs_numpy": speedup_np,
        "verdict": verdict,
        "notes": notes or "simplified TBLIS pattern: flat GEMM with k-reduction",
    }))


if __name__ == "__main__":
    main()
