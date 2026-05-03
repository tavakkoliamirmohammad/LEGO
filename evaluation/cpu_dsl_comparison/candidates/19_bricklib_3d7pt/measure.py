"""Measure 19_bricklib_3d7pt: simplified 3D 7-point stencil (brick layout pattern).

CASTLE candidate 11. Layout class: Brick.
Prior verdicts: AMD LOSS, Intel LOSS.

Simplified version: 3D stencil on flat 1D buffer with compile-time neighbor
offsets (±1, ±NZ, ±NX*NY). No integer division in kernel body. The inner
tile_range loop has unit-stride access for ±1 neighbors and strided gather for
±NX*NY / ±NZ neighbors.
"""
import json
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N_FLAT, _INNER, _OFFSET, TILE


def _measure(fn, warmup=100, timed=1000):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return (time.perf_counter_ns() - t0) / timed / 1e6


def main():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(N_FLAT).astype(np.float32)
    B_ref = np.zeros(N_FLAT, dtype=np.float32)

    t_numpy = _measure(lambda: kernel_scalar(A, B_ref))

    t_scalar_jit = float('nan')
    B_scalar = np.zeros(N_FLAT, dtype=np.float32)
    try:
        t_scalar_jit = kernel_cpu_dsl.bench_self_timed(A, B_scalar, n_iters=1000, n_warmup=100, target='scalar')
    except Exception:
        pass

    t_vec_jit = float('nan')
    B_vec = np.zeros(N_FLAT, dtype=np.float32)
    notes = ""
    try:
        t_vec_jit = kernel_cpu_dsl.bench_self_timed(A, B_vec, n_iters=1000, n_warmup=100, target='x86')
        # Correctness: compare interior to scalar.
        B_sc2 = np.zeros(N_FLAT, dtype=np.float32)
        kernel_cpu_dsl.compile(target='scalar')(A, B_sc2)
        B_v2 = np.zeros(N_FLAT, dtype=np.float32)
        kernel_cpu_dsl.compile(target='x86')(A, B_v2)
        np.testing.assert_allclose(B_v2[_OFFSET:_OFFSET + _INNER],
                                   B_sc2[_OFFSET:_OFFSET + _INNER], rtol=1e-4)
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
        "name": "19_bricklib_3d7pt",
        "N": _INNER,
        "layout_class": "Brick",
        "prior_verdict": "LOSS",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar_jit, 4) if t_scalar_jit == t_scalar_jit else t_scalar_jit,
        "vec_jit_ms": round(t_vec_jit, 4) if t_vec_jit == t_vec_jit else t_vec_jit,
        "speedup_isolated_jit": speedup_iso,
        "speedup_vs_numpy": speedup_np,
        "verdict": verdict,
        "notes": notes or "simplified brick pattern: flat 3D stencil with offset-based addressing",
    }))


if __name__ == "__main__":
    main()
