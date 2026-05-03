"""Measure 37_stencil_nonpow2_brick: simplified stencil with non-pow2 brick size."""
import json
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, N_FLAT, _INNER, _OFFSET


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
    try:
        sj = kernel_cpu_dsl.compile(target='scalar')
        B_sc = np.zeros(N_FLAT, dtype=np.float32)
        t_scalar_jit = _measure(lambda: sj(A, B_sc))
    except Exception:
        pass

    t_vec_jit = float('nan')
    notes = ""
    try:
        vj = kernel_cpu_dsl.compile(target='x86')
        B_v = np.zeros(N_FLAT, dtype=np.float32)
        t_vec_jit = _measure(lambda: vj(A, B_v))
        B_sc2 = np.zeros(N_FLAT, dtype=np.float32)
        kernel_cpu_dsl.compile(target='scalar')(A, B_sc2)
        B_v2 = np.zeros(N_FLAT, dtype=np.float32)
        vj(A, B_v2)
        np.testing.assert_allclose(B_v2[_OFFSET:_OFFSET + _INNER],
                                   B_sc2[_OFFSET:_OFFSET + _INNER], rtol=1e-4)
    except Exception as e:
        t_vec_jit = float('nan')
        notes = str(e)

    def _sr(a, b):
        return round(a / b, 4) if a == a and b == b and b > 0 else float('nan')

    sp_iso = _sr(t_scalar_jit, t_vec_jit)
    sp_np  = _sr(t_numpy, t_vec_jit)
    verdict = "ERROR" if (t_vec_jit != t_vec_jit and notes) else (
        "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso > 0.95 else "LOSS"
    ) if t_vec_jit == t_vec_jit else "PARITY"

    print(json.dumps({
        "name": "37_stencil_nonpow2_brick",
        "N": _INNER,
        "layout_class": "Brick+non-pow2",
        "prior_verdict": "LOSS",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar_jit, 4) if t_scalar_jit == t_scalar_jit else t_scalar_jit,
        "vec_jit_ms": round(t_vec_jit, 4) if t_vec_jit == t_vec_jit else t_vec_jit,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sp_np,
        "verdict": verdict,
        "notes": notes or "simplified non-pow2 brick: 5pt 2D stencil on 30x30 grid",
    }))


if __name__ == "__main__":
    main()
