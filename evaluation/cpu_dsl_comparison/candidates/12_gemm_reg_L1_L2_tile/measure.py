"""Measure 12_gemm_reg_L1_L2_tile.

CASTLE candidate 4. Layout class: Reg+L1+L2 tile.
Prior verdicts: AMD WIN, Intel WIN.

CASTLE WIN mechanism: tiling restructures iteration order to fit working
set in L1/L2 caches → fewer cache misses → faster than naive triple loop
that gcc -O3 cannot tile without guided hints.

Kernel: N×N GEMM (N=512) with register + L1 tiling.
C baseline (gemm_O3, N=512): plain 3-loop i,k,j naive GEMM.

Note: LEGO vectorizes the innermost j-tile loop (TILE_L1=16 → L_strip=16
one AVX-512 register). The outer i,k loops are scalar. The WIN vs naive C
comes from cache-friendly iteration order, not from wider SIMD.
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_cpu_dsl, kernel_scalar, N, _NN


def _measure(fn, warmup=5, timed=30):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def main():
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal(_NN).astype(np.float32)
    B_np = rng.standard_normal(_NN).astype(np.float32)
    C_np = np.zeros(_NN, dtype=np.float32)

    # NumPy reference
    t_numpy = _measure(lambda: kernel_scalar(A_np, B_np, C_np))

    # Scalar JIT
    t_scalar = float("nan")
    try:
        sj = kernel_cpu_dsl.compile(target="scalar")
        C_sc = np.zeros(_NN, dtype=np.float32)
        t_scalar = _measure(lambda: sj(A_np, B_np, C_sc))
    except Exception:
        pass

    # Vectorized JIT
    t_vec = float("nan")
    notes = ""
    try:
        vj = kernel_cpu_dsl.compile(target="x86")
        C_vec = np.zeros(_NN, dtype=np.float32)
        t_vec = _measure(lambda: vj(A_np, B_np, C_vec))
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a / b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float("nan")

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")
    print(json.dumps({
        "name": "12_gemm_reg_L1_L2_tile",
        "N": _NN,
        "layout_class": "Reg+L1+L2 tile",
        "prior_verdict": "WIN",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar, 4) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 4) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sr(t_numpy, t_vec),
        "verdict": verdict,
        "notes": notes,
    }))


if __name__ == "__main__":
    main()
