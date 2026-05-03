"""Measure 09_gemm_zmorton: GEMM-style gather with Z-Morton index swizzle.

CASTLE candidate 01 (polybench-gemm-zmorton). Prior: AMD WIN, Intel WIN.
Layout class: Z-Morton.
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

N_BENCH = 1 << 16  # 64K — large enough to stress the gather path
TILE = 16


@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _kernel_bench_vec(A: Buffer[N_BENCH], B: Buffer[N_BENCH], C: Buffer[N_BENCH]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N_BENCH - 1)
        C[i] = C[i] + A[morton] * B[i]


def _measure(fn, warmup=50, timed=300):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def _numpy_scalar(A, B, C):
    """Inline NumPy scalar equivalent (avoids N mismatch from kernel.py)."""
    indices = np.arange(N_BENCH, dtype=np.int32)
    ti = indices & 0x5555
    tj = (indices >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N_BENCH - 1)
    C[:] += A[morton] * B


def main():
    rng = np.random.default_rng(42)
    A_big = rng.standard_normal(N_BENCH).astype(np.float32)
    B_big = rng.standard_normal(N_BENCH).astype(np.float32)
    C_big = np.zeros(N_BENCH, dtype=np.float32)

    # NumPy baseline
    C_np = C_big.copy()
    t_numpy = _measure(lambda: _numpy_scalar(A_big, B_big, C_np))

    # Scalar JIT — bench_self_timed uses clock_gettime inside the JIT'd code.
    t_scalar = float('nan')
    C_s = C_big.copy()
    try:
        t_scalar = _kernel_bench_vec.bench_self_timed(A_big, B_big, C_s, n_iters=300, n_warmup=50, target='scalar')
    except Exception:
        pass

    # Vector JIT — bench_self_timed for MLIR-level timing.
    t_vec = float('nan')
    C_v = C_big.copy()
    notes = ""
    try:
        t_vec = _kernel_bench_vec.bench_self_timed(A_big, B_big, C_v, n_iters=300, n_warmup=50, target='x86')
    except Exception as e:
        notes = str(e)

    def _sr(a, b):
        return round(a/b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float('nan')

    sp_iso = _sr(t_scalar, t_vec)
    sp_np = _sr(t_numpy, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else
               "PARITY" if sp_iso >= 0.95 else "LOSS")

    print(json.dumps({
        "name": "09_gemm_zmorton",
        "N": N_BENCH,
        "layout_class": "Z-Morton",
        "prior_verdict": "WIN",
        "numpy_ms": round(t_numpy, 4),
        "scalar_jit_ms": round(t_scalar, 4) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 4) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sp_np,
        "verdict": verdict,
        "notes": notes,
    }))


if __name__ == "__main__":
    main()
