"""Measure 25_nw_antidiag.

CASTLE candidate 17. Layout class: Antidiag tile.
Prior verdicts: AMD MIXED, Intel LOSS.

This candidate models a stride-2 access pattern characteristic of Antidiag tile.
NumPy baseline uses vectorized strided indexing (A[::2] * 2.0) for fair comparison.
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

N_BENCH = 16384
STRIDE = 2
TILE = 16


@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _bench(A: Buffer[N_BENCH * STRIDE], B: Buffer[N_BENCH]):
    for i in tile_range:
        B[i] = A[i * STRIDE] * 2.0


def _measure(fn, warmup=50, timed=300):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def main():
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal(N_BENCH * STRIDE).astype(np.float32)
    B_np = np.zeros(N_BENCH, dtype=np.float32)

    # NumPy baseline: vectorized strided copy (A[::STRIDE] * 2.0 → B).
    # This is the reference that any decent compiler should match.
    t_numpy = _measure(lambda: np.multiply(A_np[::STRIDE], 2.0, out=B_np))

    t_scalar = float("nan")
    try:
        sj = _bench.compile(target="scalar")
        t_scalar = _measure(lambda: sj(A_np, B_np))
    except Exception:
        pass

    t_vec = float("nan")
    notes = ""
    try:
        vj = _bench.compile(target="x86")
        t_vec = _measure(lambda: vj(A_np, B_np))
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a/b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float("nan")

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")
    print(json.dumps({
        "name": "25_nw_antidiag",
        "N": N_BENCH,
        "layout_class": "Antidiag tile",
        "prior_verdict": "LOSS",
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
