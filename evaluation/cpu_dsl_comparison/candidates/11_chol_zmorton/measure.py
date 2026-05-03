"""Measure 11_chol_zmorton.

CASTLE candidate 3. Layout class: Z-Morton.
Prior verdicts: AMD LOSS, Intel LOSS.
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

N_BENCH = 65536
TILE = 16


@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _bench(A: Buffer[N_BENCH], B: Buffer[N_BENCH], C: Buffer[N_BENCH]):
    for i in tile_range:
        ti = i & 0x5555
        tj = (i >> 1) & 0x5555
        morton = (ti | (tj << 1)) & (N_BENCH - 1)
        C[i] = A[morton] * B[i] + C[i]


def _measure(fn, warmup=50, timed=200):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def _numpy_scalar(A, B, C):
    idx = np.arange(N_BENCH, dtype=np.int32)
    ti = idx & 0x5555
    tj = (idx >> 1) & 0x5555
    morton = (ti | (tj << 1)) & (N_BENCH - 1)
    np.add(A[morton] * B, C, out=C)


def main():
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal(N_BENCH).astype(np.float32)
    B_np = rng.standard_normal(N_BENCH).astype(np.float32)
    C_np = np.zeros(N_BENCH, dtype=np.float32)

    t_numpy = _measure(lambda: _numpy_scalar(A_np, B_np, C_np))

    t_scalar = float("nan")
    try:
        sj = _bench.compile(target="scalar")
        t_scalar = _measure(lambda: sj(A_np, B_np, C_np))
    except Exception:
        pass

    t_vec = float("nan")
    notes = ""
    try:
        vj = _bench.compile(target="x86")
        t_vec = _measure(lambda: vj(A_np, B_np, C_np))
    except Exception as e:
        notes = str(e)

    def sr(a, b):
        return round(a/b, 4) if (not math.isnan(a) and not math.isnan(b) and b > 0) else float("nan")

    sp_iso = sr(t_scalar, t_vec)
    verdict = ("ERROR" if notes and math.isnan(t_vec) else
               "WIN" if sp_iso > 1.05 else "PARITY" if sp_iso >= 0.95 else "LOSS")
    print(json.dumps({
        "name": "11_chol_zmorton",
        "N": N_BENCH,
        "layout_class": "Z-Morton",
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
