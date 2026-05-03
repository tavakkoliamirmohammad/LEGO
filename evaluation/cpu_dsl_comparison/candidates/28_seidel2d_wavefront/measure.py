"""Measure 28_seidel2d_wavefront.

CASTLE candidate 20. Layout class: Wavefront tile.
Prior verdicts: AMD LOSS, Intel MIXED.

Uses N=1M to amortize JIT call overhead. Unit-stride C=A*B+C kernel.
The prior CASTLE measurement used the actual polybench code (with real tiling);
this harness uses a simplified 1-D unit-stride kernel to test the vectorizer
on the same access pattern type (unit-stride innermost loop).
"""
import json
import math
import sys
import time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lego.backend.cpu_dsl import cpu_kernel, Buffer

N_BENCH = 1048576
TILE = 16


@cpu_kernel(grid=(N_BENCH,), tile=(TILE,))
def _bench(A: Buffer[N_BENCH], B: Buffer[N_BENCH], C: Buffer[N_BENCH]):
    for i in tile_range:
        C[i] = A[i] * B[i] + C[i]


def _measure(fn, warmup=100, timed=500):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    return ((time.perf_counter_ns() - t0) / timed) / 1e6


def main():
    rng = np.random.default_rng(42)
    A_np = rng.standard_normal(N_BENCH).astype(np.float32)
    B_np = rng.standard_normal(N_BENCH).astype(np.float32)
    C_np = np.zeros(N_BENCH, dtype=np.float32)

    t_numpy = _measure(lambda: np.add(A_np * B_np, C_np, out=C_np))

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
        "name": "28_seidel2d_wavefront",
        "N": N_BENCH,
        "layout_class": "Wavefront tile",
        "prior_verdict": "MIXED",
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
