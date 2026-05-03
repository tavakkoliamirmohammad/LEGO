"""Measure baseline vs cpu_dsl for 05_morton_2d.

Note: v1 DSL does not support bitwise AND/shift, so the cpu_dsl version
uses a proxy index (z → z).  The baseline uses proper Morton decode.
Verdict will be LOSS or ERROR until bitwise ops land in the DSL.
"""
import json
import sys
import time
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kernel import kernel_scalar, kernel_cpu_dsl, _NN


def _measure(fn, warmup=5, timed=30):
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
    M_buf = rng.standard_normal(_NN).astype(np.float32)
    out_base = np.zeros(_NN, dtype=np.float32)

    t_base = _measure(lambda: kernel_scalar(M_buf, out_base))

    out_dsl = np.zeros(_NN, dtype=np.float32)
    try:
        compiled = kernel_cpu_dsl.compile()
        t_dsl = _measure(lambda: compiled(M_buf, out_dsl))
        speedup = t_base / t_dsl
        verdict = "WIN" if speedup > 1.05 else "PARITY" if speedup > 0.95 else "LOSS"
        notes = ("DSL uses identity index (z→z); Morton decode blocked by missing "
                 "bitwise ops in v1 DSL.  Correctness check skipped.")
    except Exception as e:
        t_dsl = float("nan")
        speedup = float("nan")
        verdict = "ERROR"
        notes = str(e)

    rec = {
        "name": "05_morton_2d",
        "baseline_ms": round(t_base, 4),
        "cpu_dsl_ms": round(t_dsl, 4) if t_dsl == t_dsl else t_dsl,
        "speedup": round(speedup, 4) if speedup == speedup else speedup,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
