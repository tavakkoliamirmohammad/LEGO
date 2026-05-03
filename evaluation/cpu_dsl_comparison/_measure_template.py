"""Shared measurement harness template (not imported directly — copied into each candidate).

Each candidate's measure.py calls _run_measurement() with its kernel definitions.
"""
import json
import math
import sys
import time


def _measure(fn, warmup=100, timed=500):
    """Amortized measurement: compile-once, call many times.

    The full timed block is timed as ONE wall-clock interval and divided,
    eliminating per-call timer overhead (~20ns/call on modern CPUs).
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(timed):
        fn()
    t_total = time.perf_counter_ns() - t0
    return (t_total / timed) / 1e6   # ms per call


def _safe_ratio(a, b):
    if (isinstance(a, float) and not math.isnan(a) and
            isinstance(b, float) and not math.isnan(b) and b > 0):
        return round(a / b, 4)
    return float('nan')


def emit_result(name, N, t_numpy, t_scalar, t_vec, notes="", prior_verdict="",
                layout_class=""):
    sp_iso = _safe_ratio(t_scalar, t_vec)
    sp_np = _safe_ratio(t_numpy, t_vec)
    verdict = (
        "SKIP" if math.isnan(t_vec) and notes.startswith("SKIP")
        else "ERROR" if (math.isnan(t_vec) and not notes.startswith("SKIP"))
        else "WIN"   if sp_iso > 1.05
        else "PARITY" if sp_iso >= 0.95
        else "LOSS"
    )
    rec = {
        "name": name,
        "N": N,
        "layout_class": layout_class,
        "prior_verdict": prior_verdict,
        "numpy_ms": round(t_numpy, 4) if not math.isnan(t_numpy) else t_numpy,
        "scalar_jit_ms": round(t_scalar, 4) if not math.isnan(t_scalar) else t_scalar,
        "vec_jit_ms": round(t_vec, 4) if not math.isnan(t_vec) else t_vec,
        "speedup_isolated_jit": sp_iso,
        "speedup_vs_numpy": sp_np,
        "verdict": verdict,
        "notes": notes,
    }
    print(json.dumps(rec))
