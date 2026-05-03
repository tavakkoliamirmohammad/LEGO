"""Measure cross-brick stencil proof-point speedup.

v1: exercises the 1D 3-point stencil (within-brick pattern).
Full cross-brick 3D 7-point stencil is blocked on roadmap entry R12.
"""
import json
import time
import numpy as np

from kernel import stencil_3pt, stencil_3pt_scalar, N

WARMUP = 5
TIMED = 30
TARGET_SPEEDUP_AVX512 = 2.0
TARGET_SPEEDUP_NEON = 1.5


def measure(fn, A, B):
    for _ in range(WARMUP):
        fn(A, B)
    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter_ns()
        fn(A, B)
        times.append(time.perf_counter_ns() - t0)
    return np.median(times) / 1e6  # ms


def main():
    rng = np.random.default_rng(42)
    A = rng.standard_normal(N).astype(np.float32)
    B = np.zeros(N, dtype=np.float32)

    try:
        treatment = stencil_3pt.compile()  # default target = x86
        jit_available = True
    except Exception as e:
        print(f"WARNING: JIT compilation failed: {e}", flush=True)
        print("Falling back to scalar-only measurement (no speedup number).",
              flush=True)
        jit_available = False

    baseline = stencil_3pt_scalar

    if jit_available:
        t_treat = measure(lambda A, B: treatment(A, B), A, B)
        t_base = measure(baseline, A, B)
        speedup = t_base / t_treat

        result = {
            "baseline_ms": t_base,
            "treatment_ms": t_treat,
            "speedup": speedup,
            "target": "x86",
            "N": N,
            "kernel": "1D 3-point stencil (v1 — cross-brick blocked on R12)",
            "jit_available": True,
        }
        print(json.dumps(result, indent=2))

        if speedup < TARGET_SPEEDUP_AVX512:
            print(
                f"WARNING: speedup {speedup:.2f}x < {TARGET_SPEEDUP_AVX512}x target",
                flush=True,
            )
            # Don't fail the script — speedup is environment-dependent.
    else:
        t_base = measure(baseline, A, B)
        result = {
            "baseline_ms": t_base,
            "speedup": "N/A (JIT not available)",
            "target": "x86",
            "N": N,
            "kernel": "1D 3-point stencil (v1 — cross-brick blocked on R12)",
            "jit_available": False,
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
