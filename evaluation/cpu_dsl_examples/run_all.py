#!/usr/bin/env python3
"""Run every cpu_dsl candidate and print a verification + timing summary.

Usage::

    source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
    cd evaluation/cpu_dsl_examples
    python run_all.py
    python run_all.py --measure-repeats 5
    python run_all.py --target=arm-neon

Each candidate is a single ``<short_name>.py`` using ``@benchmark`` from
``lego.testing``; its ``__main__`` block prints one JSON record per run
with timings and a ``verified: bool`` field.

Outputs:
    results.json     — one record per candidate
    dashboard.md     — human-readable table
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Python executable (prefer venv)
# ---------------------------------------------------------------------------
_VENV_PYTHON = Path("/scratch/general/vast/u1419116/LEGO/venv/bin/python")
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

_BUILD_LEGO = "/scratch/general/vast/u1419116/LEGO/build/python_packages/lego"
_existing_pp = os.environ.get("PYTHONPATH", "")
_filtered = ":".join(p for p in _existing_pp.split(":") if p and "LEGO/python" not in p)
_NEW_PP = _BUILD_LEGO + (":" + _filtered if _filtered else "")

_SUB_ENV = dict(os.environ)
_SUB_ENV["PYTHONPATH"] = _NEW_PP


def _sub_env_for_target(target: str) -> dict:
    e = dict(_SUB_ENV)
    e["LEGO_CPU_TARGET"] = target
    return e


# ---------------------------------------------------------------------------
# Kernel discovery + execution
# ---------------------------------------------------------------------------
def _discover_kernels():
    """Yield each numbered kernel .py directly under cpu_dsl_examples/."""
    return sorted(ROOT.glob("[0-9][0-9]_*.py"))


_TIMING_FIELDS = ("numpy_ms", "scalar_jit_ms", "vec_jit_ms")


def _run_measure_once(measure_py: Path, sub_env: dict, cand_name: str) -> dict:
    try:
        proc = subprocess.run(
            [PYTHON, str(measure_py)],
            capture_output=True, text=True, timeout=300,
            env=sub_env,
        )
        stdout = proc.stdout.strip()
        last_line = stdout.split("\n")[-1] if stdout else ""
        try:
            rec = json.loads(last_line)
        except json.JSONDecodeError:
            rec = {
                "name": cand_name,
                "verdict": "ERROR",
                "notes": (proc.stdout[-300:] + "\n" + proc.stderr[-300:]).strip(),
            }
        rec["name"] = cand_name
        return rec
    except subprocess.TimeoutExpired:
        return {"name": cand_name, "verdict": "ERROR", "notes": "timeout after 300s"}
    except Exception as exc:
        return {"name": cand_name, "verdict": "ERROR", "notes": str(exc)}


def run_measure(measure_py: Path, sub_env: dict, repeats: int = 1) -> dict:
    """Run the kernel K times and take the min of the timing fields.

    Min beats median on a shared node: every measurement is bounded above
    by hardware speed and pushed up by random preemption / cache eviction
    noise.  The min is the closest sample to true steady-state throughput.
    """
    cand_name = measure_py.stem
    runs = [_run_measure_once(measure_py, sub_env, cand_name)
            for _ in range(max(1, repeats))]
    base = runs[-1]
    if repeats <= 1 or len(runs) == 1:
        return base
    if base.get("verdict") == "ERROR":
        return base

    rec = dict(base)
    rec["measure_repeats"] = len(runs)
    for field in _TIMING_FIELDS:
        vals = [r.get(field) for r in runs
                if isinstance(r.get(field), (int, float))
                and not (isinstance(r.get(field), float) and math.isnan(r.get(field)))]
        if vals:
            rec[field] = round(min(vals), 6)

    t_scalar = rec.get("scalar_jit_ms", float('nan'))
    t_vec    = rec.get("vec_jit_ms",    float('nan'))
    t_numpy  = rec.get("numpy_ms",      float('nan'))
    if (isinstance(t_scalar, float) and isinstance(t_vec, float)
            and not math.isnan(t_scalar) and not math.isnan(t_vec) and t_vec > 0):
        rec["speedup_isolated_jit"] = round(t_scalar / t_vec, 4)
    if (isinstance(t_numpy, float) and isinstance(t_vec, float)
            and not math.isnan(t_numpy) and not math.isnan(t_vec) and t_vec > 0):
        rec["speedup_vs_numpy"] = round(t_numpy / t_vec, 4)
    return rec


# ---------------------------------------------------------------------------
# Dashboard formatting
# ---------------------------------------------------------------------------
def _fv(v, w, fmt=".4f"):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{w}{fmt}}"
    return f"{'NaN':>{w}}"


def _fx(v, w):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{w-1}.2f}x"
    return f"{'NaN':>{w}}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LEGO cpu_dsl candidate harness")
    p.add_argument("--target", default="x86",
                   help="cpu_dsl compilation target: x86 (default), arm-neon, scalar")
    p.add_argument("--measure-repeats", type=int, default=5,
                   help="Run each candidate N times and take the min "
                        "(default: 5).")
    return p.parse_args()


def main():
    args = parse_args()
    sub_env = _sub_env_for_target(args.target)

    print()
    print("=" * 100)
    print(f"  LEGO cpu_dsl_examples")
    print(f"  Target: {args.target}  |  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()

    kernels = _discover_kernels()

    results = []
    print(f"  Running {len(kernels)} kernels ...")
    for measure_py in kernels:
        print(f"    → {measure_py.stem}", flush=True)
        rec = run_measure(measure_py, sub_env, repeats=args.measure_repeats)
        rec["verify_status"] = "VERIFIED" if rec.get("verified") is True else "FAIL"
        results.append(rec)
    print()

    # Print dashboard
    _has_layout = any("layout_class" in r for r in results)

    print("=" * 110)
    print("  LEGO cpu_dsl_examples")
    print("=" * 110)

    hdr_parts = [f"{'Candidate':<34}"]
    if _has_layout:
        hdr_parts.append(f"{'LayoutClass':<18}")
    hdr_parts += [
        f"{'N':>8}", f"{'numpy_ms':>10}", f"{'scalar_ms':>10}",
        f"{'vec_ms':>10}", f"{'vec_iso':>8}", f"{'vs_numpy':>8}",
        f"{'Verify':>10}",
    ]
    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))

    n_verified = n_fail = 0
    vec_iso_gt15 = 0

    for r in results:
        name    = r.get("name", "?")[:34]
        layout  = r.get("layout_class", "")[:18]
        n_val   = r.get("N", "")
        n_str   = f"{n_val:>8}" if isinstance(n_val, int) else f"{'?':>8}"

        t_numpy  = r.get("numpy_ms",      float('nan'))
        t_scalar = r.get("scalar_jit_ms", float('nan'))
        t_vec    = r.get("vec_jit_ms",    float('nan'))
        sp_iso   = r.get("speedup_isolated_jit", float('nan'))
        sp_np    = r.get("speedup_vs_numpy",     float('nan'))

        verify_st = r.get("verify_status", "FAIL")
        if verify_st == "VERIFIED":
            n_verified += 1
            verify_short = "VERIFIED"
        else:
            n_fail += 1
            verify_short = "FAIL"

        if isinstance(sp_iso, float) and not math.isnan(sp_iso) and sp_iso > 1.5:
            vec_iso_gt15 += 1

        row_parts = [f"{name:<34}"]
        if _has_layout:
            row_parts.append(f"{layout:<18}")
        row_parts += [
            n_str,
            _fv(t_numpy,  10), _fv(t_scalar, 10), _fv(t_vec, 10),
            _fx(sp_iso, 8),    _fx(sp_np, 8),
            f"{verify_short:>10}",
        ]
        print("  ".join(row_parts))

    print("-" * len(hdr))
    print()
    total = len(results)
    print(f"  VERIFIED:           {n_verified} / {total}")
    print(f"  FAIL:               {n_fail} / {total}")
    print(f"  vec_iso > 1.5x:     {vec_iso_gt15} / {total}  (vs scalar_jit)")
    print()

    out_json = ROOT / "results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda x: None)
    print(f"  Full results saved to: {out_json}")

    out_md = ROOT / "dashboard.md"
    _write_markdown(results, out_md, args.target,
                    n_verified, n_fail, vec_iso_gt15)
    print(f"  Dashboard Markdown saved to: {out_md}")
    print()


def _write_markdown(results, out_path, target,
                    n_verified, n_fail, vec_iso_gt15):
    def _fmt(v):
        if isinstance(v, float) and not math.isnan(v):
            return f"{v:.4f}"
        return "NaN"

    def _fmtx(v):
        if isinstance(v, float) and not math.isnan(v):
            return f"{v:.2f}x"
        return "NaN"

    lines = [
        f"# LEGO cpu_dsl_examples",
        f"",
        f"**Target:** `{target}` | **Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total | {len(results)} |",
        f"| VERIFIED | {n_verified} |",
        f"| FAIL | {n_fail} |",
        f"| vec_iso > 1.5x | {vec_iso_gt15} |",
        f"",
        f"| Candidate | Layout | N | numpy_ms | scalar_ms | vec_ms | vec_iso | vs_numpy | Verify |",
        f"|-----------|--------|---|----------|-----------|--------|---------|----------|--------|",
    ]
    for r in results:
        name   = r.get("name", "?")[:32]
        layout = r.get("layout_class", "")[:18]
        n_val  = r.get("N", "?")
        t_np   = r.get("numpy_ms",      float('nan'))
        t_sc   = r.get("scalar_jit_ms", float('nan'))
        t_vec  = r.get("vec_jit_ms",    float('nan'))
        sp_iso = r.get("speedup_isolated_jit", float('nan'))
        sp_np  = r.get("speedup_vs_numpy",     float('nan'))
        verify = r.get("verify_status", "FAIL")
        verify_short = "OK" if verify == "VERIFIED" else "FAIL"

        lines.append(
            f"| {name} | {layout} | {n_val} | {_fmt(t_np)} | {_fmt(t_sc)} | "
            f"{_fmt(t_vec)} | {_fmtx(sp_iso)} | {_fmtx(sp_np)} | {verify_short} |"
        )

    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
