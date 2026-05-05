#!/usr/bin/env python3
"""Run every cpu_dsl_comparison benchmark and print a summary dashboard.

Usage (activate the venv first):
    source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
    cd evaluation/cpu_dsl_comparison
    python run_all.py                       # full run (build C baselines + measure)
    python run_all.py --quick               # skip C baseline build (use cached binaries)
    python run_all.py --measure-repeats 5   # median of 5 measurements per candidate
    python run_all.py --target=arm-neon     # cpu_dsl compiled with ARM NEON target

Each candidate is a single ``<short_name>.py`` using ``@benchmark`` from
``lego.testing``; its ``__main__`` block prints one JSON record per run with
timings and a ``verified: bool`` field. See ``candidates/01_saxpy/saxpy.py``
for the canonical template.

Dashboard columns:
    Candidate  LayoutClass  PriorVerd  N  numpy_ms  scalar_jit  vec_jit
    vec_iso  vs_numpy  c_O3_ms  vs_c_O3  c_agg_ms  vs_c_agg
    [c_gcc_ms  vs_gcc]  [c_clang_ms  vs_clang]
    Verify  Verdict

Primary verdict basis: vs_c_O3 = c_O3_ms / vec_jit_ms  (CASTLE-aligned).
  WIN    if vs_c_O3 > 1.05×
  PARITY if vs_c_O3 >= 0.95×
  LOSS   otherwise

Dashboard JSON saved to results.json alongside this script.
Dashboard Markdown saved to dashboard.md.
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
CANDIDATES_DIR = ROOT / "candidates"
C_BASELINES_DIR = ROOT / "c_baselines"

# ---------------------------------------------------------------------------
# C baseline map: candidate name → (binary_name, N_to_pass)
#
# Binary naming convention:
#   {bin}       — legacy GCC aggressive (backward compat, candidates 01-08)
#   {bin}_O3    — GCC -O3 only (CASTLE-aligned, no march/fast-math)
#   {bin}_agg   — GCC -O3 -march=native -mavx512f -ffast-math (aggressive)
#   {bin}_clang — Clang aggressive (candidates 01-08 only)
#
# Baseline kernel groupings:
#   fma_1M              : unit-stride FMA loop, N=1M  (candidates 12-17, 28, 32-33, 39-42)
#   fma_64k             : unit-stride FMA loop, N=64K (kept for reference; not used for 34-35)
#   morton_fma_64k      : Z-Morton gather FMA, N=64K (candidates 09-11, 34-35, 36)
#   stride2_16k         : stride-2 deinterleave, N=16K (candidates 23-25, 38) or N=4096 (26-27)
#   stride4_16k         : stride-4 deinterleave, N=16K (candidates 29-31)
#   stencil_3d7pt       : 3D 7-point stencil, 32x32x32 (candidates 19, 21)
#   stencil_3d13pt      : 3D 13-point stencil, 32x32x32 (candidate 20)
#   stencil_2d5pt       : 2D 5-point jacobi stencil, 256x256 flat (candidate 22)
#   stencil_2d5pt_30x30 : 2D 5-point jacobi stencil, 30x30 flat, N_INNER=840 (candidate 37)
# ---------------------------------------------------------------------------
_C_BASELINE_MAP = {
    # Curated 10-kernel set, one representative per distinct lowering path.
    # Each candidate's N matches its C source's DEFAULT_N, so the binaries
    # are invoked without an argv N (None) and pick up DEFAULT_N internally.
    # That keeps the _clang_const variant valid (BENCH_ASSUME_N(N) holds
    # because N == DEFAULT_N at runtime).
    "01_saxpy":               ("saxpy",              None),  # unit-stride pointwise
    "02_gemm_row_major":      ("gemm",               None),  # reduction loop
    "03_3pt_stencil_1d":      ("stencil_3pt",        None),  # constant-offset gather
    "05_morton_2d":           ("morton",             None),  # bit-permuted index
    "07_mixed_precision":     ("mixed_precision",    None),  # mixed dtypes
    "08_brick_within_cell":   ("brick_within_cell",  None),  # 3D brick
    "43_spmv_indirect":       ("spmv_indirect",      None),  # data-dependent index
    "44_predicated_fma":      ("predicated_fma",     None),  # predicated update
    "46_scatter_compute":     ("scatter_compute",    None),  # scatter
    "47_multi_reduce":        ("multi_reduce",       None),  # multi-output reduction
}


def _run_c_binary_once(bin_path: Path, N) -> float:
    """Run a C baseline binary once, return ms_per_call or NaN."""
    if not bin_path.exists():
        return float('nan')
    cmd = [str(bin_path)]
    if N is not None:
        cmd.append(str(N))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stdout = proc.stdout.strip()
        last_line = stdout.split("\n")[-1] if stdout else ""
        data = json.loads(last_line)
        return float(data.get("c_baseline_ms_per_call", math.nan))
    except Exception:
        return float('nan')


def _run_c_binary(bin_path: Path, N, repeats: int = 1) -> float:
    """Run a C baseline `repeats` times, return min ms_per_call.

    Same min-of-N rationale as ``run_measure``: we want the steady-state
    best-case the C compiler produces, free of preemption noise.
    """
    if repeats <= 1:
        return _run_c_binary_once(bin_path, N)
    vals = []
    for _ in range(repeats):
        v = _run_c_binary_once(bin_path, N)
        if isinstance(v, float) and not math.isnan(v):
            vals.append(v)
    return min(vals) if vals else float('nan')


def run_c_O3_baseline(cand_name: str, repeats: int = 1) -> float:
    """Run GCC ``-O3`` baseline — conservative reference."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_O3", N, repeats)


def run_clang_baseline(cand_name: str, repeats: int = 1) -> float:
    """Run clang aggressive baseline (-O3 -march=native -mavx512f -ffast-math)."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_clang", N, repeats)


def run_clang_const_baseline(cand_name: str, repeats: int = 1) -> float:
    """Run clang -O3 -march=native -mavx512f + ``__builtin_assume(N == DEFAULT_N)``.

    Apples-to-apples baseline against LEGO's JIT.  LEGO bakes each
    candidate's N as a Python compile-time literal, so clang needs the
    same compile-time-N visibility to do the same algebraic folds.  Each
    .c source defines a DEFAULT_N matching its candidate's N; the binary
    is invoked with no argv and the assume holds.
    """
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_clang_const", N, repeats)


# ---------------------------------------------------------------------------
# Python executable (prefer venv)
# ---------------------------------------------------------------------------
_VENV_PYTHON = Path("/scratch/general/vast/u1419116/LEGO/venv/bin/python")
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# PYTHONPATH: use the build directory
_BUILD_LEGO = "/scratch/general/vast/u1419116/LEGO/build/python_packages/lego"
_existing_pp = os.environ.get("PYTHONPATH", "")
_filtered = ":".join(p for p in _existing_pp.split(":") if p and "LEGO/python" not in p)
_NEW_PP = _BUILD_LEGO + (":" + _filtered if _filtered else "")

_SUB_ENV = dict(os.environ)
_SUB_ENV["PYTHONPATH"] = _NEW_PP


def _sub_env_for_target(target: str) -> dict:
    """Return environ dict with LEGO_CPU_TARGET set."""
    e = dict(_SUB_ENV)
    e["LEGO_CPU_TARGET"] = target
    return e


# ---------------------------------------------------------------------------
# Build C baselines
# ---------------------------------------------------------------------------
def build_c_baselines(verbose: bool = False):
    makefile = C_BASELINES_DIR / "Makefile"
    if not makefile.exists():
        print(f"  [build] No Makefile found in {C_BASELINES_DIR}, skipping C baseline build.")
        return
    print(f"  [build] Building C baselines in {C_BASELINES_DIR} ...")
    cmd = ["make", "-C", str(C_BASELINES_DIR), "all"]
    proc = subprocess.run(cmd, capture_output=not verbose, text=True)
    if proc.returncode == 0:
        print("  [build] C baselines built successfully.")
    else:
        print(f"  [build] WARNING: C baseline build returned {proc.returncode}.")
        if not verbose and proc.stderr:
            print(proc.stderr[-400:])


# ---------------------------------------------------------------------------
# Candidate format detection
# ---------------------------------------------------------------------------
def _find_consolidated_kernel(cand_dir: Path) -> Path | None:
    """Return the candidate's single consolidated kernel file.

    Each candidate directory contains exactly one ``<short_name>.py`` using
    ``@benchmark`` from ``lego.testing``. The script's ``__main__`` prints a
    JSON record with timing + a ``verified: bool`` field — see saxpy.py.
    """
    for py in sorted(cand_dir.glob("*.py")):
        if py.name != "__init__.py":
            return py
    return None


# ---------------------------------------------------------------------------
# Run measure.py for a single candidate
# ---------------------------------------------------------------------------
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
        # Force the candidate-directory name as the canonical identifier.
        rec["name"] = cand_name
        return rec
    except subprocess.TimeoutExpired:
        return {"name": cand_name, "verdict": "ERROR", "notes": "timeout after 300s"}
    except Exception as exc:
        return {"name": cand_name, "verdict": "ERROR", "notes": str(exc)}


def run_measure(cand_dir: Path, sub_env: dict, repeats: int = 1) -> dict:
    """Run the candidate's consolidated ``@benchmark`` kernel.

    The script's ``__main__`` block prints one JSON record per invocation
    with timings + a ``verified`` field. With ``repeats > 1`` the candidate
    is run K times and the timing fields take the **min** across runs.

    Why min, not median: on a shared node, every measurement is bounded
    above by hardware speed and pushed up by random external interference
    (CPU steal, interrupts, neighbor processes, page-placement luck).
    The min is the closest sample to true hardware throughput; everything
    above it is noise. With only N=5 samples the median can sit in the
    middle of a noisy distribution and flip WIN/PARITY across runs even
    on identical compiled code. This is standard practice for steady-state
    CPU microbenchmarks (Google Benchmark, Criterion, hyperfine).
    """
    measure_py = _find_consolidated_kernel(cand_dir)
    if measure_py is None:
        return {"name": cand_dir.name, "verdict": "ERROR", "notes": "no candidate .py file"}

    runs = [_run_measure_once(measure_py, sub_env, cand_dir.name)
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
# Dashboard formatting helpers
# ---------------------------------------------------------------------------
def _fv(v, w, fmt=".3f"):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{w}{fmt}}"
    return f"{'NaN':>{w}}"


def _fx(v, w):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{w-1}.2f}x"
    return f"{'NaN':>{w}}"


def _safe_ratio(a, b):
    if (isinstance(a, float) and not math.isnan(a) and
            isinstance(b, float) and not math.isnan(b) and b > 0):
        return a / b
    return float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LEGO cpu_dsl_comparison benchmark harness")
    p.add_argument("--quick", action="store_true",
                   help="Skip building C baselines (faster iteration)")
    p.add_argument("--target", default="x86",
                   help="cpu_dsl compilation target: x86 (default), arm-neon, scalar")
    p.add_argument("--verbose", action="store_true",
                   help="Show C baseline build output")
    p.add_argument("--measure-repeats", type=int, default=5,
                   help="Run each candidate's measure.py N times and median "
                        "the timing fields. Stabilizes verdicts at the "
                        "WIN/PARITY boundary (1.05x) where one cold-process "
                        "outlier in a 3-run median can flip classification. "
                        "Default: 5 (median of 5 tolerates one outlier).")
    return p.parse_args()


def main():
    args = parse_args()
    sub_env = _sub_env_for_target(args.target)

    print()
    print("=" * 100)
    print(f"  LEGO cpu_dsl_comparison — Full Coverage Benchmark Harness")
    print(f"  Target: {args.target}  |  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()

    # Step 1: Build C baselines
    if not args.quick:
        build_c_baselines(args.verbose)
    else:
        print("  [build] Skipping C baseline build (--quick).")
    print()

    cand_dirs = sorted(
        d for d in CANDIDATES_DIR.iterdir()
        if d.is_dir() and _find_consolidated_kernel(d) is not None
    )

    # Step 2: Time + verify each candidate. With the @benchmark format, the
    # consolidated script's ``__main__`` reports both timing and a
    # ``verified`` field in one JSON record — no separate verify pass needed.
    results = []
    print(f"  [measure] Running {len(cand_dirs)} candidates ...")
    for cand_dir in cand_dirs:
        print(f"    → {cand_dir.name}", flush=True)
        c_O3_ms        = run_c_O3_baseline(cand_dir.name, repeats=args.measure_repeats)
        clang_ms       = run_clang_baseline(cand_dir.name, repeats=args.measure_repeats)
        clang_const_ms = run_clang_const_baseline(cand_dir.name, repeats=args.measure_repeats)
        rec = run_measure(cand_dir, sub_env, repeats=args.measure_repeats)

        # Attach C baseline timings
        if not math.isnan(c_O3_ms):
            rec["c_O3_ms"] = round(c_O3_ms, 6)
        if not math.isnan(clang_ms):
            rec["c_clang_ms"] = round(clang_ms, 6)
        if not math.isnan(clang_const_ms):
            rec["c_clang_const_ms"] = round(clang_const_ms, 6)

        rec["verify_status"] = "VERIFIED" if rec.get("verified") is True else "FAIL"

        results.append(rec)
    print()

    # Step 4: Print dashboard
    _has_c_O3        = any("c_O3_ms"  in r for r in results)
    _has_clang       = any("c_clang_ms" in r for r in results)
    _has_clang_const = any("c_clang_const_ms" in r for r in results)
    _has_prior       = any("prior_verdict" in r for r in results)
    _has_layout      = any("layout_class" in r for r in results)
    _has_verify      = True

    print("=" * 180)
    print("  LEGO cpu_dsl_comparison")
    print("  Verdict (vs gcc -O3): WIN >1.05x  |  PARITY 0.95-1.05x  |  LOSS <0.95x")
    print("=" * 180)

    # Header
    hdr_parts = [f"{'Candidate':<34}"]
    if _has_layout:
        hdr_parts.append(f"{'LayoutClass':<18}")
    hdr_parts += [
        f"{'N':>8}", f"{'numpy_ms':>10}", f"{'scalar_ms':>10}",
        f"{'vec_ms':>10}", f"{'vec_iso':>8}", f"{'vs_numpy':>8}",
    ]
    if _has_c_O3:
        hdr_parts += [f"{'c_O3_ms':>9}", f"{'vs_c_O3':>8}"]
    if _has_clang:
        hdr_parts += [f"{'c_clang_ms':>11}", f"{'vs_clang':>9}"]
    if _has_clang_const:
        hdr_parts += [f"{'c_const_ms':>11}", f"{'vs_const':>9}"]
    if _has_verify:
        hdr_parts.append(f"{'Verify':>10}")
    hdr_parts.append(f"{'Verdict':>8}")
    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))

    wins = losses = parities = skips = errors = 0
    wins_vs_clang = losses_vs_clang = parities_vs_clang = 0
    wins_vs_const = losses_vs_const = parities_vs_const = 0
    vec_iso_gt15 = 0
    n_verified = n_pending = n_fail_v = 0

    for r in results:
        name    = r.get("name", "?")[:34]
        layout  = r.get("layout_class", "")[:18]
        n_val   = r.get("N", "")
        n_str   = f"{n_val:>8}" if isinstance(n_val, int) else f"{'?':>8}"

        c_O3_ms_r      = float(r.get("c_O3_ms",         float('nan')))
        clang_ms_r     = float(r.get("c_clang_ms",      float('nan')))
        clang_cst_ms_r = float(r.get("c_clang_const_ms", float('nan')))

        t_numpy  = r.get("numpy_ms",  float('nan'))
        t_scalar = r.get("scalar_jit_ms", float('nan'))
        t_vec    = r.get("vec_jit_ms", float('nan'))
        sp_iso   = r.get("speedup_isolated_jit", float('nan'))
        sp_np    = r.get("speedup_vs_numpy", float('nan'))

        vs_c_O3        = _safe_ratio(c_O3_ms_r,      t_vec)
        vs_clang       = _safe_ratio(clang_ms_r,     t_vec)
        vs_clang_const = _safe_ratio(clang_cst_ms_r, t_vec)

        # Primary verdict: based on vs_c_O3 (CASTLE-aligned).
        # Fall back to measure.py's own verdict if c_O3_ms is unavailable.
        measure_verdict = r.get("verdict", "ERROR")
        if not math.isnan(vs_c_O3):
            if vs_c_O3 > 1.05:
                verdict = "WIN"
            elif vs_c_O3 >= 0.95:
                verdict = "PARITY"
            else:
                verdict = "LOSS"
        else:
            verdict = measure_verdict   # fall back

        verify_st = r.get("verify_status", "NOT_RUN")
        if verify_st == "VERIFIED":
            n_verified += 1
            verify_short = "VERIFIED"
        elif "PENDING" in verify_st:
            n_pending += 1
            verify_short = "PENDING"
        elif verify_st == "NOT_RUN":
            verify_short = "-"
        else:
            n_fail_v += 1
            verify_short = "FAIL"

        # Tally primary (vs_c_O3)
        if verdict == "WIN":      wins += 1
        elif verdict == "LOSS":   losses += 1
        elif verdict == "PARITY": parities += 1
        elif verdict == "SKIP":   skips += 1
        else:                     errors += 1

        if not math.isnan(vs_clang):
            if vs_clang > 1.05:   wins_vs_clang += 1
            elif vs_clang >= 0.95: parities_vs_clang += 1
            else:                  losses_vs_clang += 1

        if not math.isnan(vs_clang_const):
            if vs_clang_const > 1.05:    wins_vs_const += 1
            elif vs_clang_const >= 0.95: parities_vs_const += 1
            else:                         losses_vs_const += 1

        if isinstance(sp_iso, float) and not math.isnan(sp_iso) and sp_iso > 1.5:
            vec_iso_gt15 += 1

        row_parts = [f"{name:<34}"]
        if _has_layout:
            row_parts.append(f"{layout:<18}")
        row_parts += [
            n_str,
            _fv(t_numpy,  10, '.4f'),
            _fv(t_scalar, 10, '.4f'),
            _fv(t_vec,    10, '.4f'),
            _fx(sp_iso, 8),
            _fx(sp_np,  8),
        ]
        if _has_c_O3:
            row_parts += [_fv(c_O3_ms_r, 9, '.4f'), _fx(vs_c_O3, 8)]
        if _has_clang:
            row_parts += [_fv(clang_ms_r, 11, '.4f'), _fx(vs_clang, 9)]
        if _has_clang_const:
            row_parts += [_fv(clang_cst_ms_r, 11, '.4f'), _fx(vs_clang_const, 9)]
        if _has_verify:
            row_parts.append(f"{verify_short:>10}")
        row_parts.append(f"{verdict:>8}")
        print("  ".join(row_parts))

    print("-" * len(hdr))
    print()

    # Summary statistics
    total    = len(results)
    measured = total - skips
    print(f"  Total candidates:         {total}")
    print(f"  Measured (non-SKIP):      {measured}")
    print(f"  SKIP (XFAIL/unbundled):   {skips}")
    print(f"  Errors:                   {errors}")
    print()
    print(f"  --- Correctness (verify.py) ---")
    print(f"  VERIFIED:   {n_verified:3d} / {total}")
    print(f"  PENDING:    {n_pending:3d} / {total}")
    print(f"  FAIL/MISS:  {n_fail_v:3d} / {total}")
    print()
    print(f"  --- Verdicts vs gcc -O3 (conservative reference) ---")
    print(f"  WIN:     {wins:3d} / {measured}")
    print(f"  PARITY:  {parities:3d} / {measured}")
    print(f"  LOSS:    {losses:3d} / {measured}")
    print()
    print(f"  --- Verdicts vs clang aggressive (clang-20 -O3 -march=native -mavx512f -ffast-math) ---")
    print(f"  WIN:     {wins_vs_clang:3d} / {measured}")
    print(f"  PARITY:  {parities_vs_clang:3d} / {measured}")
    print(f"  LOSS:    {losses_vs_clang:3d} / {measured}")
    print()
    n_const = wins_vs_const + parities_vs_const + losses_vs_const
    print(f"  --- Verdicts vs clang CONST  (apples-to-apples: same flags + __builtin_assume(N == DEFAULT_N)) ---")
    print(f"  WIN:     {wins_vs_const:3d} / {n_const}")
    print(f"  PARITY:  {parities_vs_const:3d} / {n_const}")
    print(f"  LOSS:    {losses_vs_const:3d} / {n_const}")
    print(f"  (Note: only candidates whose N matches each source's DEFAULT_N have a const baseline.)")
    print()
    print(f"  vec_iso > 1.5× (vs scalar_jit): {vec_iso_gt15:3d} / {measured}")
    print()

    # Step 5: Save JSON + Markdown
    out_json = ROOT / "results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=lambda x: None)
    print(f"  Full results saved to: {out_json}")

    out_md = ROOT / "dashboard.md"
    _write_markdown(results, out_md, args.target,
                    wins, parities, losses, skips, errors,
                    n_verified, n_pending, n_fail_v,
                    measured, vec_iso_gt15,
                    wins_vs_clang, parities_vs_clang, losses_vs_clang,
                    wins_vs_const, parities_vs_const, losses_vs_const)
    print(f"  Dashboard Markdown saved to: {out_md}")
    print()


def _write_markdown(results, out_path: Path, target: str,
                    wins, parities, losses, skips, errors,
                    n_verified, n_pending, n_fail_v,
                    measured, vec_iso_gt15,
                    wins_vs_clang=0, parities_vs_clang=0, losses_vs_clang=0,
                    wins_vs_const=0, parities_vs_const=0, losses_vs_const=0):
    """Write a compact Markdown dashboard."""

    def _fmt(v):
        if isinstance(v, float) and not math.isnan(v):
            return f"{v:.4f}"
        return "NaN"

    def _fmtx(v):
        if isinstance(v, float) and not math.isnan(v):
            return f"{v:.2f}x"
        return "NaN"

    lines = [
        f"# LEGO cpu_dsl_comparison Dashboard",
        f"",
        f"**Target:** `{target}` | **Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total candidates | {len(results)} |",
        f"| Measured | {measured} |",
        f"| VERIFIED (correctness) | {n_verified} / {len(results)} |",
        f"| WIN / PARITY / LOSS vs gcc -O3 | {wins} / {parities} / {losses} |",
        f"| WIN / PARITY / LOSS vs clang aggressive (-ffast-math) | {wins_vs_clang} / {parities_vs_clang} / {losses_vs_clang} |",
        f"| WIN / PARITY / LOSS vs clang CONST | {wins_vs_const} / {parities_vs_const} / {losses_vs_const} |",
        f"| vec_iso > 1.5x | {vec_iso_gt15} |",
        f"",
        f"`clang_const` adds `__builtin_assume(N == DEFAULT_N)` so clang sees N as a",
        f"compile-time constant — matches LEGO's frontend, which bakes N as a Python",
        f"literal.",
        f"",
        f"## Per-Candidate Results",
        f"",
        f"| Candidate | Layout | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_clang_ms | vs_clang | c_const_ms | vs_const | Verify | Verdict |",
        f"|-----------|--------|---|----------|-----------|--------|---------|---------|---------|------------|----------|------------|----------|--------|---------|",
    ]
    for r in results:
        name   = r.get("name", "?")[:32]
        layout = r.get("layout_class", "")[:15]
        n_val  = r.get("N", "?")
        t_np   = r.get("numpy_ms",      float('nan'))
        t_sc   = r.get("scalar_jit_ms", float('nan'))
        t_vec  = r.get("vec_jit_ms",    float('nan'))
        sp_iso = r.get("speedup_isolated_jit", float('nan'))
        c_O3   = float(r.get("c_O3_ms",         float('nan')))
        c_cl   = float(r.get("c_clang_ms",      float('nan')))
        c_cst  = float(r.get("c_clang_const_ms", float('nan')))
        vs_O3  = c_O3  / t_vec if (not math.isnan(c_O3)  and not math.isnan(t_vec) and t_vec > 0) else float('nan')
        vs_cl  = c_cl  / t_vec if (not math.isnan(c_cl)  and not math.isnan(t_vec) and t_vec > 0) else float('nan')
        vs_cst = c_cst / t_vec if (not math.isnan(c_cst) and not math.isnan(t_vec) and t_vec > 0) else float('nan')
        verify = r.get("verify_status", "-")
        verify_short = "OK" if verify == "VERIFIED" else ("?" if "PENDING" in verify else ("–" if verify == "NOT_RUN" else "FAIL"))

        if not math.isnan(vs_O3):
            if vs_O3 > 1.05:   md_verdict = "**WIN**"
            elif vs_O3 >= 0.95: md_verdict = "PARITY"
            else:               md_verdict = "LOSS"
        else:
            md_verdict = r.get("verdict", "?")

        lines.append(
            f"| {name} | {layout} | {n_val} | {_fmt(t_np)} | {_fmt(t_sc)} | {_fmt(t_vec)} | {_fmtx(sp_iso)} | "
            f"{_fmt(c_O3)} | {_fmtx(vs_O3)} | {_fmt(c_cl)} | {_fmtx(vs_cl)} | {_fmt(c_cst)} | {_fmtx(vs_cst)} | "
            f"{verify_short} | {md_verdict} |"
        )

    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
