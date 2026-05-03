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
    # ---- Original 8 candidates (have all three C variants) ----
    "01_saxpy":               ("saxpy",              1 << 20),
    "02_gemm_row_major":      ("gemm",               64),
    "03_3pt_stencil_1d":      ("stencil_3pt",        1024),
    "04_col_major_inner":     ("col_major",           256),
    "05_morton_2d":           ("morton",              1 << 16),
    "06_self_update":         ("self_update",         4096),
    "07_mixed_precision":     ("mixed_precision",     1 << 20),
    "08_brick_within_cell":   ("brick_within_cell",   1 << 20),
    # ---- Z-Morton gather FMA, N=64K ----
    "09_gemm_zmorton":           ("morton_fma_64k",    None),
    "10_lu_zmorton":             ("morton_fma_64k",    None),
    "11_chol_zmorton":           ("morton_fma_64k",    None),
    # ---- Tiled 2D GEMM (N=512×512) — compare against naive 3-loop gemm.c ----
    "12_gemm_reg_L1_L2_tile":    ("gemm",              512),
    "13_3mm_reg_L1_L2_tile":     ("fma_1M",            None),
    "14_2mm_reg_L1_tile":        ("fma_1M",            None),
    "15_trmm_L1_L2_tile":        ("fma_1M",            None),
    "16_doitgen_reg_L1_tile":    ("fma_1M",            None),
    "17_tensor_contraction_gett":("fma_1M",            None),
    "18_tblis_notranspose":      ("gemm",               64),
    # ---- 3D stencil, 32x32x32 interior ----
    "19_bricklib_3d7pt":         ("stencil_3d7pt",     None),
    # 20 is a 3D 13-point stencil — use the correct 13pt C baseline, not the 7pt one.
    # gcc -O3 cannot auto-vectorize the 13pt stencil (too many strided gathers),
    # so LEGO's vectorized output wins significantly vs the scalar C reference.
    "20_bricklib_3d13pt":        ("stencil_3d13pt",    None),
    "21_heat3d_brick":           ("stencil_3d7pt",     None),
    # 22 is a 2D jacobi stencil (256x256 flat), not 3D — use the correct
    # stencil_2d5pt baseline that matches the kernel's flat-loop structure.
    "22_jacobi2d_brick":         ("stencil_2d5pt",     None),
    # ---- Stride-2 deinterleave, N=16K ----
    "23_symm_rfp":               ("stride2_16k",       None),
    "24_syrk_rfp":               ("stride2_16k",       None),
    "25_nw_antidiag":            ("stride2_16k",       None),
    # ---- Skew tile, stride-2 gather at N=4096 ----
    # kernel.py and measure.py both use stride-2 at N=4096; pass N=4096 to C baseline.
    "26_nussinov_skew":          ("stride2_16k",        4096),
    "27_zuker_skew":             ("stride2_16k",        4096),
    "28_seidel2d_wavefront":     ("fma_1M",            None),
    # ---- AoSoA stride-4, N=16K ----
    "29_particlefilter_aosoA":   ("stride4_16k",       None),
    "30_lulesh_aosoA":           ("stride4_16k",       None),
    "31_hpccg_aosoA":            ("stride4_16k",       None),
    # ---- Block-cyclic / tiled, unit-stride N=1M ----
    "32_fdtd2d_block_cyclic":    ("fma_1M",            None),
    "33_adi_block_cyclic":       ("fma_1M",            None),
    # ---- Pow2-padded Morton gather, N=64K ----
    # These kernels apply the same Z-Morton bit-interleaving index as candidates
    # 09-11 and 36 (ti = i & 0x5555; tj = (i>>1) & 0x5555; morton = ti|(tj<<1)).
    # Although that particular bitmask produces the identity permutation for
    # N=65536, gcc -O3 cannot prove this and emits scalar code (same as
    # morton_fma_64k.c).  The fair baseline is morton_fma_64k, not fma_64k
    # (which is unit-stride and auto-vectorized by gcc to ~4× faster).
    "34_gemm_pow2_pad":          ("morton_fma_64k",    None),
    "35_heat3d_pow2_pad":        ("morton_fma_64k",    None),
    # ---- Non-pow2 Morton gather, N=64K ----
    "36_gemm_nonpow2_morton":    ("morton_fma_64k",    None),
    # ---- Non-pow2 2D stencil (28x30=840 interior) ----
    # Use the matching 30x30 baseline (N_INNER=840), not the 32x32x32 3D baseline (N_INNER=30720).
    "37_stencil_nonpow2_brick":  ("stencil_2d5pt_30x30", None),
    # ---- Non-pow2 skew, stride-2, N=16K ----
    "38_nussinov_nonpow2_skew":  ("stride2_16k",       None),
    # ---- Tiled, unit-stride N=1M ----
    "39_hotspot_tile":           ("fma_1M",            None),
    "40_mvt_L1_tile":            ("fma_1M",            None),
    "41_bicg_L1_tile":           ("fma_1M",            None),
    "42_dgemm_reg_L1_L2_tile":   ("fma_1M",            None),
}


def _run_c_binary(bin_path: Path, N) -> float:
    """Run a C baseline binary, return ms_per_call or NaN."""
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


def run_c_baseline(cand_name: str) -> float:
    """Run legacy GCC aggressive binary (backward compat, candidates 01-08)."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / bin_name, N)


def run_c_O3_baseline(cand_name: str) -> float:
    """Run GCC -O3 only baseline (CASTLE-aligned, primary verdict basis)."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_O3", N)


def run_c_agg_baseline(cand_name: str) -> float:
    """Run GCC -O3 -march=native -mavx512f -ffast-math baseline (aggressive)."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_agg", N)


def run_clang_baseline(cand_name: str) -> float:
    """Run Clang aggressive baseline (candidates 01-08 only)."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_clang", N)


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
    is run K times and the timing fields (``numpy_ms``, ``scalar_jit_ms``,
    ``vec_jit_ms``) are medianed — defends against transient interference
    on shared nodes where one bad sample can flip a WIN/LOSS verdict.
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
            rec[field] = round(_median(vals), 6)

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


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float('nan')
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


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
    p.add_argument("--measure-repeats", type=int, default=3,
                   help="Run each candidate's measure.py N times and median "
                        "the timing fields. Stabilizes verdicts for tiny "
                        "kernels with bimodal page-placement noise. Default: 3.")
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
        c_ms      = run_c_baseline(cand_dir.name)   # legacy (01-08 only)
        c_O3_ms   = run_c_O3_baseline(cand_dir.name)
        c_agg_ms  = run_c_agg_baseline(cand_dir.name)
        clang_ms  = run_clang_baseline(cand_dir.name)
        rec = run_measure(cand_dir, sub_env, repeats=args.measure_repeats)

        # Attach C baseline timings
        if not math.isnan(c_ms):
            rec["c_baseline_ms"] = round(c_ms, 6)
        if not math.isnan(c_O3_ms):
            rec["c_O3_ms"] = round(c_O3_ms, 6)
        if not math.isnan(c_agg_ms):
            rec["c_agg_ms"] = round(c_agg_ms, 6)
        if not math.isnan(clang_ms):
            rec["c_clang_ms"] = round(clang_ms, 6)

        rec["verify_status"] = "VERIFIED" if rec.get("verified") is True else "FAIL"

        results.append(rec)
    print()

    # Step 4: Print dashboard
    _has_c = any("c_baseline_ms" in r for r in results)
    _has_c_O3   = any("c_O3_ms"  in r for r in results)
    _has_c_agg  = any("c_agg_ms" in r for r in results)
    _has_clang  = any("c_clang_ms" in r for r in results)
    _has_prior  = any("prior_verdict" in r for r in results)
    _has_layout = any("layout_class" in r for r in results)
    _has_verify = True

    print("=" * 220)
    print("  LEGO cpu_dsl_comparison — Full Coverage Dashboard (dual C baselines)")
    print("  Verdict basis: vs_c_O3 = c_O3_ms / vec_jit_ms")
    print("    WIN    if vs_c_O3 > 1.05×  |  PARITY if >= 0.95×  |  LOSS otherwise")
    print("=" * 220)

    # Header
    hdr_parts = [f"{'Candidate':<34}"]
    if _has_layout:
        hdr_parts.append(f"{'LayoutClass':<18}")
    if _has_prior:
        hdr_parts.append(f"{'PriorVerd':>9}")
    hdr_parts += [
        f"{'N':>8}", f"{'numpy_ms':>10}", f"{'scalar_ms':>10}",
        f"{'vec_ms':>10}", f"{'vec_iso':>8}", f"{'vs_numpy':>8}",
    ]
    if _has_c_O3:
        hdr_parts += [f"{'c_O3_ms':>9}", f"{'vs_c_O3':>8}"]
    if _has_c_agg:
        hdr_parts += [f"{'c_agg_ms':>9}", f"{'vs_c_agg':>9}"]
    if _has_clang:
        hdr_parts += [f"{'c_clang_ms':>11}", f"{'vs_clang':>9}"]
    if _has_verify:
        hdr_parts.append(f"{'Verify':>10}")
    hdr_parts.append(f"{'Verdict':>8}")
    hdr = "  ".join(hdr_parts)
    print(hdr)
    print("-" * len(hdr))

    wins = losses = parities = skips = errors = 0
    wins_vs_agg = losses_vs_agg = parities_vs_agg = 0
    improved = maintained = regressed = 0
    vec_iso_gt15 = 0
    n_verified = n_pending = n_fail_v = 0

    for r in results:
        name    = r.get("name", "?")[:34]
        layout  = r.get("layout_class", "")[:18]
        prior   = r.get("prior_verdict", "")
        n_val   = r.get("N", "")
        n_str   = f"{n_val:>8}" if isinstance(n_val, int) else f"{'?':>8}"

        c_ms_r     = float(r.get("c_baseline_ms", float('nan')))
        c_O3_ms_r  = float(r.get("c_O3_ms",       float('nan')))
        c_agg_ms_r = float(r.get("c_agg_ms",      float('nan')))
        clang_ms_r = float(r.get("c_clang_ms",    float('nan')))

        t_numpy  = r.get("numpy_ms",  float('nan'))
        t_scalar = r.get("scalar_jit_ms", float('nan'))
        t_vec    = r.get("vec_jit_ms", float('nan'))
        sp_iso   = r.get("speedup_isolated_jit", float('nan'))
        sp_np    = r.get("speedup_vs_numpy", float('nan'))

        vs_c_O3  = _safe_ratio(c_O3_ms_r,  t_vec)
        vs_c_agg = _safe_ratio(c_agg_ms_r, t_vec)
        vs_clang = _safe_ratio(clang_ms_r, t_vec)

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

        # Tally vs_c_agg (secondary, optional)
        if not math.isnan(vs_c_agg):
            if vs_c_agg > 1.05:   wins_vs_agg += 1
            elif vs_c_agg >= 0.95: parities_vs_agg += 1
            else:                   losses_vs_agg += 1

        if isinstance(sp_iso, float) and not math.isnan(sp_iso) and sp_iso > 1.5:
            vec_iso_gt15 += 1

        if prior in ("WIN", "PARITY", "LOSS", "MIXED"):
            new_win  = verdict in ("WIN",)
            new_skip = verdict in ("SKIP",)
            if not new_skip:
                if prior == "WIN" and new_win:           maintained += 1
                elif prior != "WIN" and new_win:         improved += 1
                elif prior == "WIN" and not new_win:     regressed += 1
                else:                                    maintained += 1

        row_parts = [f"{name:<34}"]
        if _has_layout:
            row_parts.append(f"{layout:<18}")
        if _has_prior:
            row_parts.append(f"{prior:>9}")
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
        if _has_c_agg:
            row_parts += [_fv(c_agg_ms_r, 9, '.4f'), _fx(vs_c_agg, 9)]
        if _has_clang:
            row_parts += [_fv(clang_ms_r, 11, '.4f'), _fx(vs_clang, 9)]
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
    print(f"  PENDING:    {n_pending:3d} / {total}  (known infra gaps, see R19)")
    print(f"  FAIL/MISS:  {n_fail_v:3d} / {total}")
    print()
    print(f"  --- Verdicts vs C O3 only (CASTLE-aligned, primary) ---")
    print(f"  WIN:     {wins:3d} / {measured}")
    print(f"  PARITY:  {parities:3d} / {measured}")
    print(f"  LOSS:    {losses:3d} / {measured}")
    print()
    print(f"  --- Verdicts vs C aggressive (-O3 -march=native -mavx512f -ffast-math) ---")
    print(f"  WIN:     {wins_vs_agg:3d} / {measured}")
    print(f"  PARITY:  {parities_vs_agg:3d} / {measured}")
    print(f"  LOSS:    {losses_vs_agg:3d} / {measured}")
    print()
    print(f"  --- vec_iso speedup distribution ---")
    print(f"  vec_iso > 1.5× (real wins vs scalar_jit): {vec_iso_gt15:3d} / {measured}")
    print()
    print(f"  --- Prior verdict comparison (vs CASTLE Intel audit) ---")
    print(f"  Maintained: {maintained}")
    print(f"  Improved:   {improved}")
    print(f"  Regressed:  {regressed}")
    if measured > 0:
        win_rate = (wins + parities) / measured * 100
        print()
        print(f"  WIN+PARITY rate vs C O3: {win_rate:.1f}% of measured candidates")
        if wins > 0:
            print(f"  {wins} candidate(s) WIN vs C O3 (vs_c_O3 > 1.05×).")
        if regressed > 0:
            print(f"  WARNING: {regressed} candidate(s) regressed vs prior CASTLE verdict.")
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
                    wins_vs_agg, parities_vs_agg, losses_vs_agg)
    print(f"  Dashboard Markdown saved to: {out_md}")
    print()


def _write_markdown(results, out_path: Path, target: str,
                    wins, parities, losses, skips, errors,
                    n_verified, n_pending, n_fail_v,
                    measured, vec_iso_gt15,
                    wins_vs_agg=0, parities_vs_agg=0, losses_vs_agg=0):
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
        f"| SKIP | {skips} |",
        f"| **WIN (vs C O3)** | **{wins}** |",
        f"| **PARITY (vs C O3)** | **{parities}** |",
        f"| **LOSS (vs C O3)** | **{losses}** |",
        f"| WIN (vs C agg) | {wins_vs_agg} |",
        f"| PARITY (vs C agg) | {parities_vs_agg} |",
        f"| LOSS (vs C agg) | {losses_vs_agg} |",
        f"| ERROR | {errors} |",
        f"| VERIFIED (correctness) | {n_verified} |",
        f"| PENDING (correctness) | {n_pending} |",
        f"| vec_iso > 1.5× | {vec_iso_gt15} |",
        f"",
        f"**Verdict basis:** `vs_c_O3 = c_O3_ms / vec_jit_ms`",
        f"WIN if > 1.05×, PARITY if >= 0.95×, LOSS otherwise.",
        f"",
        f"## Per-Candidate Results",
        f"",
        f"| Candidate | Layout | Prior | N | numpy_ms | scalar_ms | vec_ms | vec_iso | c_O3_ms | vs_c_O3 | c_agg_ms | vs_c_agg | Verify | Verdict |",
        f"|-----------|--------|-------|---|----------|-----------|--------|---------|---------|---------|----------|----------|--------|---------|",
    ]
    for r in results:
        name   = r.get("name", "?")[:32]
        layout = r.get("layout_class", "")[:15]
        prior  = r.get("prior_verdict", "")
        n_val  = r.get("N", "?")
        t_np   = r.get("numpy_ms",      float('nan'))
        t_sc   = r.get("scalar_jit_ms", float('nan'))
        t_vec  = r.get("vec_jit_ms",    float('nan'))
        sp_iso = r.get("speedup_isolated_jit", float('nan'))
        c_O3   = float(r.get("c_O3_ms",  float('nan')))
        c_agg  = float(r.get("c_agg_ms", float('nan')))
        vs_O3  = c_O3  / t_vec if (not math.isnan(c_O3)  and not math.isnan(t_vec) and t_vec > 0) else float('nan')
        vs_agg = c_agg / t_vec if (not math.isnan(c_agg) and not math.isnan(t_vec) and t_vec > 0) else float('nan')
        verify = r.get("verify_status", "-")
        verify_short = "OK" if verify == "VERIFIED" else ("?" if "PENDING" in verify else ("–" if verify == "NOT_RUN" else "FAIL"))

        # Recompute verdict from vs_O3 for markdown
        if not math.isnan(vs_O3):
            if vs_O3 > 1.05:   md_verdict = "**WIN**"
            elif vs_O3 >= 0.95: md_verdict = "PARITY"
            else:               md_verdict = "LOSS"
        else:
            md_verdict = r.get("verdict", "?")

        lines.append(
            f"| {name} | {layout} | {prior} | {n_val} | {_fmt(t_np)} | {_fmt(t_sc)} | {_fmt(t_vec)} | {_fmtx(sp_iso)} | {_fmt(c_O3)} | {_fmtx(vs_O3)} | {_fmt(c_agg)} | {_fmtx(vs_agg)} | {verify_short} | {md_verdict} |"
        )

    lines += [
        f"",
        f"## Known Gaps",
        f"",
        f"- **R20 (deinterleave)**: Implemented for stride 2/4/8. Generates ShuffleOp chains",
        f"  instead of vector.gather for constant-stride accesses. Correctness verified.",
        f"",
        f"- **R19 (strided gather indices)**: Strided-gather index vector mismatch in",
        f"  `LegoVectorize::emitVectorBody`. The catch-all arith path vectorizes `MulIOp(iv, stride)`",
        f"  before the Strided path reads it, producing incorrect gather indices.",
        f"  Affects candidates 23-27, 29-31, 38. Fix: use pre-vectorization scalar index.",
        f"",
        f"- **R18 (reduction guard)**: k-reduction loops correctly skip vectorization.",
        f"  This is correct behavior but contributes to PARITY (not WIN) for GEMM variants.",
        f"",
        f"- **invoke() overhead**: For small-N kernels (N=16K), the MLIR ExecutionEngine",
        f"  invoke() call costs ~4-5ms, dominating the actual kernel time (~0.002ms).",
        f"  The vec_jit_ms measurement includes this overhead; only the net kernel time",
        f"  (vec_jit_ms - invoke_overhead) is comparable to C baselines.",
        f"",
    ]

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
