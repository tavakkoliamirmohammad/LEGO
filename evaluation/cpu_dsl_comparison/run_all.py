#!/usr/bin/env python3
"""Run every cpu_dsl_comparison benchmark and print a summary dashboard.

Usage (activate the venv first):
    source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
    python run_all.py

Each candidate's measure.py is run in a subprocess.  The last line of
stdout must be a single JSON record in the isolation schema:

    {
      "name":                "<candidate_id>",
      "N":                   <int>,
      "layout_class":        "<class>",        # optional, shown in dashboard
      "prior_verdict":       "WIN|PARITY|LOSS|MIXED",  # from CASTLE audit
      "numpy_ms":            <float | NaN>,
      "scalar_jit_ms":       <float | NaN>,
      "vec_jit_ms":          <float | NaN>,
      "speedup_isolated_jit": <float | NaN>,   # scalar_jit / vec_jit
      "speedup_vs_numpy":    <float | NaN>,
      "verdict":             "WIN|PARITY|LOSS|SKIP|ERROR",
      "notes":               "<optional string>"
    }

The top-level results.json is written alongside this script.

C-baseline integration:
    For each candidate, run_all.py also invokes the matching C-O3-march=native
    binary from c_baselines/ before the Python benchmark.  Binaries must be
    pre-built: cd c_baselines && make all

Dashboard bottom-line metrics:
    - How many candidates show vec_iso > 1.5× (real speedup)?
    - How many maintained/improved their prior CASTLE verdict?
    - How many regressed?
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CANDIDATES_DIR = ROOT / "candidates"
C_BASELINES_DIR = ROOT / "c_baselines"

# Map candidate directory prefix → (c_binary_name, N_to_pass_or_None).
_C_BASELINE_MAP = {
    "01_saxpy":               ("saxpy",              1 << 20),
    "02_gemm_row_major":      ("gemm",               64),
    "03_3pt_stencil_1d":      ("stencil_3pt",        1024),
    "04_col_major_inner":     ("col_major",           256),
    "05_morton_2d":           ("morton",              1 << 16),
    "06_self_update":         ("self_update",         4096),
    "07_mixed_precision":     ("mixed_precision",     1 << 20),
    "08_brick_within_cell":   ("brick_within_cell",   1 << 20),
}


def _run_c_binary(bin_path, N) -> float:
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
    """Run the GCC C-baseline binary for `cand_name`, return ms_per_call or NaN."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    if bin_name is None:
        return float('nan')
    return _run_c_binary(C_BASELINES_DIR / bin_name, N)


def run_clang_baseline(cand_name: str) -> float:
    """Run the Clang C-baseline binary for `cand_name`, return ms_per_call or NaN."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    if bin_name is None:
        return float('nan')
    return _run_c_binary(C_BASELINES_DIR / f"{bin_name}_clang", N)


# Prefer the venv Python so MLIR .so files resolve correctly.
_VENV_PYTHON = Path("/scratch/general/vast/u1419116/LEGO/venv/bin/python")
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

results = []
for cand_dir in sorted(CANDIDATES_DIR.iterdir()):
    if not cand_dir.is_dir():
        continue
    measure_py = cand_dir / "measure.py"
    if not measure_py.exists():
        continue
    print(f"Running {cand_dir.name}...", flush=True)

    c_ms = run_c_baseline(cand_dir.name)
    clang_ms = run_clang_baseline(cand_dir.name)
    if not math.isnan(c_ms):
        print(f"  C baseline (gcc):   {c_ms:.4f} ms/call", flush=True)
    if not math.isnan(clang_ms):
        print(f"  C baseline (clang): {clang_ms:.4f} ms/call", flush=True)

    try:
        _build_lego = "/scratch/general/vast/u1419116/LEGO/build/python_packages/lego"
        existing_pp = os.environ.get("PYTHONPATH", "")
        filtered = ":".join(
            p for p in existing_pp.split(":") if p and "LEGO/python" not in p
        )
        new_pp = _build_lego + (":" + filtered if filtered else "")
        sub_env = dict(os.environ)
        sub_env["PYTHONPATH"] = new_pp

        proc = subprocess.run(
            [PYTHON, str(measure_py)],
            capture_output=True,
            text=True,
            timeout=300,
            env=sub_env,
        )
        stdout = proc.stdout.strip()
        last_line = stdout.split("\n")[-1] if stdout else ""
        try:
            rec = json.loads(last_line)
        except json.JSONDecodeError:
            rec = {
                "name": cand_dir.name,
                "verdict": "ERROR",
                "notes": (proc.stdout[-400:] + "\n" + proc.stderr[-400:]).strip(),
            }
        rec.setdefault("name", cand_dir.name)
        if not math.isnan(c_ms):
            rec["c_baseline_ms"] = round(c_ms, 6)
        if not math.isnan(clang_ms):
            rec["c_clang_ms"] = round(clang_ms, 6)
        results.append(rec)
    except subprocess.TimeoutExpired:
        results.append({"name": cand_dir.name, "verdict": "ERROR",
                        "notes": "timeout after 300s",
                        **({"c_baseline_ms": round(c_ms, 6)} if not math.isnan(c_ms) else {}),
                        **({"c_clang_ms": round(clang_ms, 6)} if not math.isnan(clang_ms) else {})})
    except Exception as exc:
        results.append({"name": cand_dir.name, "verdict": "ERROR",
                        "notes": str(exc),
                        **({"c_baseline_ms": round(c_ms, 6)} if not math.isnan(c_ms) else {}),
                        **({"c_clang_ms": round(clang_ms, 6)} if not math.isnan(clang_ms) else {})})


# ============================================================================
# Dashboard formatting
# ============================================================================

def _fv(v, w, fmt=".4f"):
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


_has_c = any("c_baseline_ms" in r for r in results)
_has_clang = any("c_clang_ms" in r for r in results)
_has_prior = any("prior_verdict" in r for r in results)
_has_layout = any("layout_class" in r for r in results)

print()
print("=" * 160)
print("  LEGO cpu_dsl_comparison — CASTLE Round-1 Full Coverage Dashboard")
print("=" * 160)

# Header
hdr_parts = [f"{'Candidate':<32}"]
if _has_layout:
    hdr_parts.append(f"{'LayoutClass':<20}")
if _has_prior:
    hdr_parts.append(f"{'PriorVerd':>9}")
hdr_parts += [f"{'N':>8}", f"{'numpy_ms':>11}", f"{'scalar_jit':>11}",
               f"{'vec_jit':>11}", f"{'vec_iso':>9}", f"{'vs_numpy':>9}"]
if _has_c:
    hdr_parts += [f"{'c_gcc_ms':>11}", f"{'vs_gcc':>8}"]
if _has_clang:
    hdr_parts += [f"{'c_clang_ms':>11}", f"{'vs_clang':>9}"]
hdr_parts.append(f"{'Verdict':>8}")
hdr = "  ".join(hdr_parts)
print(hdr)
print("-" * len(hdr))

wins = losses = parities = skips = errors = 0
improved = maintained = regressed = 0
vec_iso_gt15 = 0

for r in results:
    name = r.get("name", "?")[:32]
    verdict = r.get("verdict", "ERROR")
    layout = r.get("layout_class", "")[:20]
    prior = r.get("prior_verdict", "")
    n_val = r.get("N", "")
    n_str = f"{n_val:>8}" if isinstance(n_val, int) else f"{'?':>8}"
    c_ms_r = r.get("c_baseline_ms", float('nan'))
    if not isinstance(c_ms_r, float):
        c_ms_r = float('nan')
    clang_ms_r = r.get("c_clang_ms", float('nan'))
    if not isinstance(clang_ms_r, float):
        clang_ms_r = float('nan')

    t_numpy  = r.get("numpy_ms",  float('nan'))
    t_scalar = r.get("scalar_jit_ms", float('nan'))
    t_vec    = r.get("vec_jit_ms", float('nan'))
    sp_iso   = r.get("speedup_isolated_jit", float('nan'))
    sp_np    = r.get("speedup_vs_numpy", float('nan'))
    vs_c     = _safe_ratio(c_ms_r, t_vec)
    vs_clang = _safe_ratio(clang_ms_r, t_vec)

    # Tally
    if verdict == "WIN":
        wins += 1
    elif verdict == "LOSS":
        losses += 1
    elif verdict == "PARITY":
        parities += 1
    elif verdict == "SKIP":
        skips += 1
    else:
        errors += 1

    if not math.isnan(sp_iso) and sp_iso > 1.5:
        vec_iso_gt15 += 1

    # Prior comparison
    if prior in ("WIN", "PARITY", "LOSS", "MIXED"):
        prior_win  = prior in ("WIN",)
        new_win    = verdict in ("WIN",)
        new_skip   = verdict in ("SKIP",)
        if not new_skip:
            if prior_win and new_win:
                maintained += 1
            elif not prior_win and new_win:
                improved += 1
            elif prior_win and not new_win:
                regressed += 1
            else:
                maintained += 1  # LOSS→PARITY etc. counts as maintained

    row_parts = [f"{name:<32}"]
    if _has_layout:
        row_parts.append(f"{layout:<20}")
    if _has_prior:
        row_parts.append(f"{prior:>9}")
    row_parts += [n_str, _fv(t_numpy, 11, '.3f'), _fv(t_scalar, 11, '.3f'),
                   _fv(t_vec, 11, '.3f'), _fx(sp_iso, 9), _fx(sp_np, 9)]
    if _has_c:
        row_parts += [_fv(c_ms_r, 11, '.3f'), _fx(vs_c, 8)]
    if _has_clang:
        row_parts += [_fv(clang_ms_r, 11, '.3f'), _fx(vs_clang, 9)]
    row_parts.append(f"{verdict:>8}")
    print("  ".join(row_parts))

print("-" * len(hdr))

# Summary stats
total = len(results)
measured = total - skips
print()
print(f"  Total candidates:      {total}")
print(f"  Measured (non-SKIP):   {measured}")
print(f"  SKIP (XFAIL/unbundled): {skips}")
print(f"  Errors:                {errors}")
print()
print(f"  --- New cpu_dsl verdicts (measured only) ---")
print(f"  WIN:    {wins:3d} / {measured}")
print(f"  PARITY: {parities:3d} / {measured}")
print(f"  LOSS:   {losses:3d} / {measured}")
print()
print(f"  --- vec_iso speedup distribution ---")
print(f"  vec_iso > 1.5× (real wins):    {vec_iso_gt15:3d} / {measured}")
print()
print(f"  --- Prior verdict comparison (vs CASTLE Intel audit) ---")
print(f"  Maintained (WIN→WIN, LOSS→LOSS, etc.): {maintained}")
print(f"  Improved  (non-WIN → WIN):             {improved}")
print(f"  Regressed (WIN → non-WIN):             {regressed}")
print()
if measured > 0:
    win_rate = (wins + parities) / measured * 100
    print(f"  WIN+PARITY rate: {win_rate:.1f}% of measured candidates")
    if vec_iso_gt15 > 0:
        print(f"  {vec_iso_gt15} candidate(s) show vec_iso > 1.5× — genuine cpu_dsl vectorization wins.")
    if wins > 0:
        print(f"  {wins} candidate(s) WIN (vec_iso > 1.05×).")
    if regressed > 0:
        print(f"  WARNING: {regressed} candidate(s) regressed vs prior CASTLE verdict.")

# Save full results
out_path = ROOT / "results.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nFull results saved to: {out_path}")
