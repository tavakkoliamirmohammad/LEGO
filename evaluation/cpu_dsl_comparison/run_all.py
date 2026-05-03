#!/usr/bin/env python3
"""Run every cpu_dsl_comparison benchmark and print a summary table.

Usage (activate the venv first; no PYTHONPATH override needed):
    source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
    python run_all.py

Each candidate's measure.py is run in a subprocess.  The last line of
stdout must be a single JSON record.  Two JSON schemas are accepted:

Legacy schema (old harnesses):
    {
      "name":        "<candidate_id>",
      "baseline_ms": <float>,
      "cpu_dsl_ms":  <float | NaN>,
      "speedup":     <float | NaN>,
      "verdict":     "WIN" | "PARITY" | "LOSS" | "ERROR",
      "notes":       "<optional string>"
    }

Isolation schema (new harnesses, apples-to-apples):
    {
      "name":                "<candidate_id>",
      "N":                   <int>,
      "numpy_ms":            <float>,
      "scalar_jit_ms":       <float | NaN>,
      "vec_jit_ms":          <float | NaN>,
      "speedup_isolated_jit": <float | NaN>,
      "speedup_vs_numpy":    <float | NaN>,
      "verdict":             "WIN" | "PARITY" | "LOSS" | "ERROR",
      "notes":               "<optional string>"
    }

The top-level results.json is written alongside this script.

C-baseline integration:
    For each candidate, run_all.py also invokes the matching C-O3-march=native
    binary from c_baselines/ before the Python benchmark. The JSON record gains
    a "c_baseline_ms" field (NaN if no matching binary) and a "vs_c" column in
    the dashboard: cpu_dsl_vec / c_baseline (ratio < 1 means we beat C-O3).
    Binaries must be pre-built: cd c_baselines && make all
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
# N_to_pass must match the actual N used in measure.py's timed section.
# Candidates with N_BENCH=1<<20 (1M) use large N for timing; others use kernel N.
_C_BASELINE_MAP = {
    "01_saxpy":          ("saxpy",            1 << 20),   # measure.py N_BENCH=1M
    "02_gemm_row_major": ("gemm",             64),        # kernel M=N=64
    "03_3pt_stencil_1d": ("stencil_3pt",      1024),      # kernel N=1024
    "04_col_major_inner":("col_major",        256),       # kernel M=N=256
    "05_morton_2d":      (None,               None),      # no equivalent C kernel
    "06_self_update":    ("self_update",       4096),      # kernel N=4096
    "07_mixed_precision":("mixed_precision",  1 << 20),   # measure.py N_BENCH=1M
    "08_brick_within_cell": ("brick_within_cell", 1 << 20),  # measure.py N_BENCH=1M
}


def run_c_baseline(cand_name: str) -> float:
    """Run the C-baseline binary for `cand_name`, return ms_per_call or NaN."""
    entry = _C_BASELINE_MAP.get(cand_name)
    if entry is None:
        return float('nan')
    bin_name, N = entry
    if bin_name is None:
        return float('nan')
    bin_path = C_BASELINES_DIR / bin_name
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

# Prefer the venv Python so MLIR .so files resolve correctly.
# Falls back to sys.executable if the venv Python is not present.
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

    # Run the C baseline first (does NOT require the venv).
    c_ms = run_c_baseline(cand_dir.name)
    if not math.isnan(c_ms):
        print(f"  C baseline: {c_ms:.4f} ms/call", flush=True)
    try:
        # Build a clean environment.  The LEGO pass pipelines (lego-to-x86-vector
        # etc.) are registered by the compiled C extension in
        # build/python_packages/lego — that directory must be first on PYTHONPATH
        # so the built .so files take precedence over both the source tree and the
        # venv's editable-install stub.  Setting PYTHONPATH to the raw source tree
        # (LEGO/python) instead causes the MLIR _mlir_libs import to fail because
        # the compiled .so files are absent there.
        _build_lego = "/scratch/general/vast/u1419116/LEGO/build/python_packages/lego"
        existing_pp = os.environ.get("PYTHONPATH", "")
        # Remove any existing LEGO/python entry to avoid shadowing.
        filtered = ":".join(
            p for p in existing_pp.split(":") if p and "LEGO/python" not in p
        )
        new_pp = _build_lego + (":" + filtered if filtered else "")
        sub_env = dict(os.environ)
        sub_env["PYTHONPATH"] = new_pp
        # Timeout raised from 120s to 300s: amortized harness runs 100 warmup +
        # 1000 timed iterations; for large-N kernels (N=1M) this can take ~30s
        # per compiled variant × 3 variants (numpy + scalar-jit + vec-jit).
        proc = subprocess.run(
            [PYTHON, str(measure_py)],
            capture_output=True,
            text=True,
            timeout=300,
            env=sub_env,
        )
        # measure.py prints a single JSON line as its last output line.
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
        # Attach C-baseline timing to the record.
        if not math.isnan(c_ms):
            rec["c_baseline_ms"] = round(c_ms, 6)
        results.append(rec)
    except subprocess.TimeoutExpired:
        results.append({"name": cand_dir.name, "verdict": "ERROR",
                        "notes": "timeout after 300s",
                        **({"c_baseline_ms": round(c_ms, 6)} if not math.isnan(c_ms) else {})})
    except Exception as exc:
        results.append({"name": cand_dir.name, "verdict": "ERROR",
                        "notes": str(exc),
                        **({"c_baseline_ms": round(c_ms, 6)} if not math.isnan(c_ms) else {})})


def _fmt_ms(v):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>13.3f}"
    return f"{'NaN':>13}"


def _fmt_sp(v, width=10):
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{width-1}.2f}x"
    return f"{'NaN':>{width}}"


def _safe_ratio(a, b):
    if (isinstance(a, float) and not math.isnan(a) and
            isinstance(b, float) and not math.isnan(b) and b > 0):
        return a / b
    return float('nan')


def _fv(v, w, fmt=".4f"):
    """Format a float value with width w; NaN if not a valid float."""
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{w}{fmt}}"
    return f"{'NaN':>{w}}"


def _fx(v, w):
    """Format a ratio as Nx with width w."""
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:>{w-1}.2f}x"
    return f"{'NaN':>{w}}"


# Check if any result uses the isolation schema.
_has_isolation = any("vec_jit_ms" in r for r in results)
# Check if any result has a C baseline.
_has_c_baseline = any("c_baseline_ms" in r for r in results)

print()
if _has_isolation:
    # Full isolation table with optional C-baseline column.
    if _has_c_baseline:
        hdr = (f"{'Name':<28} {'N':>8}"
               f" {'numpy_ms':>10} {'scalar_jit':>10} {'vec_jit':>10}"
               f" {'vec_iso':>9} {'vs_numpy':>9}"
               f" {'c_base_ms':>10} {'vs_c':>7} {'Verdict':>7}")
    else:
        hdr = (f"{'Name':<28} {'N':>8}"
               f" {'numpy_ms':>13} {'scalar_jit':>13} {'vec_jit':>13}"
               f" {'vec_isolated':>12} {'vs_numpy':>10} {'Verdict':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        name = r.get("name", "?")
        verdict = r.get("verdict", "")
        c_ms_r = r.get("c_baseline_ms", math.nan)
        if not isinstance(c_ms_r, float):
            c_ms_r = float('nan')

        if verdict == "ERROR":
            note_snippet = r.get("notes", "")[:60]
            if _has_c_baseline:
                print(f"{name:<28} {'(error)':>8}"
                      f" {'':>10} {'':>10} {'':>10} {'':>9} {'':>9}"
                      f" {_fv(c_ms_r, 10)} {'':>7} {verdict:>7}  {note_snippet}")
            else:
                print(f"{name:<28} {'(error)':>8}"
                      f" {'':>13} {'':>13} {'':>13} {'':>12} {'':>10} {verdict:>8}  {note_snippet}")
            continue

        n_val = r.get("N", "")
        n_str = f"{n_val:>8}" if isinstance(n_val, int) else f"{'?':>8}"

        # Isolation schema
        if "vec_jit_ms" in r:
            t_numpy  = r.get("numpy_ms",  math.nan)
            t_scalar = r.get("scalar_jit_ms", math.nan)
            t_vec    = r.get("vec_jit_ms", math.nan)
            sp_iso   = r.get("speedup_isolated_jit", math.nan)
            sp_np    = r.get("speedup_vs_numpy", math.nan)
            if _has_c_baseline:
                # vs_c = c_baseline_ms / vec_jit_ms
                # ratio > 1.0 → cpu_dsl is FASTER than C-O3; < 1.0 → slower.
                vs_c = _safe_ratio(c_ms_r, t_vec)
                print(f"{name:<28} {n_str}"
                      f" {_fv(t_numpy, 10)} {_fv(t_scalar, 10)} {_fv(t_vec, 10)}"
                      f" {_fx(sp_iso, 9)} {_fx(sp_np, 9)}"
                      f" {_fv(c_ms_r, 10)} {_fx(vs_c, 7)} {verdict:>7}")
            else:
                print(f"{name:<28} {n_str}"
                      f" {_fv(t_numpy, 13, '.3f')} {_fv(t_scalar, 13, '.3f')} {_fv(t_vec, 13, '.3f')}"
                      f" {_fx(sp_iso, 12)} {_fx(sp_np, 10)} {verdict:>8}")
        else:
            # Legacy schema fallback.
            base = r.get("baseline_ms", math.nan)
            dsl  = r.get("cpu_dsl_ms", math.nan)
            sp   = r.get("speedup", math.nan)
            print(f"{name:<28} {'?':>8}"
                  f" {_fv(base, 13, '.3f')} {'NaN':>13} {_fv(dsl, 13, '.3f')}"
                  f" {_fx(sp, 12)} {'NaN':>10} {verdict:>8}")
else:
    # Legacy table (no isolation results).
    print(f"{'Name':<28} {'Baseline ms':>14} {'cpu_dsl ms':>14} {'Speedup':>10} {'Verdict':>10}")
    print("-" * 82)
    for r in results:
        name = r.get("name", "?")
        verdict = r.get("verdict", "")
        base = r.get("baseline_ms", math.nan)
        dsl  = r.get("cpu_dsl_ms", math.nan)
        sp   = r.get("speedup", math.nan)
        if verdict == "ERROR":
            note_snippet = r.get("notes", "")[:50]
            print(f"{name:<28} {'(error)':>14} {'':>14} {'':>10} {verdict:>10}  {note_snippet}")
        else:
            print(f"{name:<28} {_fmt_ms(base)} {_fmt_ms(dsl)} {_fmt_sp(sp)} {verdict:>10}")

# Save full results.
out_path = ROOT / "results.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nFull results saved to: {out_path}")
