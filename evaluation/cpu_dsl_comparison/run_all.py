#!/usr/bin/env python3
"""Run every cpu_dsl_comparison benchmark and print a summary table.

Usage (activate the venv first; no PYTHONPATH override needed):
    source /scratch/general/vast/u1419116/LEGO/venv/bin/activate
    python run_all.py

Each candidate's measure.py is run in a subprocess.  The last line of
stdout must be a single JSON record conforming to the schema::

    {
      "name":        "<candidate_id>",
      "baseline_ms": <float>,
      "cpu_dsl_ms":  <float | NaN>,
      "speedup":     <float | NaN>,
      "verdict":     "WIN" | "PARITY" | "LOSS" | "ERROR",
      "notes":       "<optional string>"
    }

The top-level results.json is written alongside this script.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
CANDIDATES_DIR = ROOT / "candidates"

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
        proc = subprocess.run(
            [PYTHON, str(measure_py)],
            capture_output=True,
            text=True,
            timeout=120,
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
        results.append(rec)
    except subprocess.TimeoutExpired:
        results.append({"name": cand_dir.name, "verdict": "ERROR",
                        "notes": "timeout after 120s"})
    except Exception as exc:
        results.append({"name": cand_dir.name, "verdict": "ERROR",
                        "notes": str(exc)})

# Print summary table.
print()
print(f"{'Name':<28} {'Baseline ms':>14} {'cpu_dsl ms':>14} {'Speedup':>10} {'Verdict':>10}")
print("-" * 82)
for r in results:
    name = r.get("name", "?")
    verdict = r.get("verdict", "")
    base = r.get("baseline_ms", math.nan)
    dsl = r.get("cpu_dsl_ms", math.nan)
    sp = r.get("speedup", math.nan)

    def _fmt(v):
        return f"{v:>14.3f}" if isinstance(v, float) and not math.isnan(v) else f"{'NaN':>14}"

    def _fmt_sp(v):
        return f"{v:>9.2f}x" if isinstance(v, float) and not math.isnan(v) else f"{'NaN':>10}"

    if verdict == "ERROR":
        note_snippet = r.get("notes", "")[:50]
        print(f"{name:<28} {'(error)':>14} {'':>14} {'':>10} {verdict:>10}  {note_snippet}")
    else:
        print(f"{name:<28} {_fmt(base)} {_fmt(dsl)} {_fmt_sp(sp)} {verdict:>10}")

# Save full results.
out_path = ROOT / "results.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nFull results saved to: {out_path}")
