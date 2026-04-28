#!/usr/bin/env python3
"""Step 4 audit: re-measure suspicious-large-speedup WINs at non-pow-2
sizes only, per builder 06's principled approach.

Background: several WIN candidates (07-trmm 27.6x, 08-doitgen 4x at
NP=60, 16-syrk 7.15x at N=512, 33-mvt 3.24x at N=1000, 34-bicg 51.7x at
N~2k) had speedups inflated by the *baseline* hitting pow-2 cache-set
pathologies — not by the LEGO layout itself. To separate "real layout
gain" from "baseline pow-2 pathology", we re-run those candidates at
non-pow-2 sizes only.

Per-candidate sizes selected to:
  - span at least 4x compute volume
  - avoid stride lengths that hit 8-way L1 set conflicts on EPYC
  - stay below L3 (32 MB) so DRAM-bound is consistent

Outputs raw/audit_baseline_<size>.json and raw/audit_lego_<size>.json
inside each candidate's worktree. Does NOT overwrite the original
raw/*.json. Holds the global lock during measurement.

Usage: source venv/bin/activate && python3 evaluation/harness/audit_nonpow2.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

GLOBAL_LOCK = Path("/scratch/general/vast/u1419116/LEGO/evaluation/.lock")

# Per-candidate audit plan: (candidate_id, [size_labels_with_N])
# Each size is (label, N1, N2, ...) — depends on the kernel's parameterization.
# Sizes chosen to be NON-power-of-two and to avoid stride=N hitting L1 set conflicts.
AUDIT_PLAN = [
    # (id, [(size_label, N_or_dims), ...])
    ("07-polybench-trmm-L1-L2-tile",
     [("audit_500", "500"), ("audit_900", "900"), ("audit_1500", "1500")]),
    ("08-polybench-doitgen-reg-L1-tile",
     [("audit_NR50_NQ40_NP55", "50,40,55"),
      ("audit_NR100_NQ80_NP110", "100,80,110"),
      ("audit_NR140_NQ130_NP150", "140,130,150")]),
    ("16-polybench-syrk-rfp",
     [("audit_500", "500"), ("audit_900", "900"), ("audit_1500", "1500")]),
    ("33-polybench-mvt-L1-tile",
     [("audit_900", "900"), ("audit_1500", "1500"), ("audit_1900", "1900")]),
    ("34-polybench-bicg-L1-tile",
     [("audit_900_950", "900,950"),
      ("audit_1500_1550", "1500,1550"),
      ("audit_1900_2050", "1900,2050")]),
    ("13-polybench-heat3d-brick",
     [("audit_75", "75"), ("audit_120", "120"), ("audit_180", "180")]),
]


def median(samples):
    return statistics.median(samples)


def run_under_lock(cmd, env=None) -> tuple[int, str, str]:
    """Run a command holding the global flock. Returns (rc, stdout, stderr)."""
    args = ["flock", "-x", str(GLOBAL_LOCK), "bash", "-c", cmd]
    p = subprocess.run(args, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def find_worktree(cid: str) -> Path | None:
    p = Path(f"/scratch/general/vast/u1419116/LEGO-eval-{cid}")
    return p if p.exists() else None


def main():
    print(f"Audit pass started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Global lock: {GLOBAL_LOCK}")
    print()
    for cid, sizes in AUDIT_PLAN:
        wt = find_worktree(cid)
        if not wt:
            print(f"skip {cid}: no worktree")
            continue
        cand_dir = wt / "evaluation" / "candidates" / cid
        if not cand_dir.exists():
            print(f"skip {cid}: no candidate dir")
            continue
        # Each candidate's measure.py is custom — call it with sized arguments
        # if it accepts them. Otherwise the operator must wire up per-candidate
        # invocations. For the initial implementation we just print the plan.
        measure_py = cand_dir / "measure.py"
        if not measure_py.exists():
            print(f"skip {cid}: no measure.py")
            continue
        print(f"=== audit {cid} ({len(sizes)} non-pow-2 size(s)) ===")
        for label, size_arg in sizes:
            print(f"  size {label}: N={size_arg}")
            # Naive invocation: most measure.py scripts already pin sizes
            # internally. Audit-mode is a SEPARATE invocation that the
            # candidate's measure.py must support; if it doesn't, the
            # operator must hand-edit measure.py to accept --audit-sizes.
            # For now, this script prints the plan; the actual measurement
            # is done by re-running each candidate's measure.py manually
            # with the new sizes.
        print()
    print()
    print("NOTE: this audit script prints the plan only. Each candidate's")
    print("measure.py must be invoked manually with the audit sizes (or")
    print("modified to accept --audit-sizes <list>). This is intentional")
    print("until we have a uniform measurement interface.")
    print()
    print("Suggested manual sequence per candidate:")
    print("  cd <worktree>/evaluation/candidates/<cid>")
    print("  flock -x /scratch/general/vast/u1419116/LEGO/evaluation/.lock \\")
    print("    python3 measure.py --sizes <non-pow-2-sizes>")


if __name__ == "__main__":
    main()
