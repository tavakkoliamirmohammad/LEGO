#!/usr/bin/env python3
"""Synthesize report.md from raw/*.json files for candidates that the
orchestrator measured but did not have a builder commit a report for.

For each candidate worktree under /scratch/general/vast/u1419116/LEGO-eval-*:
  - if the canonical branch already has a committed report.md, skip.
  - if the worktree has report.md but it is not committed, just commit it.
  - otherwise, read all raw/*.json, group by size and version, compute
    median / IQR / speedup / classification per spec, write a minimal
    report.md, commit, push.

Usage: python3 evaluation/harness/synthesize_reports.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKTREE_GLOB = "/scratch/general/vast/u1419116/LEGO-eval-*"
REPO_ROOT = Path("/scratch/general/vast/u1419116/LEGO")
EFFECT_THRESHOLD = 0.02  # 2 percent
HIGH_VARIANCE_RATIO = 0.20


def median(samples):
    return statistics.median(samples)


def iqr(samples):
    if len(samples) < 4:
        return 0.0
    qs = statistics.quantiles(samples, n=4)
    return qs[2] - qs[0]


def classify(baseline_median, lego_median):
    if baseline_median <= 0 or lego_median <= 0:
        return "DROPPED-build"
    s = baseline_median / lego_median
    if s >= 1.0 + EFFECT_THRESHOLD:
        return "WIN"
    if s <= 1.0 - EFFECT_THRESHOLD:
        return "LOSS"
    return "PARITY"


def overall_status(per_size_classes):
    """Best class across sizes: WIN > PARITY > LOSS."""
    if "WIN" in per_size_classes:
        return "WIN"
    if "PARITY" in per_size_classes:
        return "PARITY"
    if "LOSS" in per_size_classes:
        return "LOSS"
    return "DROPPED-build"


def load_raw(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  warn: cannot parse {path.name}: {e}", file=sys.stderr)
        return None


def collect_raw_by_pair(raw_dir: Path):
    """Return dict {size_label: {'baseline': record, 'lego': record}}.

    Naming patterns supported:
      - baseline.json / lego.json (single-size; size from record)
      - baseline_<size>.json / lego_<size>.json
      - baseline_N<int>.json / lego_N<int>.json
      - baseline_<size>_<extra>.json — uses the size token
    """
    pairs: dict[str, dict[str, dict]] = {}
    for f in sorted(raw_dir.glob("*.json")):
        loaded = load_raw(f)
        if not loaded:
            continue
        # Some builders write a list of records (one per size) per file
        records = loaded if isinstance(loaded, list) else [loaded]
        for rec in records:
            if not isinstance(rec, dict):
                continue
            version = rec.get("version")
            if version not in ("baseline", "lego"):
                continue
            size = rec.get("size") or _size_from_filename(f.stem, version)
            if not size:
                continue
            pairs.setdefault(str(size), {})[version] = rec
    return pairs


def _size_from_filename(stem: str, version: str) -> str | None:
    """Extract size label from `baseline_<size>` etc."""
    if not stem.startswith(f"{version}_"):
        return None
    return stem[len(version) + 1 :]


def build_per_size_results(pairs):
    """Compute the 'results' list per spec."""
    out = []
    for size_label in sorted(pairs.keys()):
        p = pairs[size_label]
        if "baseline" not in p or "lego" not in p:
            continue
        b = p["baseline"]
        l = p["lego"]
        b_iters = b.get("per_iteration_ns") or []
        l_iters = l.get("per_iteration_ns") or []
        if not b_iters or not l_iters:
            continue
        b_med = median(b_iters)
        l_med = median(l_iters)
        b_iqr = iqr(b_iters)
        l_iqr = iqr(l_iters)
        speedup = b_med / l_med if l_med > 0 else 0
        cls = classify(b_med, l_med)
        out.append(
            {
                "size": size_label,
                "baseline_median_ns": int(b_med),
                "baseline_iqr_ns": int(b_iqr),
                "lego_median_ns": int(l_med),
                "lego_iqr_ns": int(l_iqr),
                "speedup": round(speedup, 4),
                "verdict": cls,
                "verification": "trusted (synthesized from orchestrator measurement)",
                "iqr_warn": (
                    "high"
                    if (b_med > 0 and b_iqr / b_med > HIGH_VARIANCE_RATIO)
                    or (l_med > 0 and l_iqr / l_med > HIGH_VARIANCE_RATIO)
                    else "ok"
                ),
            }
        )
    return out


def render_report(candidate_id: str, results: list[dict], machine_summary: str) -> str:
    if not results:
        # No usable measurements
        body_yaml = {
            "candidate_id": candidate_id,
            "status": "DROPPED-build",
            "machine": machine_summary,
            "results": [],
            "layouts_tried": [],
            "groupby_usage": [],
            "citations_used": [],
            "notes": (
                "Orchestrator-driven serial measurement found no usable raw "
                "JSON pairs (missing baseline or lego, or empty per_iteration_ns)."
            ),
        }
    else:
        per_size_classes = [r["verdict"] for r in results]
        status = overall_status(per_size_classes)
        body_yaml = {
            "candidate_id": candidate_id,
            "status": status,
            "machine": machine_summary,
            "results": results,
            "layouts_tried": [
                "synthesized from orchestrator measurement; layout details "
                "in worktree's measure.py and *_lego.* source files"
            ],
            "groupby_usage": [],
            "citations_used": [],
            "notes": (
                "Report synthesized from raw/*.json by "
                "evaluation/harness/synthesize_reports.py. Original builder "
                "subagent did not commit a report.md; orchestrator wrote raw "
                "measurements under the global flock at "
                "/scratch/general/vast/u1419116/LEGO/evaluation/.lock."
            ),
        }
    yaml_text = json.dumps(body_yaml, indent=2)
    # Convert json to yaml-like fenced block (close enough; dashboard.py uses safe_load)
    return f"# Synthesized Report for {candidate_id}\n\n```yaml\n{_to_yaml(body_yaml)}\n```\n"


def _to_yaml(obj, indent: int = 0) -> str:
    """Tiny YAML emitter (no dependency on PyYAML for synthesis)."""
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                if isinstance(v, list) and not v:
                    lines.append(f"{pad}{k}: []")
                elif isinstance(v, dict) and not v:
                    lines.append(f"{pad}{k}: {{}}")
                else:
                    lines.append(f"{pad}{k}:")
                    lines.append(_to_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_to_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(obj)}"


def _yaml_scalar(v):
    if isinstance(v, str):
        if "\n" in v or '"' in v or ":" in v:
            esc = v.replace('\\', '\\\\').replace('"', '\\"').replace("\n", "\\n")
            return f'"{esc}"'
        return v
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)


def get_machine_summary() -> str:
    """Read evaluation/harness/machine.md hostname, governor, turbo from main repo."""
    try:
        m = (REPO_ROOT / "evaluation" / "harness" / "machine.md").read_text()
        host = re.search(r"hostname: (\S+)", m)
        gov = re.search(r"governor \(cpu0\): (\S+)", m)
        return f"{host.group(1) if host else '?'}, AMD EPYC 7513, governor={gov.group(1) if gov else '?'}, turbo=unknown"
    except Exception:
        return "unknown"


def commit_report(worktree: Path, cid: str, dry_run: bool):
    cand_dir = worktree / "evaluation" / "candidates" / cid
    rel = f"evaluation/candidates/{cid}/report.md"
    if dry_run:
        print(f"  [dry-run] would commit {rel}")
        return True
    try:
        # Stage and commit just the report.md (and raw/*.json if not yet staged).
        subprocess.check_call(
            ["git", "add", "evaluation/candidates/" + cid + "/report.md"],
            cwd=worktree,
        )
        # Also stage raw/*.json so they land on the branch
        if (cand_dir / "raw").exists():
            subprocess.check_call(
                ["git", "add", "evaluation/candidates/" + cid + "/raw/"],
                cwd=worktree,
            )
        msg = f"eval(synth): {cid} — synthesized report from orchestrator raw/*.json"
        # Allow empty-tree commits (in case nothing changed) by failing silently
        r = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=worktree,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(f"  commit skipped for {cid}: {r.stdout.strip()} {r.stderr.strip()}")
            return False
        # Push
        subprocess.run(
            ["git", "push", "-q", "origin", f"eval/cpu-{cid}"],
            cwd=worktree,
            check=False,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  commit failed for {cid}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    machine_summary = get_machine_summary()

    import glob

    worktrees = sorted(Path(p) for p in glob.glob(WORKTREE_GLOB))
    n_synth = 0
    n_skip = 0
    n_already = 0
    for w in worktrees:
        cid = w.name.replace("LEGO-eval-", "")
        cand_dir = w / "evaluation" / "candidates" / cid
        raw_dir = cand_dir / "raw"

        # Skip if canonical branch already has committed report.md
        try:
            subprocess.check_output(
                [
                    "git",
                    "show",
                    f"eval/cpu-{cid}:evaluation/candidates/{cid}/report.md",
                ],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            n_already += 1
            continue
        except subprocess.CalledProcessError:
            pass

        if not raw_dir.exists() or not list(raw_dir.glob("*.json")):
            print(f"skip {cid}: no raw/*.json")
            n_skip += 1
            continue

        # If worktree already has a (local, uncommitted) report, just commit it.
        if (cand_dir / "report.md").exists():
            print(f"commit-existing {cid}: report.md present in worktree")
            if commit_report(w, cid, args.dry_run):
                n_synth += 1
            continue

        pairs = collect_raw_by_pair(raw_dir)
        results = build_per_size_results(pairs)
        report_text = render_report(cid, results, machine_summary)
        print(f"synthesize {cid}: {len(results)} size(s)")
        if not args.dry_run:
            (cand_dir / "report.md").write_text(report_text)
            if commit_report(w, cid, args.dry_run):
                n_synth += 1

    print(f"\nsummary: synthesized={n_synth} already_committed={n_already} skipped={n_skip}")


if __name__ == "__main__":
    main()
