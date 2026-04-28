"""Regenerate evaluation/dashboard.md from candidate report.md files.

Each candidate directory under evaluation/candidates/ should contain a
report.md whose first ```yaml block has the schema described in spec
Section 9. This script collects them all, builds a status table, and
writes evaluation/dashboard.md.

Run as: python -m evaluation.harness.dashboard
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_DIR = EVAL_DIR / "candidates"
DEFAULT_DASHBOARD_PATH = EVAL_DIR / "dashboard.md"

YAML_BLOCK_RE = re.compile(r"```yaml\s*\n(.*?)\n```", re.DOTALL)


def parse_report(report_path: Path) -> dict[str, Any] | None:
    text = report_path.read_text()
    m = YAML_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        loaded = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def collect_reports(candidates_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not candidates_dir.exists():
        return reports
    for cand_dir in sorted(candidates_dir.iterdir()):
        if not cand_dir.is_dir():
            continue
        report_path = cand_dir / "report.md"
        if not report_path.exists():
            reports.append({
                "candidate_id": cand_dir.name,
                "status": "PENDING",
            })
            continue
        data = parse_report(report_path)
        if data is None:
            reports.append({
                "candidate_id": cand_dir.name,
                "status": "MALFORMED",
            })
        else:
            reports.append(data)
    return reports


def _best_speedup_row(report: dict[str, Any]) -> dict[str, Any] | None:
    results = report.get("results") or []
    if not results:
        return None
    best = None
    for r in results:
        try:
            sp = float(r.get("speedup", 0))
        except (TypeError, ValueError):
            continue
        if best is None or sp > float(best.get("speedup", 0)):
            best = r
    return best


def _last_layout(report: dict[str, Any]) -> str:
    layouts = report.get("layouts_tried") or []
    if not layouts:
        return "—"
    last = layouts[-1]
    if isinstance(last, dict) and last:
        return next(iter(last))
    return str(last)


def _repro_summary(report: dict[str, Any]) -> str:
    machine = str(report.get("machine", ""))
    for tok in machine.split(","):
        tok = tok.strip()
        if tok.startswith("turbo="):
            return tok
    return ""


def render_dashboard(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# CASTLE CPU Evaluation Dashboard",
        "",
        f"Auto-generated from {len(reports)} candidate report(s).",
        "",
        "| id | status | speedup | layout | size | repro_setup |",
        "|----|--------|---------|--------|------|-------------|",
    ]
    for r in reports:
        cid = r.get("candidate_id", "?")
        status = r.get("status", "?")
        best = _best_speedup_row(r)
        if best is not None:
            speedup = f"{float(best.get('speedup', 0)):.2f}x"
            size = str(best.get("size", "—"))
        else:
            speedup = "—"
            size = "—"
        layout = _last_layout(r)
        repro = _repro_summary(r)
        lines.append(f"| {cid} | {status} | {speedup} | {layout} | {size} | {repro} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    reports = collect_reports(DEFAULT_CANDIDATES_DIR)
    DEFAULT_DASHBOARD_PATH.write_text(render_dashboard(reports))
    print(f"Wrote {DEFAULT_DASHBOARD_PATH} ({len(reports)} candidates)")


if __name__ == "__main__":
    main()
