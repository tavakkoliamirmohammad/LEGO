"""Tests for dashboard generation from candidate report.md files."""

from pathlib import Path

from evaluation.harness import dashboard


REPORT_WIN = """
# Candidate report

```yaml
candidate_id: polybench-2mm-zcurve
status: WIN
machine: notch001, Xeon Gold, governor=ondemand, turbo=unknown
results:
  - size: medium
    baseline_median_ns: 1234567
    lego_median_ns: 987654
    speedup: 1.25
layouts_tried:
  - "OrderBy(Row).TileBy(...)+GenP(morton)": WIN at 1.25x
groupby_usage: []
```
"""

REPORT_LOSS = """
```yaml
candidate_id: rodinia-lud-rfp
status: LOSS
machine: notch001, Xeon Gold, governor=ondemand, turbo=off
results:
  - size: large
    baseline_median_ns: 1000000
    lego_median_ns: 1100000
    speedup: 0.91
layouts_tried:
  - "OrderBy(Row).TileBy(...) RFP": LOSS at 0.91x
groupby_usage: []
```
"""


def _make_candidate(parent: Path, slug: str, body: str) -> None:
    d = parent / slug
    d.mkdir()
    (d / "report.md").write_text(body)


def test_collect_reports_finds_two(tmp_path: Path) -> None:
    cands = tmp_path / "candidates"
    cands.mkdir()
    _make_candidate(cands, "01-polybench-2mm-zcurve", REPORT_WIN)
    _make_candidate(cands, "02-rodinia-lud-rfp", REPORT_LOSS)
    reports = dashboard.collect_reports(cands)
    assert len(reports) == 2
    statuses = sorted(r["status"] for r in reports)
    assert statuses == ["LOSS", "WIN"]


def test_collect_reports_marks_pending_when_missing(tmp_path: Path) -> None:
    cands = tmp_path / "candidates"
    cands.mkdir()
    (cands / "03-pending").mkdir()  # dir exists, no report.md
    reports = dashboard.collect_reports(cands)
    assert len(reports) == 1
    assert reports[0]["status"] == "PENDING"


def test_render_includes_speedup(tmp_path: Path) -> None:
    cands = tmp_path / "candidates"
    cands.mkdir()
    _make_candidate(cands, "01-polybench-2mm-zcurve", REPORT_WIN)
    reports = dashboard.collect_reports(cands)
    rendered = dashboard.render_dashboard(reports)
    assert "polybench-2mm-zcurve" in rendered
    assert "WIN" in rendered
    assert "1.25" in rendered
    assert "| id |" in rendered  # table header present


def test_render_with_no_candidates(tmp_path: Path) -> None:
    cands = tmp_path / "candidates"
    cands.mkdir()
    rendered = dashboard.render_dashboard(dashboard.collect_reports(cands))
    assert "0 candidate" in rendered
