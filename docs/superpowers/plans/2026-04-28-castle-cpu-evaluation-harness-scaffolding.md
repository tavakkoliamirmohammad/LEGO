# CASTLE CPU Evaluation Harness Scaffolding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the `evaluation/harness/` infrastructure for the CASTLE CPU layout-evaluation research harness, per the design spec at `docs/superpowers/specs/2026-04-28-castle-cpu-evaluation-harness-design.md`. Produces every file the scout and builder subagents will need before they're dispatched.

**Architecture:** A flat collection of Python utilities (lock, stats, verify, dashboard) under `evaluation/harness/`, three structured templates (`build_flags.json`, `result_schema.json`, `candidate_schema.md`), two long subagent prompts (`scout_prompt.md`, `builder_prompt.md`), wrapper scripts (`abort.sh`, `restart.sh`), an auto-captured `machine.md` fingerprint, and `evaluation/README.md`. The Python code is small (~30–100 lines per file), unit-tested with pytest, has zero LEGO imports, and depends only on stdlib + PyYAML.

**Tech Stack:** Python 3 stdlib (`fcntl`, `hashlib`, `re`, `statistics`, `pathlib`), PyYAML for parsing report YAML blocks, pytest for tests, Bash for wrapper scripts.

**Branch:** `eval/cpu-source-emission` (already exists, current HEAD `76468ac`).

**Pre-flight assumption:** `python/venv` is already created (per project convention) and PyYAML + pytest are importable from it. Activation command is `source venv/bin/activate` from the repo root.

---

## File Structure

```
evaluation/
├── README.md                       (Task 13)
├── references.bib                  (Task 13 — empty header only)
├── harness/
│   ├── __init__.py                 (Task 3)
│   ├── lock.py                     (Task 3)
│   ├── stats.py                    (Task 4)
│   ├── verify.py                   (Task 5)
│   ├── dashboard.py                (Task 6)
│   ├── build_flags.json            (Task 2)
│   ├── result_schema.json          (Task 7)
│   ├── candidate_schema.md         (Task 8)
│   ├── scout_prompt.md             (Task 9)
│   ├── builder_prompt.md           (Task 10)
│   ├── abort.sh                    (Task 11)
│   ├── restart.sh                  (Task 11)
│   ├── machine.md                  (Task 1)
│   └── tests/
│       ├── __init__.py
│       ├── test_lock.py            (Task 3)
│       ├── test_stats.py           (Task 4)
│       ├── test_verify.py          (Task 5)
│       └── test_dashboard.py       (Task 6)
├── candidates/
│   └── .gitkeep                    (Task 13)
└── (run-time only:)
    ├── .lock                       (created by lock.py at first acquire)
    ├── survey.md                   (created by scout)
    ├── survey_summary.md           (created by scout)
    └── dashboard.md                (created by dashboard.py)
```

---

## Task 1: Capture machine fingerprint

**Files:**
- Create: `evaluation/harness/machine.md`

- [ ] **Step 1: Make the directory**

```bash
mkdir -p evaluation/harness/tests
```

- [ ] **Step 2: Capture machine fingerprint**

Run this exact command from the repo root (it embeds `$(...)` substitutions at run time):

```bash
cat > evaluation/harness/machine.md <<'HEADER'
# Measurement Node Fingerprint

This file captures the node every measurement in this round runs on.
It is generated once at scaffolding time. If the node changes (e.g. the
Slurm allocation moves), the harness's pre-measurement check refuses to
run and asks for a re-fingerprint.

HEADER

cat >> evaluation/harness/machine.md <<EOF
## Identity

- captured_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- captured_by: $USER
- hostname: $(hostname)
- uname: $(uname -a)

## CPU

\`\`\`
$(lscpu)
\`\`\`

## NUMA

\`\`\`
$(numactl --hardware 2>/dev/null || echo "numactl unavailable")
\`\`\`

## Memory

\`\`\`
$(free -h)
\`\`\`

## Frequency / governor / turbo (as observed; no sudo to modify)

- governor (cpu0): $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
- intel_pstate.no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown)
- cpufreq.scaling_max_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo unknown)

## Compilers

- gcc: $(gcc --version 2>/dev/null | head -1 || echo unavailable)
- g++: $(g++ --version 2>/dev/null | head -1 || echo unavailable)
- gfortran: $(gfortran --version 2>/dev/null | head -1 || echo unavailable)
- rustc: $(rustc --version 2>/dev/null || echo unavailable)
- julia: $(julia --version 2>/dev/null || echo unavailable)

## GPU (recorded for opportunistic CUDA-C candidates only)

\`\`\`
$(nvidia-smi -L 2>/dev/null || echo "no NVIDIA GPU visible")
\`\`\`
EOF
```

- [ ] **Step 3: Inspect the result**

```bash
cat evaluation/harness/machine.md
```

Expected: a populated markdown file with non-empty CPU, NUMA, compiler, and GPU sections (some may say "unavailable" — that's fine and intentionally honest).

- [ ] **Step 4: Commit**

```bash
git add evaluation/harness/machine.md
git commit -m "eval: capture measurement-node fingerprint"
```

---

## Task 2: Fixed compiler flags

**Files:**
- Create: `evaluation/harness/build_flags.json`

Identical flags for baseline and LEGO version is non-negotiable per spec Section 6.

- [ ] **Step 1: Write the file**

```bash
cat > evaluation/harness/build_flags.json <<'EOF'
{
  "_doc": "Compiler flags applied identically to baseline and LEGO-rewritten kernels. The flags are intentionally the suite-default '-O3 -march=native -fopenmp' shape so that a layout-level win cannot be confused with a flag-tuning win.",
  "c": ["-O3", "-march=native", "-fopenmp"],
  "cpp": ["-O3", "-march=native", "-fopenmp", "-std=c++17"],
  "fortran": ["-O3", "-march=native", "-fopenmp"],
  "rust": ["-C", "opt-level=3", "-C", "target-cpu=native"],
  "julia": ["--check-bounds=no", "-O3"]
}
EOF
```

- [ ] **Step 2: Validate JSON**

```bash
python3 -c "import json; json.load(open('evaluation/harness/build_flags.json'))"
```

Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add evaluation/harness/build_flags.json
git commit -m "eval: pin compiler flags per language for build determinism"
```

---

## Task 3: lock.py — global mutex via flock

**Files:**
- Create: `evaluation/harness/__init__.py` (empty)
- Create: `evaluation/harness/tests/__init__.py` (empty)
- Create: `evaluation/harness/lock.py`
- Create: `evaluation/harness/tests/test_lock.py`

- [ ] **Step 1: Empty package markers**

```bash
touch evaluation/harness/__init__.py evaluation/harness/tests/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `evaluation/harness/tests/test_lock.py`:

```python
"""Test harness lock serializes critical sections across threads."""

import threading
import time
from pathlib import Path

import pytest

from evaluation.harness import lock as lock_mod


def test_acquire_serializes_concurrent_holders(tmp_path: Path) -> None:
    """No two threads ever hold the lock simultaneously."""
    lock_path = tmp_path / ".lock"
    inside_count = [0]
    max_inside = [0]
    state_guard = threading.Lock()

    def worker() -> None:
        with lock_mod.acquire(lock_path):
            with state_guard:
                inside_count[0] += 1
                if inside_count[0] > max_inside[0]:
                    max_inside[0] = inside_count[0]
            time.sleep(0.05)
            with state_guard:
                inside_count[0] -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_inside[0] == 1, (
        f"lock allowed {max_inside[0]} concurrent holders"
    )


def test_acquire_creates_lock_file_if_missing(tmp_path: Path) -> None:
    lock_path = tmp_path / "nested" / "dir" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_mod.acquire(lock_path):
        assert lock_path.exists()


def test_acquire_releases_on_exception(tmp_path: Path) -> None:
    lock_path = tmp_path / ".lock"
    with pytest.raises(RuntimeError):
        with lock_mod.acquire(lock_path):
            raise RuntimeError("boom")
    # Lock should be re-acquirable
    with lock_mod.acquire(lock_path):
        pass
```

- [ ] **Step 3: Run the test, expect failure**

```bash
source venv/bin/activate
cd /scratch/general/vast/u1419116/LEGO
PYTHONPATH=. pytest evaluation/harness/tests/test_lock.py -v
```

Expected: ImportError or ModuleNotFoundError on `evaluation.harness.lock`.

- [ ] **Step 4: Implement `lock.py`**

Create `evaluation/harness/lock.py`:

```python
"""File-based exclusive lock for serializing build+measure cycles.

Uses fcntl.flock(2). The lock is per-open-file-description, so each
acquire() opens a fresh fd and any concurrent holder (thread or process)
will block until release.

Default lock path: evaluation/.lock (one above this file).
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_LOCK_PATH = Path(__file__).resolve().parent.parent / ".lock"


@contextmanager
def acquire(lock_path: Path | None = None) -> Iterator[None]:
    """Acquire an exclusive flock on lock_path; release on context exit.

    Creates lock_path if missing. Releases the lock on normal exit and
    on exception.
    """
    path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    fd = open(path, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    finally:
        fd.close()
```

- [ ] **Step 5: Run the test, expect pass**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_lock.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add evaluation/harness/__init__.py evaluation/harness/tests/__init__.py \
        evaluation/harness/lock.py evaluation/harness/tests/test_lock.py
git commit -m "eval: add flock-based global mutex for serial measurement"
```

---

## Task 4: stats.py — median, IQR, classification

**Files:**
- Create: `evaluation/harness/stats.py`
- Create: `evaluation/harness/tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

Create `evaluation/harness/tests/test_stats.py`:

```python
"""Tests for median, IQR, and WIN/PARITY/LOSS classification."""

import pytest

from evaluation.harness import stats


def test_median_odd() -> None:
    assert stats.median([1, 2, 3, 4, 5]) == 3


def test_median_even() -> None:
    assert stats.median([1, 2, 3, 4]) == 2.5


def test_iqr_basic() -> None:
    # samples [1..8] -> Q1=2.5, Q3=6.5 (statistics.quantiles default)
    samples = [1, 2, 3, 4, 5, 6, 7, 8]
    iqr = stats.iqr(samples)
    assert 3.5 <= iqr <= 4.5  # tolerant of method differences


def test_classify_clear_win() -> None:
    assert stats.classify(100, 80) == "WIN"  # 1.25x


def test_classify_at_win_threshold() -> None:
    # speedup exactly 1.02 should be WIN (>= threshold, inclusive)
    assert stats.classify(102, 100) == "WIN"


def test_classify_just_under_win_threshold() -> None:
    # speedup ~1.0199 should be PARITY
    assert stats.classify(101.9, 100) == "PARITY"


def test_classify_clear_loss() -> None:
    assert stats.classify(80, 100) == "LOSS"


def test_classify_at_loss_threshold() -> None:
    # speedup exactly 0.98 should be LOSS (<= threshold, inclusive)
    assert stats.classify(98, 100) == "LOSS"


def test_classify_parity() -> None:
    assert stats.classify(100, 100) == "PARITY"
    assert stats.classify(99, 100) == "PARITY"
    assert stats.classify(101, 100) == "PARITY"


def test_classify_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        stats.classify(0, 100)
    with pytest.raises(ValueError):
        stats.classify(100, -1)


def test_speedup_basic() -> None:
    assert stats.speedup(100, 80) == 1.25
    assert stats.speedup(80, 100) == 0.8


def test_iqr_ratio_to_median() -> None:
    samples = [100, 100, 100, 100, 100, 100, 100, 100]
    assert stats.iqr_ratio(samples) == 0.0  # zero variance


def test_iqr_ratio_high_variance() -> None:
    samples = [50, 100, 150, 200]
    # IQR ~125, median 125, ratio ~1.0
    assert stats.iqr_ratio(samples) > 0.5
```

- [ ] **Step 2: Run the test, expect failure**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_stats.py -v
```

Expected: ImportError on `evaluation.harness.stats`.

- [ ] **Step 3: Implement `stats.py`**

Create `evaluation/harness/stats.py`:

```python
"""Median, IQR, and WIN/PARITY/LOSS speedup classification.

Verdict is gated only on the median ratio. IQR is recorded for sanity-
checking but does not gate. Defaults are the values agreed in the spec:

    EFFECT_THRESHOLD = 0.02   (i.e. 2 percent)
    HIGH_VARIANCE_RATIO = 0.20 (IQR / median; flag for re-run review)
"""

from __future__ import annotations

import statistics
from typing import Sequence

EFFECT_THRESHOLD: float = 0.02
HIGH_VARIANCE_RATIO: float = 0.20


def median(samples: Sequence[float]) -> float:
    return statistics.median(samples)


def iqr(samples: Sequence[float]) -> float:
    """Interquartile range Q3 - Q1 using the default exclusive method."""
    if len(samples) < 4:
        # Quantiles need at least 2 data points per quartile; fall back to 0.
        return 0.0
    qs = statistics.quantiles(samples, n=4)
    return qs[2] - qs[0]


def iqr_ratio(samples: Sequence[float]) -> float:
    """IQR divided by median; useful for variance-based re-run flagging."""
    m = median(samples)
    if m == 0:
        return float("inf")
    return iqr(samples) / abs(m)


def speedup(baseline_median: float, lego_median: float) -> float:
    """Speedup ratio. >1 means LEGO is faster."""
    if baseline_median <= 0 or lego_median <= 0:
        raise ValueError(
            f"non-positive medians: baseline={baseline_median} lego={lego_median}"
        )
    return baseline_median / lego_median


def classify(baseline_median: float, lego_median: float) -> str:
    """Return 'WIN', 'PARITY', or 'LOSS'.

    speedup = baseline / lego.
        speedup >= 1 + EFFECT_THRESHOLD -> WIN
        speedup <= 1 - EFFECT_THRESHOLD -> LOSS
        otherwise                       -> PARITY
    """
    s = speedup(baseline_median, lego_median)
    if s >= 1.0 + EFFECT_THRESHOLD:
        return "WIN"
    if s <= 1.0 - EFFECT_THRESHOLD:
        return "LOSS"
    return "PARITY"
```

- [ ] **Step 4: Run the test, expect pass**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_stats.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness/stats.py evaluation/harness/tests/test_stats.py
git commit -m "eval: add stats module (median, IQR, 2% WIN/LOSS classifier)"
```

---

## Task 5: verify.py — output comparator + GroupBy counter

**Files:**
- Create: `evaluation/harness/verify.py`
- Create: `evaluation/harness/tests/test_verify.py`

- [ ] **Step 1: Write the failing test**

Create `evaluation/harness/tests/test_verify.py`:

```python
"""Tests for output verification and GroupBy usage counting."""

from pathlib import Path

from evaluation.harness import verify


def test_sha256_match(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    assert verify.integer_outputs_match(a, b)


def test_sha256_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello there")
    assert not verify.integer_outputs_match(a, b)


def test_fp_within_tolerance() -> None:
    ref = [1.0, 2.0, 3.0, 4.0]
    cand = [1.0000001, 2.0, 3.0, 4.0000001]
    passed, max_abs, max_rel = verify.fp_outputs_within_tolerance(ref, cand)
    assert passed
    assert max_abs < 1e-6
    assert max_rel < 1e-6


def test_fp_exceeds_tolerance() -> None:
    ref = [1.0, 2.0]
    cand = [1.0, 2.5]
    passed, max_abs, max_rel = verify.fp_outputs_within_tolerance(ref, cand)
    assert not passed
    assert max_abs == 0.5


def test_fp_length_mismatch() -> None:
    ref = [1.0, 2.0, 3.0]
    cand = [1.0, 2.0]
    passed, max_abs, max_rel = verify.fp_outputs_within_tolerance(ref, cand)
    assert not passed


def test_count_groupby_zero(tmp_path: Path) -> None:
    src = tmp_path / "k.py"
    src.write_text(
        "from lego import OrderBy, TileBy, Row\n"
        "L = OrderBy(Row(M, N)).TileBy((4, 4), (16, 16))\n"
    )
    assert verify.count_groupby_uses(src) == []


def test_count_groupby_finds_uses(tmp_path: Path) -> None:
    src = tmp_path / "k.py"
    src.write_text(
        "from lego import GroupBy\n"
        "L = GroupBy([2, 2], inner)\n"
        "M = GroupBy([4, 4], other)\n"
    )
    occurrences = verify.count_groupby_uses(src)
    assert len(occurrences) == 2
    assert occurrences[0][0] == 2
    assert occurrences[1][0] == 3


def test_count_groupby_ignores_substring(tmp_path: Path) -> None:
    src = tmp_path / "k.py"
    # 'GroupByName' should NOT match 'GroupBy('
    src.write_text("x = GroupByName(1)\n")
    assert verify.count_groupby_uses(src) == []
```

- [ ] **Step 2: Run the test, expect failure**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_verify.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `verify.py`**

Create `evaluation/harness/verify.py`:

```python
"""Output-hashing reference comparator and GroupBy usage counter.

integer_outputs_match: byte-identical SHA-256 comparison.
fp_outputs_within_tolerance: element-wise relative-error check for
    floating-point output sequences.
count_groupby_uses: regex-counts occurrences of GroupBy( in a python
    source file, returns line numbers + text.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

GROUPBY_PATTERN = re.compile(r"\bGroupBy\s*\(")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def integer_outputs_match(reference: Path, candidate: Path) -> bool:
    return sha256_file(reference) == sha256_file(candidate)


def fp_outputs_within_tolerance(
    reference: Sequence[float],
    candidate: Sequence[float],
    rel_tol: float = 5e-6,
    abs_tol: float = 1e-12,
) -> tuple[bool, float, float]:
    """Element-wise comparison.

    Returns (passed, max_abs_error, max_rel_error).
    Passes when every elementwise relative error is <= rel_tol, ignoring
    elements with |reference| < abs_tol (where rel error is undefined).
    """
    if len(reference) != len(candidate):
        return False, float("inf"), float("inf")
    max_abs = 0.0
    max_rel = 0.0
    for r, c in zip(reference, candidate):
        d = abs(r - c)
        if d > max_abs:
            max_abs = d
        if abs(r) > abs_tol:
            rel = d / abs(r)
            if rel > max_rel:
                max_rel = rel
    passed = max_rel <= rel_tol
    return passed, max_abs, max_rel


def count_groupby_uses(python_source: Path) -> list[tuple[int, str]]:
    """Return [(line_number, line_text), ...] for each GroupBy( occurrence.

    Matches `\\bGroupBy\\s*\\(` so identifiers like `GroupByName(...)`
    do not match.
    """
    occurrences: list[tuple[int, str]] = []
    with open(python_source) as f:
        for lineno, line in enumerate(f, 1):
            if GROUPBY_PATTERN.search(line):
                occurrences.append((lineno, line.rstrip()))
    return occurrences
```

- [ ] **Step 4: Run the test, expect pass**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_verify.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness/verify.py evaluation/harness/tests/test_verify.py
git commit -m "eval: add verify module (sha256/fp comparison + GroupBy counter)"
```

---

## Task 6: dashboard.py — render evaluation/dashboard.md

**Files:**
- Create: `evaluation/harness/dashboard.py`
- Create: `evaluation/harness/tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Create `evaluation/harness/tests/test_dashboard.py`:

```python
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
```

- [ ] **Step 2: Run the test, expect failure**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_dashboard.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `dashboard.py`**

Create `evaluation/harness/dashboard.py`:

```python
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
```

- [ ] **Step 4: Run the test, expect pass**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/test_dashboard.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add evaluation/harness/dashboard.py evaluation/harness/tests/test_dashboard.py
git commit -m "eval: add dashboard generator (reads report.md YAML, renders status table)"
```

---

## Task 7: result_schema.json — JSON schema for raw timing files

**Files:**
- Create: `evaluation/harness/result_schema.json`

- [ ] **Step 1: Write the schema**

Create `evaluation/harness/result_schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "raw timing record",
  "description": "Schema for evaluation/candidates/<id>/raw/{baseline,lego}.json",
  "type": "object",
  "required": [
    "candidate_id",
    "version",
    "size",
    "build",
    "repro_setup",
    "warmup_iterations",
    "timed_iterations",
    "per_iteration_ns",
    "median_ns",
    "iqr_ns"
  ],
  "properties": {
    "candidate_id": {"type": "string"},
    "version": {"enum": ["baseline", "lego"]},
    "size": {"type": "string", "description": "small | medium | large or specific size tag"},
    "build": {
      "type": "object",
      "required": ["language", "compiler_version", "flags"],
      "properties": {
        "language": {"type": "string"},
        "compiler_version": {"type": "string"},
        "flags": {"type": "array", "items": {"type": "string"}}
      }
    },
    "repro_setup": {
      "type": "object",
      "description": "Best-effort reproducibility settings as actually applied. Fields are 'unknown' when not modifiable.",
      "required": ["taskset_cores", "numactl_membind", "governor", "turbo"],
      "properties": {
        "taskset_cores": {"type": "string"},
        "numactl_membind": {"type": "string"},
        "governor": {"type": "string"},
        "turbo": {"type": "string"}
      }
    },
    "warmup_iterations": {"type": "integer", "minimum": 0},
    "timed_iterations": {"type": "integer", "minimum": 1},
    "per_iteration_ns": {
      "type": "array",
      "items": {"type": "integer", "minimum": 0}
    },
    "median_ns": {"type": "integer", "minimum": 0},
    "iqr_ns": {"type": "integer", "minimum": 0},
    "machine_fingerprint_sha256": {
      "type": "string",
      "description": "sha256 of evaluation/harness/machine.md at the time of measurement, so we can detect node drift"
    }
  }
}
```

- [ ] **Step 2: Validate JSON**

```bash
python3 -c "import json; print(len(json.load(open('evaluation/harness/result_schema.json')).get('properties', {})))"
```

Expected: prints `9`.

- [ ] **Step 3: Commit**

```bash
git add evaluation/harness/result_schema.json
git commit -m "eval: add JSON schema for raw timing records"
```

---

## Task 8: candidate_schema.md — required structure for survey rows

**Files:**
- Create: `evaluation/harness/candidate_schema.md`

- [ ] **Step 1: Write the schema doc**

Create `evaluation/harness/candidate_schema.md`:

```markdown
# Candidate Schema

Every row in `evaluation/survey.md` is a YAML block with the structure
below. Any candidate missing any required field is dropped from the
survey before builders are dispatched.

## Required fields

```yaml
id: <slug>                       # also the candidates/<id>/ directory name;
                                 # convention: NN-<suite>-<kernel>-<layout>
suite: <name and version>        # e.g. "PolyBench/C 4.2.1"
kernel: <kernel name>            # specific kernel within the suite
upstream_url: <full URL>         # tarball, git ref, or release tag — exact
license: <SPDX id>               # accepted: MIT, BSD-2-Clause, BSD-3-Clause,
                                 # Apache-2.0, ISC, public-domain, CC0
                                 # copyleft (GPL/LGPL/AGPL) -> drop
language: <one of: c, cpp, fortran, rust, julia>
baseline:
  source_files: [<path within upstream>]
  build: "<exact build command>"
  threading: "<single-threaded | NN-thread OpenMP | etc>"
layout_trick: <short description>
layout_trick_citation: <bibtex key in references.bib>
why_compiler_cant: |
  <one paragraph naming the specific compiler pass that would be
   required (polyhedral, non-affine vectorization, etc.) and why it
   does NOT fire on naive code at the suite's baseline build flags>
lego_expressibility: |
  <Python sketch of the LEGO expression, using OrderBy + TileBy as
   building blocks; GroupBy permitted with one-line justification>
predicted_win:
  value: <"X.Yx" or "X.Yx – Z.Wx" or "unknown">
  source: <bibtex key OR "unknown">
  type: <"published" | "extrapolated" | "unknown">
power_of_two_restriction:
  baseline_assumes_pow2: <true | false>
  test_at_non_pow2_size: <true | false>
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl,
  governor, turbo as observed.
estimated_builder_effort: <"X-Y days">
risk_flags:
  - <one line per known risk>
```

## Drop rules

A candidate is dropped before the builder is dispatched if any of the
following are true:

1. Any required field is missing or empty.
2. License is not in the accepted list.
3. `language` is not one of the five CPU emission targets.
4. `predicted_win.source` is not "unknown" but the BibTeX key is not
   present in `references.bib`.
5. `lego_expressibility` requires a primitive outside `Row`, `Col`,
   `RegP`, `GenP`, `OrderBy`, `TileBy` (e.g. references the Tensor API
   directly).
6. `why_compiler_cant` does not name a specific compiler pass.

## Granularity rule

One row per **(kernel × layout-trick)** tuple, not per kernel. The same
kernel under different layouts becomes multiple candidates so the
paper's evaluation matrix can compare layouts head-to-head.

## Honesty rules

- No invented numbers. `predicted_win.type: unknown` is the right
  answer when no published number exists.
- Every BibTeX key must resolve to a real paper with DOI or arXiv ID.
- "Why the compiler can't recover this" must be specific enough that a
  reviewer can verify it (name the pass, name the flag, link the GCC
  bug if relevant).
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/harness/candidate_schema.md
git commit -m "eval: add candidate schema spec for scout output rows"
```

---

## Task 9: scout_prompt.md — exact prompt for the scout subagent

**Files:**
- Create: `evaluation/harness/scout_prompt.md`

- [ ] **Step 1: Write the prompt**

Create `evaluation/harness/scout_prompt.md`:

```markdown
# Scout Subagent Prompt

You are the **CASTLE evaluation scout**. You produce a literature- and
benchmark-suite-grounded list of candidate (kernel × layout-trick)
tuples for the CASTLE/TACO paper's Section 7.5 CPU evaluation. You do
NOT write code. You do NOT modify the LEGO repository. You do NOT run
benchmarks. You produce three files:

- `evaluation/survey.md`
- `evaluation/references.bib`
- `evaluation/survey_summary.md`

## Hard rules

1. **No invented numbers.** Any predicted speedup must cite a real
   published paper (DOI or arXiv) or be marked `type: unknown`.
2. **No invented papers.** Every BibTeX entry must point to a real
   paper. If you are not sure a paper exists, omit it and mark the
   citation `unknown`.
3. **Cite everything.** Every claim about a benchmark, transform, or
   measurement points to a BibTeX key in `references.bib`.
4. **No code, no LEGO repo edits, no benchmark runs.** You are
   research-only.

## What you are looking for

CPU benchmarks where a **layout-level** transform (not just an index-
arithmetic simplification) yields a real, published speedup over the
suite's as-shipped naive baseline, and where that layout is expressible
using only the LEGO primitives `Row`, `Col`, `RegP`, `GenP`, `OrderBy`,
`TileBy`. Prefer `OrderBy + TileBy`; `GroupBy` is allowed when needed.

### Eligible layout classes

1. Cache-oblivious recursive layouts (Z-Morton, Hilbert)
2. Multi-level cache-conscious tiling (register × L1 × L2 × L3)
3. Recursive bricking for stencils
4. Triangular / symmetric packing (RFP-style)
5. Skewed / shifted layouts (LU, NW, dynamic-programming wavefronts)
6. AoSoA / interleaved struct packing for vectorization
7. Block-cyclic distribution for thread-level locality
8. Padding to break power-of-two stride associativity conflicts
9. **Power-of-two-restricted optimizations applied at non-power-of-two
   sizes.** CASTLE has no pow-2 restriction; reproducing a pow-2-only
   published win at a non-pow-2 size is itself a paper-grade result.

### Out of scope

- Anything requiring new MLIR dialect ops or new lowering paths in
  CASTLE. If a candidate needs new compiler features, drop it.
- The Tensor API (`lego.ZCurve`, `lego.Swizzle`, `lego.Tiled`,
  `Batched`, `BlockCyclic`) and `torch.compile` integration. The path
  is `LEGO algebra → SymPy → MLIR → source emission` only.
- GPU-DSL hardware feature work.
- Distributed-memory layouts beyond what `BlockCyclic`-style
  expressions stand in for on a single node.

## Suites worth surveying

This list is not exhaustive — extend it as you find more sources, but
every suite you add must have a permissive license:

- PolyBench/C 4.2.1
- NAS Parallel Benchmarks (NPB) serial / OMP variants
- Rodinia (CPU subset)
- HPCC
- Mantevo proxy apps (HPCCG, MiniFE, MiniGhost, …)
- LULESH, MiniWeather, MiniSweep
- BrickLib stencil suite
- Tensor-contraction benchmarks (TCCG, TBLIS-style)
- Image-processing reference set (Halide/PolyMage style)
- Numerical recipes / dynamic programming (LU, NW, Smith-Waterman)

## Output: `evaluation/survey.md`

A markdown file containing one entry per candidate. Each entry has a
3–4 sentence prose intro followed by a YAML block conforming to
`evaluation/harness/candidate_schema.md`. Sort entries by:

1. Layout class (group same-class candidates together)
2. Within class, by predicted speedup magnitude (descending)

## Output: `evaluation/references.bib`

A standard BibTeX file. Every entry MUST have a `doi` or `archivePrefix +
eprint` field. No "personal communication" or unverifiable references.

Example entry:

```bibtex
@inproceedings{frigo1999cacheoblivious,
  author    = {Frigo, Matteo and Leiserson, Charles E. and Prokop, Harald and Ramachandran, Sridhar},
  title     = {Cache-Oblivious Algorithms},
  booktitle = {40th Annual Symposium on Foundations of Computer Science (FOCS '99)},
  year      = {1999},
  pages     = {285--297},
  doi       = {10.1109/SFFCS.1999.814600}
}
```

## Output: `evaluation/survey_summary.md`

A short summary listing:

- Layout classes represented (with count of candidates per class)
- Layout classes with no candidates and why
- Kernels that were considered and dropped, with the drop reason
- Total count of survivors

## Drop rules (apply before writing a row)

A candidate is dropped if any of:

1. Any required `candidate_schema.md` field is empty.
2. License is not in: MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, ISC,
   public-domain, CC0.
3. `language` is not in: c, cpp, fortran, rust, julia.
4. `predicted_win.source` references a BibTeX key you cannot back with
   a real paper (DOI or arXiv).
5. `lego_expressibility` requires anything outside `Row`, `Col`, `RegP`,
   `GenP`, `OrderBy`, `TileBy`.
6. `why_compiler_cant` is hand-wavy ("the compiler doesn't optimize
   this well") rather than specific (which pass, which flag).

## Granularity

One row per (kernel × layout-trick) tuple. Same kernel under multiple
layouts becomes multiple rows.

## Estimated count

Realistic survivor count is **30–50 candidates** spanning at least
six layout classes. No upper cap — return every candidate that passes
the drop rules. Triage happens after, not during.

## Format check before you finish

Before writing the final files, sanity-check:

- Does every yaml block parse with `python3 -c "import yaml; yaml.safe_load(open('survey.md').read())"`? (Test individually if needed.)
- Does every `layout_trick_citation` resolve to a key in
  `references.bib`?
- Does every BibTeX entry have a DOI or arXiv ID?

If any check fails, fix it before declaring done.
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/harness/scout_prompt.md
git commit -m "eval: add scout subagent prompt"
```

---

## Task 10: builder_prompt.md — exact prompt for builder subagents

**Files:**
- Create: `evaluation/harness/builder_prompt.md`

- [ ] **Step 1: Write the prompt**

Create `evaluation/harness/builder_prompt.md`:

```markdown
# Builder Subagent Prompt

You are a **CASTLE evaluation builder**. You own one candidate from the
survey. Your job is to port that one kernel from its as-shipped form to
a LEGO-rewritten form, verify correctness, measure both versions on the
locked node, and write a structured report.

You are NOT permitted to:

- Modify CASTLE source. `git diff main -- lib/ include/ python/lego/
  tools/ test/` must remain empty in your worktree.
- Compare against a baseline you retuned. Baseline is the suite's
  as-shipped form with the suite's documented build command (or, if it
  doesn't document one, the flags from `evaluation/harness/build_flags.json`).
- Pick problem sizes that flatter LEGO. Use the suite-defined small /
  medium / large sweep, or the closest equivalent if the suite doesn't
  ship a size sweep.
- Report a number without a corresponding `raw/*.json` file the number
  was computed from.
- Use any `lego.ZCurve`, `lego.Swizzle`, `Tensor`, or `torch.compile`
  API. The path is `LEGO algebra → SymPy → MLIR → source emission`.

## Inputs you are given

- `<candidate_id>`: your candidate's slug.
- `<candidate_yaml_block>`: the YAML block from `evaluation/survey.md`
  describing your candidate.
- `evaluation/harness/machine.md`: the locked node's fingerprint.
- `evaluation/harness/build_flags.json`: per-language compiler flags.
- `evaluation/harness/result_schema.json`: schema for raw timing files.

## Worktree setup

```bash
cd /scratch/general/vast/u1419116/LEGO
git worktree add ../LEGO-eval-<candidate_id> eval/cpu-source-emission
cd ../LEGO-eval-<candidate_id>
git checkout -b eval/cpu-<candidate_id>
source venv/bin/activate
mkdir -p evaluation/candidates/<candidate_id>/{upstream,raw}
```

## Build–measure–iterate loop

For each layout attempt (max 8 per candidate):

1. **Vendor baseline** under `evaluation/candidates/<candidate_id>/upstream/`.
   Untouched; use git clone / curl / wget. Verify license matches the
   survey's `license` field.
2. **Build baseline** with the suite's documented command (or the
   matching flags from `build_flags.json`). Record exact command in
   `run.sh`. Run baseline and confirm it produces the suite's reference
   output.
3. **Author `kernel_lego.py`** using only `Row`, `Col`, `RegP`, `GenP`,
   `OrderBy`, `TileBy`. Prefer `OrderBy + TileBy`. If `GroupBy` is
   needed, record each use in `report.md`'s `groupby_usage` field with
   a one-sentence justification.
4. **Generate source** via `lego.<lang>_gen.generate(...)` and splice
   it into the suite's kernel skeleton, replacing the index-arithmetic
   block. Data structures, I/O, and the timing harness stay unchanged.
5. **Verification gate.** Run baseline + LEGO on the same input. For
   integer kernels compare via `verify.integer_outputs_match`. For FP
   kernels use `verify.fp_outputs_within_tolerance` (rel_tol=5e-6 for
   FP32, 1e-12 for FP64). On failure, write
   `status: DROPPED-verification` and STOP — no timing.
6. **Acquire the global lock** via `python -c "from
   evaluation.harness.lock import acquire; ..."` or `flock
   evaluation/.lock <command>`. Hold for the full build + measurement
   cycle.
7. **Measure.**
   - Apply `taskset -c <core>` for thread pinning. (Always works.)
   - Apply `numactl --membind`, governor=performance, turbo-disable IF
     available; record the actual state in each raw json's
     `repro_setup` field. If unavailable, record `unknown` honestly.
   - 25 warmup → 100 timed iterations.
   - Sweep at least 3 problem sizes (small / medium / large).
   - Same protocol for baseline and LEGO.
   - Output `raw/baseline.json` and `raw/lego.json` conforming to
     `result_schema.json`.
8. **Release the lock.**
9. **Classify** with `evaluation.harness.stats.classify(baseline_median,
   lego_median)` per size. The candidate's overall status is the best
   class across sizes (WIN > PARITY > LOSS).
10. **Iterate.** If LOSS or unconvincing PARITY, try other layouts (Z-
    Morton, RFP, deeper tiling, different tile sizes). Each attempt is
    one entry in `layouts_tried`. Hard cap: 8 attempts.
11. **Write `report.md`** per the schema in spec Section 9.
12. **Commit and push** branch `eval/cpu-<candidate_id>`. Do NOT open a
    PR.

## Verification checklist before declaring done

- [ ] `git diff main -- lib/ include/ python/lego/ tools/ test/` is empty.
- [ ] `kernel_lego.py` exists and imports only `lego` primitives.
- [ ] `raw/baseline.json` and `raw/lego.json` validate against
      `result_schema.json`.
- [ ] Verification gate passed and the result is logged in
      `verify.log`.
- [ ] `report.md` contains a YAML block parseable by
      `evaluation.harness.dashboard.parse_report`.
- [ ] Every speedup number in `report.md` references a number in `raw/`.
- [ ] Branch `eval/cpu-<candidate_id>` is pushed.

## Honesty clauses

- If `repro_setup.turbo == "unknown"`, say so in `report.md`. Do not
  fake a frequency-locked claim.
- If you cannot beat the baseline after 8 layout attempts, ship the
  best LOSS / PARITY result honestly. Negative results are paper-
  strengthening when honest.
- If you discover the candidate cannot be expressed without modifying
  CASTLE source, write `status: DROPPED-needs-lowering` with a
  one-paragraph explanation in `notes` and stop. Do not modify CASTLE.

## Cleanup

```bash
cd /scratch/general/vast/u1419116/LEGO
git worktree remove ../LEGO-eval-<candidate_id>
```

(Worktree removal is best-effort; the orchestrator can clean stragglers
later.)
```

- [ ] **Step 2: Commit**

```bash
git add evaluation/harness/builder_prompt.md
git commit -m "eval: add builder subagent prompt"
```

---

## Task 11: abort.sh and restart.sh wrapper scripts

**Files:**
- Create: `evaluation/harness/abort.sh`
- Create: `evaluation/harness/restart.sh`

- [ ] **Step 1: Write `abort.sh`**

```bash
cat > evaluation/harness/abort.sh <<'EOF'
#!/usr/bin/env bash
# abort.sh <candidate_id>
#
# Best-effort abort of a builder agent for the given candidate.
# Cleanly removes the worktree and the candidate's raw/ directory so
# restart.sh can re-dispatch from a clean slate. The orchestrator
# (Claude) is responsible for actually stopping the running subagent
# via TaskStop; this script just cleans the filesystem state.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <candidate_id>" >&2
  exit 1
fi

cid="$1"
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
worktree="${repo_root%/*}/LEGO-eval-${cid}"

if git -C "${repo_root}" worktree list | grep -q "${worktree}"; then
  echo "removing worktree ${worktree}"
  git -C "${repo_root}" worktree remove --force "${worktree}" || true
fi

cand_dir="${repo_root}/evaluation/candidates/${cid}"
if [[ -d "${cand_dir}/raw" ]]; then
  echo "clearing raw/ for ${cid}"
  rm -rf "${cand_dir}/raw"
fi

echo "abort complete for ${cid}"
EOF
chmod +x evaluation/harness/abort.sh
```

- [ ] **Step 2: Write `restart.sh`**

```bash
cat > evaluation/harness/restart.sh <<'EOF'
#!/usr/bin/env bash
# restart.sh <candidate_id>
#
# Calls abort.sh, then prints a one-liner the orchestrator can use to
# re-dispatch the builder. Does NOT itself dispatch a subagent (that's
# the orchestrator's job).

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <candidate_id>" >&2
  exit 1
fi

cid="$1"
here="$(cd "$(dirname "$0")" && pwd)"

bash "${here}/abort.sh" "${cid}"

cat <<MSG
Re-dispatch instructions:
  - Read evaluation/candidates/${cid}/ for the candidate's YAML block.
  - Dispatch a fresh builder subagent with the prompt from
    evaluation/harness/builder_prompt.md, parameterized on
    candidate_id=${cid}.
MSG
EOF
chmod +x evaluation/harness/restart.sh
```

- [ ] **Step 3: Smoke-test the scripts**

```bash
evaluation/harness/abort.sh nonexistent-candidate-id
evaluation/harness/restart.sh nonexistent-candidate-id
```

Expected: scripts run without error, print sensible messages.

- [ ] **Step 4: Commit**

```bash
git add evaluation/harness/abort.sh evaluation/harness/restart.sh
git commit -m "eval: add abort/restart wrapper scripts for builder lifecycle"
```

---

## Task 12: Run all unit tests one more time as a smoke check

- [ ] **Step 1: Run the full harness test suite**

```bash
cd /scratch/general/vast/u1419116/LEGO
source venv/bin/activate
PYTHONPATH=. pytest evaluation/harness/tests/ -v
```

Expected: all tests pass (lock × 3, stats × 11, verify × 8, dashboard × 4 = 26 passed).

- [ ] **Step 2: Render the dashboard against the (empty) candidates dir**

```bash
mkdir -p evaluation/candidates
python -m evaluation.harness.dashboard
cat evaluation/dashboard.md
```

Expected: a markdown file saying "Auto-generated from 0 candidate report(s)." with an empty table.

---

## Task 13: README.md, references.bib, and candidates/.gitkeep

**Files:**
- Create: `evaluation/README.md`
- Create: `evaluation/references.bib`
- Create: `evaluation/candidates/.gitkeep`

- [ ] **Step 1: Write `evaluation/README.md`**

```bash
cat > evaluation/README.md <<'EOF'
# CASTLE CPU Evaluation

Research harness for the CASTLE/TACO paper's Section 7.5 CPU evaluation.
Detailed design lives in
`docs/superpowers/specs/2026-04-28-castle-cpu-evaluation-harness-design.md`.

## What is here

- `harness/` — prompts, schemas, and Python utilities used by the scout
  and builder subagents.
- `references.bib` — BibTeX, owned by the scout, appended to by builders.
- `survey.md` — scout output (one entry per candidate). Created by the
  scout subagent; not present until Step 2 of the orchestration.
- `survey_summary.md` — scout drop-list summary.
- `dashboard.md` — auto-regenerated status table during builder runs.
- `candidates/` — one directory per candidate, owned by one builder.

## Workflow at a glance

1. Orchestrator scaffolds `harness/` (this state).
2. Scout subagent surveys benchmarks, writes `survey.md` and
   `references.bib`.
3. One builder subagent per surviving survey row, in its own git
   worktree on branch `eval/cpu-<id>`, runs build → verify → measure
   → classify under the global mutex `evaluation/.lock`.
4. Orchestrator distills surviving builder reports into the paper's
   Section 7.5 prose.

## How to reproduce one round

```bash
# 1. Capture the node fingerprint at scaffolding time
bash evaluation/harness/abort.sh dummy   # smoke check
PYTHONPATH=. pytest evaluation/harness/tests/ -v

# 2. Run the scout. (Orchestrator dispatches a subagent with
#    evaluation/harness/scout_prompt.md.)

# 3. For each row in survey.md, dispatch a builder subagent with
#    evaluation/harness/builder_prompt.md and the row's YAML block.

# 4. Regenerate dashboard.md as builders complete:
python -m evaluation.harness.dashboard

# 5. Distill survivor reports into CASTLE-tex Section 7.5.
```

## Conventions

- LEGO layouts in user-facing code prefer `OrderBy` + `TileBy`.
  `GroupBy` is allowed when needed but each occurrence must carry a
  one-sentence justification in the candidate's `report.md`.
- Compiler flags are pinned per-language in
  `harness/build_flags.json` and applied identically to baseline and
  LEGO versions.
- Verdict thresholds: speedup ≥ 1.02× → WIN, ≤ 0.98× → LOSS, else
  PARITY. Tunable via `harness/stats.py:EFFECT_THRESHOLD`.
- Single global mutex via `flock(evaluation/.lock)` serializes all
  build + measurement cycles to keep timings honest on the shared node.
EOF
```

- [ ] **Step 2: Write the BibTeX skeleton**

```bash
cat > evaluation/references.bib <<'EOF'
% References cited by the CASTLE CPU evaluation harness.
% Owned by the scout subagent; appended to by builders when they cite
% additional sources during their layout attempts. Every entry MUST
% have a DOI or arXiv eprint field. No "personal communication" or
% unverifiable references.
EOF
```

- [ ] **Step 3: Empty candidates/ marker**

```bash
mkdir -p evaluation/candidates
touch evaluation/candidates/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add evaluation/README.md evaluation/references.bib evaluation/candidates/.gitkeep
git commit -m "eval: add README, empty references.bib skeleton, and candidates/ marker"
```

---

## Task 14: Final tree review and handoff

- [ ] **Step 1: Print the full scaffolded tree**

```bash
find evaluation -type f -not -path '*/__pycache__/*' | sort
```

Expected output (order doesn't matter, must include all of these):

```
evaluation/README.md
evaluation/candidates/.gitkeep
evaluation/harness/__init__.py
evaluation/harness/abort.sh
evaluation/harness/build_flags.json
evaluation/harness/builder_prompt.md
evaluation/harness/candidate_schema.md
evaluation/harness/dashboard.py
evaluation/harness/lock.py
evaluation/harness/machine.md
evaluation/harness/restart.sh
evaluation/harness/result_schema.json
evaluation/harness/scout_prompt.md
evaluation/harness/stats.py
evaluation/harness/tests/__init__.py
evaluation/harness/tests/test_dashboard.py
evaluation/harness/tests/test_lock.py
evaluation/harness/tests/test_stats.py
evaluation/harness/tests/test_verify.py
evaluation/harness/verify.py
evaluation/references.bib
```

- [ ] **Step 2: Final test run**

```bash
PYTHONPATH=. pytest evaluation/harness/tests/ -v
```

Expected: 26 passed.

- [ ] **Step 3: Commit log review**

```bash
git log --oneline main..HEAD
```

Expected: ~13 commits on `eval/cpu-source-emission` describing each task above (plus the earlier spec commits).

- [ ] **Step 4: Stop and hand back to user**

The harness is now scaffolded. Surface to the user:

- The full file tree from Step 1.
- The pytest output from Step 2.
- The commit log from Step 3.
- A reminder that the next step is dispatching the scout subagent
  (Step 2 of the orchestration in spec Section 12), which the user
  should approve before it runs.

Do NOT dispatch the scout in this plan. The plan terminates here.

---

## Self-review checklist (the plan author runs this; not a separate task)

After writing the plan, the plan author looked at the spec with fresh
eyes and confirmed:

- **Spec coverage:** Every section of the design spec is implemented by
  one or more tasks. Section 12's Step 1 ("Orchestrator scaffolds
  harness/") is exactly Tasks 1–14 of this plan.
- **Placeholder scan:** No "TBD", "TODO", "implement later", or
  "similar to Task N" strings.
- **Type consistency:** Function names, module paths, and YAML keys
  are consistent across tasks. `evaluation.harness.lock.acquire`,
  `evaluation.harness.stats.classify`, `evaluation.harness.verify
  .integer_outputs_match`, `evaluation.harness.dashboard.parse_report`
  are referenced consistently across the prompts in Tasks 9–10 and the
  test code in Tasks 3–6.
- **Scope check:** This plan ONLY scaffolds Step 1. It deliberately
  does not dispatch the scout or any builder. Those are separate
  follow-on actions gated on user review.
