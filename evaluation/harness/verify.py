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
