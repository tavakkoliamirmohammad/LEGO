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
