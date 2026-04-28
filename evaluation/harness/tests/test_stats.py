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
