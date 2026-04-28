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
