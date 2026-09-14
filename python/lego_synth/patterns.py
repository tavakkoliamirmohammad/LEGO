"""Warp access patterns: for each instruction, the logical (i, j) touched by each lane.

Tiles wider or taller than a warp take several instructions per row or column.
Lanes past the tile edge are folded back onto in-range coordinates (a broadcast
of an element some other lane already touches), which never adds a conflict and
never hides one, since every real coordinate stays represented.
"""
from dataclasses import dataclass
from typing import Callable, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class AccessPattern:
    name: str
    n_instr: int
    warp: int
    coords: Callable[[int], Tuple[np.ndarray, np.ndarray]]  # instr -> (i[lane], j[lane])


def row_vector(rows: int, cols: int, warp: int) -> AccessPattern:
    """Consecutive lanes along a row: instruction (r, c0) has lane t at (r, c0 + t)."""
    lanes = np.arange(warp)
    per_row = math.ceil(cols / warp)

    def coords(k):
        r, chunk = divmod(k, per_row)
        c0 = chunk * warp
        width = min(warp, cols - c0)
        return np.full(warp, r), c0 + (lanes % width)

    return AccessPattern("row_vector", rows * per_row, warp, coords)


def col_vector(rows: int, cols: int, warp: int) -> AccessPattern:
    """Consecutive lanes down a column: instruction (c, r0) has lane t at (r0 + t, c)."""
    lanes = np.arange(warp)
    per_col = math.ceil(rows / warp)

    def coords(k):
        c, chunk = divmod(k, per_col)
        r0 = chunk * warp
        height = min(warp, rows - r0)
        return r0 + (lanes % height), np.full(warp, c)

    return AccessPattern("col_vector", cols * per_col, warp, coords)
