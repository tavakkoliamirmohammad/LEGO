"""Warp access patterns: for each instruction, the logical (i, j) touched by each lane."""
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


@dataclass(frozen=True)
class AccessPattern:
    name: str
    n_instr: int
    warp: int
    coords: Callable[[int], Tuple[np.ndarray, np.ndarray]]  # instr -> (i[lane], j[lane])


def row_vector(rows: int, cols: int, warp: int) -> AccessPattern:
    """Instruction k: lane t touches (k, t). Consecutive lanes along a row."""
    lanes = np.arange(warp)
    return AccessPattern("row_vector", rows, warp,
                         lambda k: (np.full(warp, k), lanes % cols))


def col_vector(rows: int, cols: int, warp: int) -> AccessPattern:
    """Instruction k: lane t touches (t, k). Consecutive lanes down a column."""
    lanes = np.arange(warp)
    return AccessPattern("col_vector", cols, warp,
                         lambda k: (lanes % rows, np.full(warp, k)))
