"""Concrete 2-D layout families: logical (i, j) -> element index, vectorized."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RowMajor:
    rows: int
    cols: int
    pad: int = 0

    def flat(self, i, j):
        return np.asarray(i) * (self.cols + self.pad) + np.asarray(j)

    @property
    def footprint(self) -> int:
        return self.rows * (self.cols + self.pad)


@dataclass(frozen=True)
class XorSwizzle:
    """Row-major with the column XOR-ed by a shifted, masked row index.

    `unit` is the swizzle granularity in elements (a power of two): the XOR acts on
    the chunk index j // unit and leaves j % unit untouched, so `unit`-element
    vectors stay contiguous (unit=8 for 16-byte chunks of 2-byte elements).
    """
    rows: int
    cols: int
    shift: int = 0
    mask: int = 0
    unit: int = 1

    def flat(self, i, j):
        i = np.asarray(i); j = np.asarray(j)
        return i * self.cols + (j ^ (((i >> self.shift) & self.mask) * self.unit))

    @property
    def footprint(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class Rotation:
    """Row-major with the column rotated by r * row (a linear skewing scheme)."""
    rows: int
    cols: int
    r: int = 1

    def flat(self, i, j):
        i = np.asarray(i); j = np.asarray(j)
        return i * self.cols + ((j + self.r * i) % self.cols)

    @property
    def footprint(self) -> int:
        return self.rows * self.cols
