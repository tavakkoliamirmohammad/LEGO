"""Layout templates with holes: structural choices fixed, numeric parameters left to the solver."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import z3

from .layouts import RowMajor, XorSwizzle, Rotation

BV = 32  # bit-vector width for all index arithmetic


def _bv(v: int) -> z3.BitVecRef:
    return z3.BitVecVal(v, BV)


@dataclass(frozen=True)
class XorTemplate:
    """i*cols + (j ^ (((i >> shift) & mask) * unit)), holes: shift, mask.

    `unit` (elements, power of two) is the swizzle granularity: vectors of `unit`
    elements stay contiguous. The mask ranges over chunk indices (cols / unit).
    """
    rows: int
    cols: int
    unit: int = 1

    def holes(self) -> Dict[str, Tuple[int, int]]:
        chunks = max(1, math.ceil(self.cols / self.unit))
        return {"shift": (0, max(1, math.ceil(math.log2(max(self.rows, 2))))),
                "mask": (0, (1 << max(1, math.ceil(math.log2(max(chunks, 2))))) - 1)}

    def _col(self, i: int, j: int, h) -> z3.BitVecRef:
        return _bv(j) ^ ((z3.LShR(_bv(i), h["shift"]) & h["mask"]) * _bv(self.unit))

    def flat_z3(self, i: int, j: int, h) -> z3.BitVecRef:
        return _bv(i * self.cols) + self._col(i, j, h)

    def side_constraints(self, h) -> List[z3.BoolRef]:
        # Row-local bijectivity: every swizzled column stays inside [0, cols).
        # Only the chunk starts matter (the XOR never touches the low unit bits).
        cs = []
        for i in range(self.rows):
            for j in range(0, self.cols, self.unit):
                cs.append(z3.ULT(self._col(i, j, h), _bv(self.cols)))
        return cs

    def instantiate(self, v) -> XorSwizzle:
        return XorSwizzle(self.rows, self.cols, shift=v["shift"], mask=v["mask"], unit=self.unit)


@dataclass(frozen=True)
class PaddedTemplate:
    """i*(cols+pad) + j, hole: pad."""
    rows: int
    cols: int
    max_pad: int = 4

    def holes(self):
        return {"pad": (0, self.max_pad)}

    def flat_z3(self, i, j, h):
        return _bv(i) * (_bv(self.cols) + h["pad"]) + _bv(j)

    def side_constraints(self, h):
        return []

    def instantiate(self, v) -> RowMajor:
        return RowMajor(self.rows, self.cols, pad=v["pad"])


@dataclass(frozen=True)
class RotationTemplate:
    """i*cols + rotate(chunk(j), r*i) * unit + j % unit, hole: r (linear skewing scheme)."""
    rows: int
    cols: int
    unit: int = 1

    def holes(self):
        return {"r": (0, self.cols // self.unit - 1)}

    def flat_z3(self, i, j, h):
        nchunks = self.cols // self.unit
        chunk, low = divmod(j, self.unit)
        rotated = z3.URem(_bv(chunk) + h["r"] * _bv(i), _bv(nchunks))
        return _bv(i * self.cols) + rotated * _bv(self.unit) + _bv(low)

    def side_constraints(self, h):
        return []

    def instantiate(self, v) -> Rotation:
        return Rotation(self.rows, self.cols, r=v["r"], unit=self.unit)
