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
    """i*cols + (j ^ ((i >> shift) & mask)), holes: shift, mask."""
    rows: int
    cols: int

    def holes(self) -> Dict[str, Tuple[int, int]]:
        return {"shift": (0, max(1, math.ceil(math.log2(max(self.rows, 2))))),
                "mask": (0, (1 << max(1, math.ceil(math.log2(max(self.cols, 2))))) - 1)}

    def flat_z3(self, i: int, j: int, h) -> z3.BitVecRef:
        return _bv(i * self.cols) + (_bv(j) ^ (z3.LShR(_bv(i), h["shift"]) & h["mask"]))

    def side_constraints(self, h) -> List[z3.BoolRef]:
        # Row-local bijectivity: every swizzled column stays inside [0, cols).
        cs = []
        for i in range(self.rows):
            for j in range(self.cols):
                cs.append(z3.ULT(_bv(j) ^ (z3.LShR(_bv(i), h["shift"]) & h["mask"]), _bv(self.cols)))
        return cs

    def instantiate(self, v) -> XorSwizzle:
        return XorSwizzle(self.rows, self.cols, shift=v["shift"], mask=v["mask"])


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
    """i*cols + (j + r*i) mod cols, hole: r (linear skewing scheme)."""
    rows: int
    cols: int

    def holes(self):
        return {"r": (0, self.cols - 1)}

    def flat_z3(self, i, j, h):
        return _bv(i * self.cols) + z3.URem(_bv(j) + h["r"] * _bv(i), _bv(self.cols))

    def side_constraints(self, h):
        return []

    def instantiate(self, v) -> Rotation:
        return Rotation(self.rows, self.cols, r=v["r"])
