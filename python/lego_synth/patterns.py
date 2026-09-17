"""Warp access patterns: for each instruction, the logical (i, j) touched by each lane.

Tiles wider or taller than a warp take several instructions per row or column.
Lanes past the tile edge are folded back onto in-range coordinates (a broadcast
of an element some other lane already touches), which never adds a conflict and
never hides one, since every real coordinate stays represented.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class AccessPattern:
    name: str
    n_instr: int
    warp: int
    coords: Callable[[int], Tuple[np.ndarray, np.ndarray]]  # instr -> (i[lane], j[lane]) start element
    access_elems: int = 1   # consecutive elements each lane touches per instruction

    def access_width(self, elem_bytes: int) -> int:
        return self.access_elems * elem_bytes


def row_vector(rows: int, cols: int, warp: int, vec: int = 1, dense: bool = False) -> AccessPattern:
    """Lanes along rows, `vec` consecutive elements per lane.

    dense=False (masked): one instruction per (row, chunk of warp*vec columns); lanes
      past the row's end are idle, modeled as a broadcast of an in-range element.
      This is a warp per tile row with predicated-off lanes.
    dense=True: instruction k covers the tile's row-major elements
      [k*warp*vec, (k+1)*warp*vec), so a short row shares an instruction with the
      next rows. This is a (cols/vec)-wide thread block with x-major numbering.
    """
    assert cols % vec == 0, "vector width must divide the row"
    lanes = np.arange(warp)
    if dense:
        total, per_instr = rows * cols, warp * vec

        def coords(k):
            idx = (k * per_instr + lanes * vec) % total
            return idx // cols, idx % cols

        return AccessPattern("row_vector", math.ceil(total / per_instr), warp, coords, access_elems=vec)

    per_row = math.ceil(cols / (warp * vec))

    def coords(k):
        r, chunk = divmod(k, per_row)
        c0 = chunk * warp * vec
        active = min(warp, (cols - c0) // vec)
        return np.full(warp, r), c0 + (lanes % active) * vec

    return AccessPattern("row_vector", rows * per_row, warp, coords, access_elems=vec)


def col_vector(rows: int, cols: int, warp: int, dense: bool = False) -> AccessPattern:
    """Lanes down columns, one element per lane; `dense` as in `row_vector` but in
    column-major order (a short column shares an instruction with the next columns)."""
    lanes = np.arange(warp)
    if dense:
        total = rows * cols

        def coords(k):
            idx = (k * warp + lanes) % total
            return idx % rows, idx // rows

        return AccessPattern("col_vector", math.ceil(total / warp), warp, coords)

    per_col = math.ceil(rows / warp)

    def coords(k):
        c, chunk = divmod(k, per_col)
        r0 = chunk * warp
        active = min(warp, rows - r0)
        return r0 + (lanes % active), np.full(warp, c)

    return AccessPattern("col_vector", cols * per_col, warp, coords)


def ldmatrix_x4(rows: int, cols: int) -> AccessPattern:
    """ldmatrix.x4 over a tile of 2-byte elements, one instruction per 16x16 block in
    row-major block order. Lane t supplies the 16-byte row address of row (t % 8) of
    matrix (t // 8); matrices tile the block as [rows 0-7, cols 0-7], [rows 8-15, cols 0-7],
    [rows 0-7, cols 8-15], [rows 8-15, cols 8-15].
    """
    assert rows % 16 == 0 and cols % 16 == 0
    lanes = np.arange(32)
    m, r = lanes // 8, lanes % 8
    di, dj = r + 8 * (m & 1), 8 * (m >> 1)
    blocks_per_row = cols // 16

    def coords(b):
        r0, c0 = 16 * (b // blocks_per_row), 16 * (b % blocks_per_row)
        return r0 + di, np.full(32, c0) + dj

    return AccessPattern("ldmatrix_x4", (rows // 16) * blocks_per_row, 32, coords, access_elems=8)


def mma_m16n8k16_a() -> AccessPattern:
    """mma.sync.m16n8k16 A fragment (row-major 16x16, 16-bit) loaded directly as four
    32-bit register pairs: pair k of lane t holds elements (groupID + 8*(k&1), tid*2 + 8*(k>>1))
    and the next column, with groupID = t>>2, tid = t&3.
    """
    lanes = np.arange(32)
    group, tid = lanes >> 2, lanes & 3

    def coords(k):
        return group + 8 * (k & 1), tid * 2 + 8 * (k >> 1)

    return AccessPattern("mma_m16n8k16_a", 4, 32, coords, access_elems=2)


def mma_a_direct(rows: int, cols: int) -> AccessPattern:
    """Direct (no ldmatrix) A-fragment loads over a tile: four 32-bit loads per 16x16 block,
    blocks in row-major order."""
    assert rows % 16 == 0 and cols % 16 == 0
    frag = mma_m16n8k16_a()
    blocks_per_row = cols // 16

    def coords(k):
        b, pair = divmod(k, 4)
        r0, c0 = 16 * (b // blocks_per_row), 16 * (b % blocks_per_row)
        i, j = frag.coords(pair)
        return r0 + i, c0 + j

    return AccessPattern("mma_a_direct", (rows // 16) * blocks_per_row * 4, 32, coords, access_elems=2)
