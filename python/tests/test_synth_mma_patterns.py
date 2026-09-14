"""Tests for instruction-imposed access patterns: ldmatrix and mma.sync fragments as layouts."""
import numpy as np

from lego_synth.patterns import ldmatrix_x4, mma_m16n8k16_a
from lego_synth.layouts import RowMajor
from lego_synth.eval_np import conflicts
from lego_synth.machine import ampere


def test_ldmatrix_x4_covers_four_8x8_b16_matrices_with_one_16_byte_row_per_lane():
    # A 16x16 tile of 2-byte elements: lanes 0-7 address rows 0-7 of matrix 0 (cols 0-7),
    # lanes 8-15 matrix 1 (rows 8-15, cols 0-7), 16-23 matrix 2 (rows 0-7, cols 8-15), 24-31 matrix 3.
    p = ldmatrix_x4(rows=16, cols=16)
    assert p.n_instr == 1
    i, j = p.coords(0)
    assert i.tolist() == list(range(8)) + list(range(8, 16)) + list(range(8)) + list(range(8, 16))
    assert j.tolist() == [0] * 16 + [8] * 16
    assert p.access_elems == 8         # each lane reads one 16-byte row segment (8 x b16)


def test_ldmatrix_on_row_major_16x16_b16_tile_has_8_way_conflicts_and_swizzle_removes_them():
    p = ldmatrix_x4(rows=16, cols=16)
    naive = RowMajor(16, 16)       # 32-byte rows: rows r and r+4 share banks -> conflicts
    assert conflicts(naive, p, elem_bytes=2, machine=ampere()) > 0
    # Padding each row to 24 elements (48 B) spreads the 16 B segments across banks.
    assert conflicts(RowMajor(16, 16, pad=8), p, elem_bytes=2, machine=ampere()) == 0


def test_mma_m16n8k16_a_fragment_loads_register_pairs_at_the_documented_coordinates():
    # PTX ISA: groupID = lane >> 2, tid = lane & 3. Registers hold b16 pairs (one 32-bit load each):
    # {a0,a1}: (groupID, tid*2); {a2,a3}: (groupID+8, tid*2); {a4,a5}: (groupID, tid*2+8); {a6,a7}: (groupID+8, tid*2+8)
    p = mma_m16n8k16_a()
    assert p.n_instr == 4 and p.access_elems == 2
    i, j = p.coords(0); assert i[5] == 5 >> 2 and j[5] == (5 & 3) * 2
    i, j = p.coords(1); assert i[5] == (5 >> 2) + 8 and j[5] == (5 & 3) * 2
    i, j = p.coords(2); assert i[5] == 5 >> 2 and j[5] == (5 & 3) * 2 + 8
    i, j = p.coords(3); assert i[5] == (5 >> 2) + 8 and j[5] == (5 & 3) * 2 + 8


def test_mma_a_direct_over_a_tile_issues_four_loads_per_16x16_block():
    from lego_synth.patterns import mma_a_direct
    p = mma_a_direct(rows=32, cols=48)
    assert p.n_instr == 2 * 3 * 4
    i, j = p.coords(4 * 4 + 1)     # block (row 1, col 1), register pair 1
    assert i.min() == 16 + 8 and j.min() == 16


def test_row_major_b16_tile_has_no_conflicts_for_direct_fragment_loads_at_width_64():
    # Each 32-bit load: 8 groups x 4 lanes; lanes of a group read consecutive words of one row,
    # groups read rows 8 apart... a 128 B row stride maps rows to the same banks -> conflicts.
    from lego_synth.patterns import mma_a_direct
    from lego_synth.layouts import RowMajor
    from lego_synth.eval_np import conflicts
    from lego_synth.machine import ampere
    assert conflicts(RowMajor(16, 64), mma_a_direct(16, 64), elem_bytes=2, machine=ampere()) > 0
    assert conflicts(RowMajor(16, 72), mma_a_direct(16, 64), elem_bytes=2, machine=ampere()) == 0
