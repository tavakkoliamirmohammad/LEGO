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


def test_mma_m16n8k16_a_fragment_maps_lane_and_register_to_the_documented_coordinates():
    # PTX ISA: groupID = lane >> 2, tid = lane & 3; a0,a1: (groupID, tid*2 + {0,1});
    # a2,a3: (groupID+8, ...); a4,a5: (groupID, tid*2 + 8 + {0,1}); a6,a7: (groupID+8, ... + 8)
    p = mma_m16n8k16_a()
    assert p.n_instr == 8              # one "instruction" per register (element) index
    i, j = p.coords(0)
    assert i[5] == 5 >> 2 and j[5] == (5 & 3) * 2
    i, j = p.coords(3)
    assert i[5] == (5 >> 2) + 8 and j[5] == (5 & 3) * 2 + 1
    i, j = p.coords(6)
    assert i[5] == (5 >> 2) + 8 and j[5] == (5 & 3) * 2 + 8
