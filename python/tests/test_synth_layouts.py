"""Tests for concrete layout families, access patterns, and whole-pattern conflicts."""
import numpy as np

from lego_synth.machine import ampere
from lego_synth.layouts import RowMajor, XorSwizzle, Rotation
from lego_synth.patterns import row_vector, col_vector
from lego_synth.eval_np import conflicts


def test_row_major_flat_index_with_padding():
    L = RowMajor(rows=4, cols=8, pad=1)
    assert L.flat(np.array([1]), np.array([2]))[0] == 1 * 9 + 2
    assert L.footprint == 4 * 9


def test_xor_swizzle_permutes_within_each_row():
    L = XorSwizzle(rows=32, cols=32, shift=0, mask=31)
    i = np.repeat(np.arange(32), 32)
    j = np.tile(np.arange(32), 32)
    f = L.flat(i, j)
    assert sorted(f.tolist()) == list(range(32 * 32))
    assert np.all(f // 32 == i)


def test_transpose_load_phase_on_row_major_f32_tile_costs_31_per_instruction():
    L = RowMajor(rows=32, cols=32)
    load = col_vector(rows=32, cols=32, warp=32)   # lane -> row, one column per instruction
    assert load.n_instr == 32
    assert conflicts(L, load, elem_bytes=4, machine=ampere()) == 32 * 31


def test_transpose_is_conflict_free_with_padding_or_xor_swizzle():
    store = row_vector(rows=32, cols=32, warp=32)  # lane -> col
    load = col_vector(rows=32, cols=32, warp=32)
    for L in (RowMajor(32, 32, pad=1), XorSwizzle(32, 32, shift=0, mask=31), Rotation(32, 32, r=1)):
        assert conflicts(L, store, 4, ampere()) == 0, type(L).__name__
        assert conflicts(L, load, 4, ampere()) == 0, type(L).__name__
