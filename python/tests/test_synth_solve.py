"""Tests for Z3-based hole filling: synthesize a layout that is conflict-free for a set of patterns."""
import time
import numpy as np

from lego_synth.machine import ampere
from lego_synth.patterns import row_vector, col_vector
from lego_synth.templates import XorTemplate, PaddedTemplate, RotationTemplate
from lego_synth.solve import solve_conflict_free
from lego_synth.eval_np import conflicts
from lego_synth.layouts import XorSwizzle, RowMajor, Rotation


def transpose_patterns(rows, cols):
    return [row_vector(rows, cols, 32), col_vector(rows, cols, 32)]


def _check(layout, pats, elem_bytes=4):
    for p in pats:
        assert conflicts(layout, p, elem_bytes, ampere()) == 0, p.name


def test_xor_template_finds_conflict_free_transpose_swizzle():
    pats = transpose_patterns(32, 32)
    t0 = time.time()
    sol = solve_conflict_free(XorTemplate(32, 32), pats, elem_bytes=4, machine=ampere())
    assert isinstance(sol, XorSwizzle)
    _check(sol, pats)
    assert time.time() - t0 < 60


def test_padded_template_finds_odd_padding_for_transpose():
    pats = transpose_patterns(32, 32)
    sol = solve_conflict_free(PaddedTemplate(32, 32, max_pad=4), pats, elem_bytes=4, machine=ampere())
    assert isinstance(sol, RowMajor)
    assert sol.pad % 2 == 1
    _check(sol, pats)


def test_rotation_template_finds_skewing_scheme_for_transpose():
    pats = transpose_patterns(32, 32)
    sol = solve_conflict_free(RotationTemplate(32, 32), pats, elem_bytes=4, machine=ampere())
    assert isinstance(sol, Rotation)
    _check(sol, pats)


def test_no_solution_reported_as_none():
    # An identity template (no holes) cannot fix the transpose load phase.
    pats = transpose_patterns(32, 32)
    sol = solve_conflict_free(PaddedTemplate(32, 32, max_pad=0), pats, elem_bytes=4, machine=ampere())
    assert sol is None


def test_xor_template_on_non_power_of_two_tile_keeps_rows_bijective():
    # 24x24 f32: an XOR swizzle must never send a column index outside [0, 24).
    pats = transpose_patterns(24, 24)
    sol = solve_conflict_free(XorTemplate(24, 24), pats, elem_bytes=4, machine=ampere())
    assert sol is not None
    i = np.repeat(np.arange(24), 24); j = np.tile(np.arange(24), 24)
    f = sol.flat(i, j)
    assert sorted(f.tolist()) == list(range(24 * 24))
    assert sol.footprint == 24 * 24
    _check(sol, pats)
