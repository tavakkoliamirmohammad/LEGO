"""Tests for tiered search: f2 (Triton-like), affine_xor (CuTe/Hexcute-like), full (LEGO)."""
from lego_synth.machine import ampere
from lego_synth.patterns import row_vector, col_vector
from lego_synth.search import Problem, synthesize
from lego_synth.eval_np import conflicts
from lego_synth.layouts import XorSwizzle, RowMajor


def transpose_problem(rows, cols, elem_bytes=4):
    return Problem(rows=rows, cols=cols, elem_bytes=elem_bytes,
                   patterns=[row_vector(rows, cols, 32), col_vector(rows, cols, 32)],
                   machine=ampere())


def _conflict_free(sol, prob):
    return all(conflicts(sol.layout, p, prob.elem_bytes, prob.machine) == 0 for p in prob.patterns)


def test_f2_tier_solves_power_of_two_transpose_with_xor_only():
    prob = transpose_problem(32, 32)
    sol = synthesize(prob, tier="f2")
    assert isinstance(sol.layout, XorSwizzle)
    assert sol.footprint == 32 * 32
    assert _conflict_free(sol, prob)


def test_f2_tier_must_pad_non_power_of_two_tile_to_32_columns():
    prob = transpose_problem(24, 24)
    sol = synthesize(prob, tier="f2")
    assert sol.footprint == 24 * 32
    assert _conflict_free(sol, prob)


def test_affine_xor_tier_uses_padding_on_non_power_of_two_tile():
    prob = transpose_problem(24, 24)
    sol = synthesize(prob, tier="affine_xor")
    assert isinstance(sol.layout, RowMajor)
    assert sol.footprint == 24 * 25
    assert _conflict_free(sol, prob)


def test_full_tier_keeps_exact_footprint_on_non_power_of_two_tile():
    prob = transpose_problem(24, 24)
    sol = synthesize(prob, tier="full")
    assert sol.footprint == 24 * 24
    assert _conflict_free(sol, prob)


def test_search_returns_minimum_footprint_solution():
    prob = transpose_problem(32, 32)
    sol = synthesize(prob, tier="full")
    assert sol.footprint == 32 * 32   # xor or rotation beat any padding
