"""Tests for exporting synthesized layouts to LEGO algebra expressions (SymPy level, no MLIR)."""
import numpy as np
import sympy as sp

from lego_synth.layouts import RowMajor, XorSwizzle, Rotation
from lego_synth.export import to_lego
from lego_synth.lego_adapter import load_core

core = load_core()


def _sample(rows, cols):
    rng = np.random.default_rng(0)
    i = rng.integers(0, rows, 40); j = rng.integers(0, cols, 40)
    return i, j


def test_exported_genp_forward_matches_the_concrete_layout():
    for L in (RowMajor(64, 80, pad=8), XorSwizzle(64, 80, shift=2, mask=1, unit=8), Rotation(64, 80, r=3, unit=8)):
        g = to_lego(L)
        assert isinstance(g, core.GenP)
        I, J = sp.symbols("I J", integer=True, nonnegative=True)
        expr = g.f_apply((I, J))
        for i, j in zip(*_sample(L.rows, L.cols)):
            got = int(expr.subs({I: int(i), J: int(j)}))
            assert got == int(L.flat(np.array([i]), np.array([j]))[0]), (L, i, j)


def test_exported_genp_inverse_round_trips():
    L = XorSwizzle(64, 80, shift=2, mask=1, unit=8)
    g = to_lego(L)
    I, J, F = sp.symbols("I J F", integer=True, nonnegative=True)
    fwd = g.f_apply((I, J)); inv = g.f_inv(F)
    for i, j in zip(*_sample(L.rows, L.cols)):
        f = int(fwd.subs({I: int(i), J: int(j)}))
        ii, jj = (int(e.subs({F: f})) for e in inv)
        assert (ii, jj) == (int(i), int(j))


def test_exported_tile_carries_the_padded_footprint():
    from lego_synth.export import to_lego_tile
    t = to_lego_tile(RowMajor(128, 32, pad=8))
    assert t.footprint == 128 * 40
