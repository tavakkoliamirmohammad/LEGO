"""Tests for evaluating real LEGO algebra layouts (SymPy) with the exhaustive evaluator."""
import numpy as np
import pytest
import sympy as sp

# The LEGO algebra evaluates layouts through an MLIR round trip, so these tests
# need the native extension; they are skipped until the build exists.
pytest.importorskip("lego.mlir.ir")

from lego_synth.lego_adapter import from_lego, load_core  # noqa: E402
from lego_synth.layouts import RowMajor  # noqa: E402

core = load_core()


def test_row_major_lego_layout_matches_concrete_row_major():
    i, j = sp.symbols("i j", integer=True, nonnegative=True)
    L = core.OrderBy(core.Row(32, 32)).TileBy((32, 32))
    concrete = from_lego(L, rows=32, cols=32)
    ii = np.repeat(np.arange(32), 32); jj = np.tile(np.arange(32), 32)
    assert np.array_equal(concrete.flat(ii, jj), RowMajor(32, 32).flat(ii, jj))


def test_tiled_lego_layout_is_a_permutation_of_the_index_space():
    L = core.OrderBy(core.Row(32, 32)).TileBy((4, 4), (8, 8))
    concrete = from_lego(L, rows=32, cols=32)
    ii = np.repeat(np.arange(32), 32); jj = np.tile(np.arange(32), 32)
    f = concrete.flat(ii, jj)
    assert sorted(f.tolist()) == list(range(32 * 32))
    assert concrete.footprint == 32 * 32
