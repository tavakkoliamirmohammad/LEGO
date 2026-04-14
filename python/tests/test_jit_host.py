"""Self-contained JIT execution tests for the LEGO layout compiler.

Tests that layouts compile through lego-to-llvm and execute correctly on
the host CPU via MLIR ExecutionEngine.  No PyTorch dependency — only
lego.core, lego.backend.compiler, numpy, and sympy.

Covers:
  - Forward transform (apply): logical → physical
  - Inverse transform (apply_inverse): physical → logical
  - Round-trip: transform then inverse == identity
  - Permutation tables: bijectivity check (fwd[inv[i]] == i)
  - All layout primitives: Row, Col, RegP, GenP, OrderBy, GroupBy, TileBy
  - Symbolic dimensions
  - Edge cases: 1-element, 1D, high-dimensional
"""

import numpy as np
import pytest
import sympy as sp

from lego.core import OrderBy, Row, Col, RegP, GenP, GroupBy
from lego.backend.compiler import LayoutCompiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip(layout, shape, dtype="f32"):
    """Compile, apply forward, apply inverse, check identity."""
    compiler = LayoutCompiler(layout, shape, dtype)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    fwd = compiler.transform_numpy(data)
    back = compiler.inverse_transform_numpy(fwd)
    np.testing.assert_array_equal(back, data,
                                  err_msg=f"Round-trip failed for {layout}")


def _check_permutation(layout, shape):
    """Check that permutation tables are valid bijections."""
    compiler = LayoutCompiler(layout, shape, "i64")
    fwd, inv = compiler.get_permutation_table()
    n = np.prod(shape)
    assert len(fwd) == n
    assert len(inv) == n
    # fwd and inv are mutual inverses
    np.testing.assert_array_equal(inv[fwd], np.arange(n))
    np.testing.assert_array_equal(fwd[inv], np.arange(n))


# ===========================================================================
# Row / Col basics
# ===========================================================================

class TestRowCol:
    def test_row_identity(self):
        """Row-major should be identity permutation."""
        layout = OrderBy(Row(4, 8))
        compiler = LayoutCompiler(layout, (4, 8), "f32")
        data = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = compiler.transform_numpy(data)
        np.testing.assert_array_equal(result, data)

    def test_col_roundtrip_values(self):
        """Col-major forward + inverse should preserve all values."""
        layout = OrderBy(Col(3, 4))
        compiler = LayoutCompiler(layout, (3, 4), "f32")
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        fwd = compiler.transform_numpy(data)
        back = compiler.inverse_transform_numpy(fwd)
        np.testing.assert_array_equal(back, data)
        # Forward should differ from row-major (unless trivially 1D)
        assert not np.array_equal(fwd, data), "Col should permute data"

    def test_col_roundtrip(self):
        _roundtrip(OrderBy(Col(4, 8)), (4, 8))

    def test_row_3d(self):
        _roundtrip(OrderBy(Row(2, 3, 4)), (2, 3, 4))

    def test_col_3d(self):
        _roundtrip(OrderBy(Col(2, 3, 4)), (2, 3, 4))


# ===========================================================================
# RegP
# ===========================================================================

class TestRegP:
    def test_identity_perm(self):
        """RegP with identity permutation == Row."""
        layout = OrderBy(RegP((4, 8), [0, 1]))
        compiler = LayoutCompiler(layout, (4, 8), "f32")
        data = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = compiler.transform_numpy(data)
        np.testing.assert_array_equal(result, data)

    def test_reversed_perm(self):
        """RegP with reversed permutation == Col."""
        _roundtrip(OrderBy(RegP((3, 5), [1, 0])), (3, 5))

    def test_3d_perm(self):
        """RegP with 3D permutation."""
        _roundtrip(OrderBy(RegP((2, 3, 4), [2, 0, 1])), (2, 3, 4))

    def test_permutation_table(self):
        _check_permutation(OrderBy(RegP((4, 6), [1, 0])), (4, 6))


# ===========================================================================
# GenP
# ===========================================================================

class TestGenP:
    def test_genp_identity_2d(self):
        """GenP with row-major mapping should match Row."""
        def f_apply(idx):
            return idx[0] * 8 + idx[1]

        def f_inv(flat):
            return (sp.floor(flat / 8), sp.Mod(flat, 8))

        layout = OrderBy(GenP((4, 8), f_apply, f_inv))
        _roundtrip(layout, (4, 8))

    def test_genp_permutation(self):
        def f_apply(idx):
            return idx[0] * 5 + idx[1]

        def f_inv(flat):
            return (sp.floor(flat / 5), sp.Mod(flat, 5))

        layout = OrderBy(GenP((3, 5), f_apply, f_inv))
        _check_permutation(layout, (3, 5))


# ===========================================================================
# TileBy
# ===========================================================================

class TestTileBy:
    def test_1d_tile(self):
        """1D tiling: 16 elements tiled into 4 blocks of 4."""
        layout = OrderBy(Row(16)).TileBy([4], [4])
        _roundtrip(layout, (16,))

    def test_2d_tile(self):
        """2D tiling: 8x8 tiled into 2x2 blocks of 4x4."""
        layout = OrderBy(Row(8, 8)).TileBy([2, 2], [4, 4])
        _roundtrip(layout, (8, 8))

    def test_2d_tile_permutation(self):
        layout = OrderBy(Row(8, 8)).TileBy([2, 2], [4, 4])
        _check_permutation(layout, (8, 8))

    def test_matmul_tile(self):
        """Tiled layout matching the matmul pattern from puzzle_16."""
        M, N, K, T = 32, 32, 32, 16
        for dims in [(M, K), (K, N), (M, N)]:
            layout = OrderBy(Row(*dims)).TileBy(
                [dims[0] // T, dims[1] // T], [T, T])
            _roundtrip(layout, dims)
            _check_permutation(layout, dims)

    def test_2d_tile_larger(self):
        """2D tiling on a larger matrix: 16x16 with 4x4 tiles."""
        layout = OrderBy(Row(16, 16)).TileBy([4, 4], [4, 4])
        _roundtrip(layout, (16, 16))

    def test_non_power_of_two(self):
        """TileBy with non-power-of-two sizes."""
        layout = OrderBy(Row(12, 15)).TileBy([4, 5], [3, 3])
        _roundtrip(layout, (12, 15))


# ===========================================================================
# OrderBy composition
# ===========================================================================

class TestOrderByComposition:
    def test_two_layouts(self):
        """OrderBy with two sub-layouts."""
        layout = OrderBy(Row(4), Row(8))
        _roundtrip(layout, (4, 8))

    def test_mixed_row_col(self):
        """OrderBy mixing Row and Col sub-layouts."""
        layout = OrderBy(Row(4), Col(2, 3))
        _roundtrip(layout, (4, 2, 3))


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_element(self):
        """1-element layout."""
        layout = OrderBy(Row(1))
        _roundtrip(layout, (1,))

    def test_1d(self):
        """Simple 1D layout."""
        layout = OrderBy(Row(32))
        _roundtrip(layout, (32,))

    def test_large_2d(self):
        """Larger 2D layout."""
        layout = OrderBy(Row(64, 64))
        _roundtrip(layout, (64, 64))

    def test_4d(self):
        """4D layout."""
        layout = OrderBy(Row(2, 3, 4, 5))
        _roundtrip(layout, (2, 3, 4, 5))


# ===========================================================================
# Symbolic dimensions
# ===========================================================================

class TestSymbolic:
    def test_symbolic_row(self):
        """Layout with symbolic dimensions resolved at compile time."""
        R, C = sp.symbols('R C', positive=True, integer=True)
        layout = OrderBy(Row(R, C))
        compiler = LayoutCompiler(layout, (4, 8), "f32",
                                  dim_bindings={R: 4, C: 8})
        data = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = compiler.transform_numpy(data)
        np.testing.assert_array_equal(result, data)

    def test_symbolic_roundtrip(self):
        R, C = sp.symbols('R C', positive=True, integer=True)
        layout = OrderBy(Row(R, C))
        compiler = LayoutCompiler(layout, (3, 5), "f32",
                                  dim_bindings={R: 3, C: 5})
        data = np.arange(15, dtype=np.float32).reshape(3, 5)
        fwd = compiler.transform_numpy(data)
        back = compiler.inverse_transform_numpy(fwd)
        np.testing.assert_array_equal(back, data)


# ===========================================================================
# Data types
# ===========================================================================

class TestDtypes:
    def test_f32(self):
        _roundtrip(OrderBy(Row(4, 8)), (4, 8), dtype="f32")

    def test_f64(self):
        layout = OrderBy(Col(3, 5))
        compiler = LayoutCompiler(layout, (3, 5), "f64")
        data = np.arange(15, dtype=np.float64).reshape(3, 5)
        fwd = compiler.transform_numpy(data)
        back = compiler.inverse_transform_numpy(fwd)
        np.testing.assert_array_equal(back, data)

    def test_i64(self):
        layout = OrderBy(Row(4, 4))
        compiler = LayoutCompiler(layout, (4, 4), "i64")
        data = np.arange(16, dtype=np.int64).reshape(4, 4)
        fwd = compiler.transform_numpy(data)
        back = compiler.inverse_transform_numpy(fwd)
        np.testing.assert_array_equal(back, data)
