"""
Tests for the LEGO Tensor API.

Tests the user-facing tensor API including:
  - Round-trip correctness (transform then inverse == identity)
  - RowMajor/ColMajor/Tiled convenience constructors
  - Composable descriptor API (row, col, order_by, tile_by)
  - NumPy integration
  - PyTorch integration (if available)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.backend.compiler import (
    LayoutCompiler, RegPDesc, RowDesc, ColDesc, OrderByDesc,
    GroupByDesc, TileByDesc,
)
from lego.frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, Tiled,
    row, col, reg_p, order_by, tile_by, group_by,
)


class TestLayoutCompiler:
    """Test the MLIR JIT-compiled layout compiler."""

    def test_row_major_identity(self):
        """Row-major transform should be identity (no data movement)."""
        shape = (4, 8)
        layout = RowMajor(shape)
        result = layout.create_tensor(np.float32)

        expected = np.arange(32, dtype=np.float32).reshape(shape)
        np.testing.assert_array_equal(result, expected)

    def test_col_major_round_trip(self):
        """Col-major round-trip should return original data."""
        layout = ColMajor((4, 8))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)

        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.testing.assert_array_equal(back, expected)

    def test_round_trip_regp(self):
        """RegP with reversed permutation: transform + inverse == identity."""
        shape = (4, 8)
        layout = LegoLayout(group_by(shape, order_by(reg_p(shape, [1, 0]))))

        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)

        expected = np.arange(32, dtype=np.float32).reshape(shape)
        np.testing.assert_array_almost_equal(back, expected)

    def test_3d_layout(self):
        """Test with a 3D shape."""
        shape = (2, 3, 4)
        layout = LegoLayout(group_by(shape, order_by(row(*shape))))

        result = layout.create_tensor(np.float32)
        expected = np.arange(24, dtype=np.float32).reshape(shape)
        np.testing.assert_array_equal(result, expected)


class TestGroupByTransform:
    """Test GroupBy.transform() and .inverse_transform() methods."""

    def test_groupby_transform(self):
        """GroupBy.transform works on NumPy arrays."""
        shape = (4, 8)
        layout = RowMajor(shape)
        result = layout.create_tensor(np.float32)
        assert result.shape == shape

    def test_groupby_inverse_transform(self):
        """GroupBy.inverse_transform reverses the transform."""
        shape = (4, 8)
        layout = RowMajor(shape)
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(32, dtype=np.float32).reshape(shape)
        np.testing.assert_array_equal(back, expected)


class TestConvenienceConstructors:
    """Test the convenience constructor functions."""

    def test_rowmajor_import(self):
        """RowMajor is importable from lego."""
        layout = RowMajor((4, 8))
        assert layout._shape == (4, 8)

    def test_colmajor_import(self):
        """ColMajor is importable from lego."""
        layout = ColMajor((4, 8))
        assert layout._shape == (4, 8)

    def test_tiled_import(self):
        """Tiled is importable from lego."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        assert layout._shape == (8, 8)

    def test_custom_import(self):
        """Custom wraps a descriptor layout."""
        from lego.frontends.python_mlir import Custom
        shape = (4, 8)
        desc = group_by(shape, order_by(row(*shape)))
        layout = Custom(desc, shape)
        assert layout._shape == shape

    def test_tiled_rank_mismatch(self):
        """Tiled should raise on rank mismatch."""
        with pytest.raises(ValueError, match="same rank"):
            Tiled((8, 8), tile_shape=(4,))


class TestLegoLayout:
    """Test LegoLayout wrapper class."""

    def test_lego_layout_type_check(self):
        """LegoLayout rejects non-layout inputs."""
        with pytest.raises((TypeError, AttributeError)):
            LegoLayout("not a layout", (4, 8))

    def test_lego_layout_transform_numpy(self):
        """LegoLayout.create_tensor works with NumPy."""
        layout = RowMajor((4, 8))
        result = layout.create_tensor(np.float32)
        assert result.shape == (4, 8)

    def test_lego_layout_round_trip_numpy(self):
        """LegoLayout round-trip: create_tensor + inverse_transform == identity."""
        layout = ColMajor((4, 8))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.testing.assert_array_almost_equal(back, expected)


class TestDescriptorAPI:
    """Test the composable descriptor API that mirrors MLIR ops."""

    def test_row_desc(self):
        """RowDesc creates correct dims."""
        r = RowDesc((4, 8))
        assert r.dims == (4, 8)

    def test_col_desc(self):
        """ColDesc creates correct dims."""
        c = ColDesc((4, 8))
        assert c.dims == (4, 8)

    def test_tile_by_desc_dims(self):
        """TileByDesc.dims returns flattened tile dims."""
        ob = OrderByDesc([RowDesc((8, 8))])
        tb = TileByDesc(ob, [(2, 2), (4, 4)])
        assert tb.dims == (2, 2, 4, 4)
        assert tb.d == 2
        assert tb.q == 2
        assert tb.tile_shape == [2, 2]

    def test_order_by_tile_by_chaining(self):
        """OrderByDesc.tile_by() returns a TileByDesc."""
        ob = order_by(col(4, 8))
        tb = ob.tile_by((4,), (8,))
        assert isinstance(tb, TileByDesc)
        assert tb.input is ob

    def test_functional_constructors(self):
        """Functional constructors create correct descriptor types."""
        r = row(4, 8)
        assert isinstance(r, RowDesc) and r.dims == (4, 8)

        c = col(4, 8)
        assert isinstance(c, ColDesc) and c.dims == (4, 8)

        ob = order_by(r)
        assert isinstance(ob, OrderByDesc) and len(ob.perms) == 1

        gb = group_by((4, 8), ob)
        assert isinstance(gb, GroupByDesc) and gb.dims == (4, 8)

    def test_rowmajor_uses_row_desc(self):
        """RowMajor uses RowDesc internally."""
        layout = RowMajor((4, 8))
        inner = layout._layout  # GroupByDesc
        assert isinstance(inner, GroupByDesc)
        inner_order = inner.objects[0]
        assert isinstance(inner_order, OrderByDesc)
        assert isinstance(inner_order.perms[0], RowDesc)

    def test_colmajor_uses_col_desc(self):
        """ColMajor uses ColDesc internally."""
        layout = ColMajor((4, 8))
        inner = layout._layout  # GroupByDesc
        assert isinstance(inner, GroupByDesc)
        inner_order = inner.objects[0]
        assert isinstance(inner_order, OrderByDesc)
        assert isinstance(inner_order.perms[0], ColDesc)

    def test_tiled_uses_tile_by_desc(self):
        """Tiled uses TileByDesc internally (no SymPy)."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        inner = layout._layout
        assert isinstance(inner, TileByDesc)
        assert inner.tile_groups == [(2, 2), (4, 4)]

    def test_row_major_round_trip_with_row_desc(self):
        """RowMajor with RowDesc compiles and round-trips correctly."""
        layout = RowMajor((4, 8))
        result = layout.create_tensor(np.float32)
        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.testing.assert_array_equal(result, expected)

    def test_col_major_round_trip_with_col_desc(self):
        """ColMajor with ColDesc compiles and round-trips correctly."""
        layout = ColMajor((4, 8))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.testing.assert_array_almost_equal(back, expected)

    def test_col_major_produces_col_order(self):
        """ColMajor create_tensor: result[i,j] = flat[col_major_index(i,j)]."""
        layout = ColMajor((4, 8))
        result = layout.create_tensor(np.float32)

        # Col-major indexing: flat_idx = col * nrows + row
        # So result[i,j] = flat[j * 4 + i]
        expected = np.arange(32, dtype=np.float32).reshape((4, 8), order='F')
        np.testing.assert_array_equal(result, expected)
        assert result[0, 1] == 4  # j=1, i=0 → flat[1*4+0] = flat[4] = 4

    def test_tiled_round_trip(self):
        """Tiled with TileByDesc compiles and round-trips correctly."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        np.testing.assert_array_almost_equal(back, expected)

    def test_composable_api_col_tile(self):
        """order_by(col(...)).tile_by(...) composes correctly."""
        layout = LegoLayout(order_by(col(4, 8)).tile_by((2, 4), (2, 2)))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(32, dtype=np.float32).reshape(2, 4, 2, 2)
        np.testing.assert_array_almost_equal(back, expected)

    def test_mlir_text_contains_row_op(self):
        """RowMajor MLIR text contains lego.row."""
        layout = RowMajor((4, 8))
        mlir = layout.get_mlir()
        assert "lego.row" in mlir

    def test_mlir_text_contains_col_op(self):
        """ColMajor MLIR text contains lego.col."""
        layout = ColMajor((4, 8))
        mlir = layout.get_mlir()
        assert "lego.col" in mlir

    def test_mlir_text_contains_tile_by_op(self):
        """Tiled MLIR text contains lego.tile_by."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        mlir = layout.get_mlir()
        assert "lego.tile_by" in mlir


class TestPyTorchIntegration:
    """Test PyTorch integration (skipped if torch not available)."""

    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_torch_transform(self):
        """Basic PyTorch tensor create_tensor."""
        layout = RowMajor((4, 8))
        result = layout.create_tensor(np.float32)
        assert result.shape == (4, 8)

    def test_torch_round_trip(self):
        """PyTorch round-trip: create_tensor + inverse == identity."""
        layout = ColMajor((4, 8))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.testing.assert_array_almost_equal(back, expected)

    def test_torch_col_major(self):
        """Col-major create_tensor produces Fortran-order view."""
        layout = LegoLayout(order_by(col(4, 8)).tile_by((4, 8)))
        result = layout.create_tensor(np.float32)

        # layout = LegoLayout(order_by(row(2,3), col(4,5)).tile_by((8, 15)))
        # result = layout.create_tensor(np.float32)
        # print(result)


        # Col-major: result[i,j] = flat[j * nrows + i]
        expected = np.arange(32, dtype=np.float32).reshape((4, 8), order='F')
        np.testing.assert_array_equal(result, expected)
        assert result[0, 1] == 4

        # Round-trip
        back = layout.inverse_transform(result)
        expected_back = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.testing.assert_array_almost_equal(back, expected_back)

    def test_torch_tiled(self):
        """PyTorch tiled layout round-trip."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        result = layout.create_tensor(np.float32)
        back = layout.inverse_transform(result)
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)
        np.testing.assert_array_almost_equal(back, expected)
