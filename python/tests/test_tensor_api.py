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

from lego.lego import Row, Col, OrderBy, GroupBy, RegP, get_sigma_perm
from lego.compiler import (
    LayoutCompiler, RegPDesc, RowDesc, ColDesc, OrderByDesc,
    GroupByDesc, TileByDesc,
)
from lego.tensor_api import (
    LegoLayout, RowMajor, ColMajor, Tiled,
    row, col, order_by, tile_by, group_by,
)


class TestLayoutCompiler:
    """Test the MLIR JIT-compiled layout compiler."""

    def test_row_major_identity(self):
        """Row-major transform should be identity (no data movement)."""
        shape = (4, 8)
        L = OrderBy(Row(*shape)).GroupBy([shape])
        compiler = LayoutCompiler(L, shape)

        arr = np.arange(32, dtype=np.float32).reshape(shape)
        result = compiler.transform_numpy(arr)

        np.testing.assert_array_equal(arr, result)

    def test_col_major_round_trip(self):
        """Col-major round-trip should return original data."""
        shape = (4, 8)
        L = OrderBy(Col(*shape)).GroupBy([shape])
        compiler = LayoutCompiler(L, shape)

        arr = np.arange(32, dtype=np.float32).reshape(shape)
        result = compiler.transform_numpy(arr)

        back = compiler.inverse_transform_numpy(result)
        np.testing.assert_array_equal(arr, back)

    def test_round_trip_regp(self):
        """RegP with reversed permutation: transform + inverse == identity."""
        shape = (4, 8)
        perm = [1, 0]
        L = OrderBy(RegP(shape, perm)).GroupBy([shape])
        compiler = LayoutCompiler(L, shape)

        arr = np.random.randn(*shape).astype(np.float32)
        transformed = compiler.transform_numpy(arr)
        back = compiler.inverse_transform_numpy(transformed)
        np.testing.assert_array_almost_equal(arr, back)

    def test_3d_layout(self):
        """Test with a 3D shape."""
        shape = (2, 3, 4)
        L = OrderBy(Row(*shape)).GroupBy([shape])
        compiler = LayoutCompiler(L, shape)

        arr = np.arange(24, dtype=np.float32).reshape(shape)
        result = compiler.transform_numpy(arr)
        np.testing.assert_array_equal(arr, result)


class TestGroupByTransform:
    """Test GroupBy.transform() and .inverse_transform() methods."""

    def test_groupby_transform(self):
        """GroupBy.transform works on NumPy arrays."""
        shape = (4, 8)
        L = OrderBy(Row(*shape)).GroupBy([shape])
        arr = np.arange(32, dtype=np.float32).reshape(shape)
        result = L.transform(arr)
        assert result.shape == arr.shape

    def test_groupby_inverse_transform(self):
        """GroupBy.inverse_transform reverses the transform."""
        shape = (4, 8)
        L = OrderBy(Row(*shape)).GroupBy([shape])
        arr = np.arange(32, dtype=np.float32).reshape(shape)
        result = L.transform(arr)
        back = L.inverse_transform(result)
        np.testing.assert_array_equal(arr, back)


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
        """Custom wraps a GroupBy object."""
        from lego.tensor_api import Custom
        shape = (4, 8)
        L = OrderBy(Row(*shape)).GroupBy([shape])
        layout = Custom(L, shape)
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
        """LegoLayout.transform works with NumPy."""
        layout = RowMajor((4, 8))
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = layout.transform(arr)
        assert result.shape == (4, 8)

    def test_lego_layout_round_trip_numpy(self):
        """LegoLayout round-trip: transform + inverse_transform == identity."""
        layout = RowMajor((4, 8))
        arr = np.random.randn(4, 8).astype(np.float32)
        result = layout.transform(arr)
        back = layout.inverse_transform(result)
        np.testing.assert_array_almost_equal(arr, back)


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
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = layout.transform(arr)
        np.testing.assert_array_equal(arr, result)

    def test_col_major_round_trip_with_col_desc(self):
        """ColMajor with ColDesc compiles and round-trips correctly."""
        layout = ColMajor((4, 8))
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = layout.transform(arr)
        back = layout.inverse_transform(result)
        np.testing.assert_array_almost_equal(arr, back)

    def test_col_major_produces_col_order(self):
        """ColMajor actually reorders data to column-major."""
        shape = (4, 8)
        layout = ColMajor(shape)
        arr = np.arange(32, dtype=np.float32).reshape(shape)
        result = layout.transform(arr)

        # Col-major flat order groups by column:
        # col 0: [0, 8, 16, 24], col 1: [1, 9, 17, 25], ...
        expected_flat = np.array([
            arr[r, c]
            for c in range(8)
            for r in range(4)
        ], dtype=np.float32)
        np.testing.assert_array_equal(result.ravel(), expected_flat)

    def test_tiled_round_trip(self):
        """Tiled with TileByDesc compiles and round-trips correctly."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        result = layout.transform(arr)
        back = layout.inverse_transform(result)
        np.testing.assert_array_almost_equal(arr, back)

    def test_composable_api_col_tile(self):
        """order_by(col(...)).tile_by(...) composes correctly."""
        layout = LegoLayout(
            order_by(col(4, 8)).tile_by((4,), (8,)),
            (4, 8),
        )
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = layout.transform(arr)
        back = layout.inverse_transform(result)
        np.testing.assert_array_almost_equal(arr, back)

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
        """Basic PyTorch tensor transform."""
        import torch

        layout = RowMajor((4, 8))
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        result = layout.transform(x)
        assert result.shape == (4, 8)

    def test_torch_round_trip(self):
        """PyTorch round-trip: transform + inverse == identity."""
        import torch

        layout = RowMajor((4, 8))
        x = torch.randn(4, 8)
        result = layout.transform(x)
        back = layout.inverse_transform(result)
        assert torch.allclose(x, back, atol=1e-6)

    def test_torch_col_major(self):
        """Transform a 4x8 PyTorch tensor from row-major to col-major."""
        import torch

        layout = LegoLayout(
            order_by(col(4, 8)).tile_by((4,8)),
            (4, 8),
        )
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        print("x")
        print(x)
        result = layout.transform(x)
        print("result")
        print(result)

        # Col-major reorders so that columns are contiguous in memory.
        # For a 4x8 matrix, col-major flat order groups by column:
        #   col 0: [0, 8, 16, 24], col 1: [1, 9, 17, 25], ...
        expected_flat = torch.tensor([
            x[r, c].item()
            for c in range(8)
            for r in range(4)
        ], dtype=torch.float32)
        assert torch.equal(result.reshape(-1), expected_flat), (
            f"Expected col-major flat: {expected_flat.tolist()}\n"
            f"Got: {result.reshape(-1).tolist()}"
        )

        # Round-trip must recover the original
        back = layout.inverse_transform(result)
        assert torch.allclose(x, back, atol=1e-6)

    def test_torch_tiled(self):
        """PyTorch tiled layout round-trip."""
        import torch

        layout = Tiled((8, 8), tile_shape=(4, 4))
        x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        result = layout.transform(x)
        back = layout.inverse_transform(result)
        assert torch.allclose(x, back, atol=1e-6)
