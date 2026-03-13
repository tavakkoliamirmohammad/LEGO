"""
Tests for the LEGO Tensor API.

Tests the user-facing tensor API including:
  - Round-trip correctness (transform then inverse == identity)
  - RowMajor/ColMajor/Tiled convenience constructors
  - NumPy integration
  - PyTorch integration (if available)
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.lego import Row, Col, OrderBy, GroupBy, RegP, get_sigma_perm
from lego.compiler import LayoutCompiler


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
        from lego.tensor_api import RowMajor
        layout = RowMajor((4, 8))
        assert layout._shape == (4, 8)

    def test_colmajor_import(self):
        """ColMajor is importable from lego."""
        from lego.tensor_api import ColMajor
        layout = ColMajor((4, 8))
        assert layout._shape == (4, 8)

    def test_tiled_import(self):
        """Tiled is importable from lego."""
        from lego.tensor_api import Tiled
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
        from lego.tensor_api import Tiled
        with pytest.raises(ValueError, match="same rank"):
            Tiled((8, 8), tile_shape=(4,))


class TestLegoLayout:
    """Test LegoLayout wrapper class."""

    def test_lego_layout_type_check(self):
        """LegoLayout rejects non-GroupBy inputs."""
        from lego.tensor_api import LegoLayout
        with pytest.raises(TypeError, match="GroupBy"):
            LegoLayout(Row(4, 8), (4, 8))

    def test_lego_layout_transform_numpy(self):
        """LegoLayout.transform works with NumPy."""
        from lego.tensor_api import RowMajor
        layout = RowMajor((4, 8))
        arr = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = layout.transform(arr)
        assert result.shape == (4, 8)

    def test_lego_layout_round_trip_numpy(self):
        """LegoLayout round-trip: transform + inverse_transform == identity."""
        from lego.tensor_api import RowMajor
        layout = RowMajor((4, 8))
        arr = np.random.randn(4, 8).astype(np.float32)
        result = layout.transform(arr)
        back = layout.inverse_transform(result)
        np.testing.assert_array_almost_equal(arr, back)


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
        from lego.tensor_api import RowMajor

        layout = RowMajor((4, 8))
        x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        result = layout.transform(x)
        assert result.shape == (4, 8)

    def test_torch_round_trip(self):
        """PyTorch round-trip: transform + inverse == identity."""
        import torch
        from lego.tensor_api import RowMajor

        layout = RowMajor((4, 8))
        x = torch.randn(4, 8)
        result = layout.transform(x)
        back = layout.inverse_transform(result)
        assert torch.allclose(x, back, atol=1e-6)
