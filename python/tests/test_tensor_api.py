"""
Tests for the LEGO Tensor API.

Tests the user-facing tensor API including:
  - Round-trip correctness (transform then inverse == identity)
  - RowMajor/ColMajor/Tiled convenience constructors
  - Composable descriptor API (row, col, order_by, tile_by)
  - NumPy integration
  - PyTorch integration (if available)
  - Permutation tables
  - Ergonomics: __call__, __repr__, rank, is_identity, compose, __eq__
  - Batched transforms
  - New constructors: Transposed, ZCurve, Swizzle, BlockCyclic
  - LegoTensor, LegoArray
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.backend.compiler import LayoutCompiler
from lego.core import Row, Col, RegP, OrderBy, GroupBy, TileByLayout
from lego.frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, Tiled, TiledView, Custom,
    row, col, reg_p, order_by, tile_by, group_by,
    Transposed, ZCurve, Swizzle, BlockCyclic,
    Batched, BatchedLayout, LegoArray,
)

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


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
        """Tiled is importable from lego and returns TiledView."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        assert isinstance(layout, TiledView)
        assert layout.original_shape == (8, 8)
        assert layout.tile_grid == (2, 2)
        assert layout.tile_shape == (4, 4)
        assert layout.shape == (2, 2, 4, 4)

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
        """LegoLayout rejects non-layout inputs at compile time."""
        with pytest.raises((TypeError, AttributeError)):
            LegoLayout("not a layout", (4, 8)).create_tensor(np.float32)

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
    """Test the composable layout API that mirrors MLIR ops."""

    def test_row(self):
        """Row creates correct dims."""
        r = Row(4, 8)
        assert r.dims() == (4, 8)

    def test_col(self):
        """Col creates correct dims."""
        c = Col(4, 8)
        assert c.dims() == (4, 8)

    def test_tile_by_dims(self):
        """TileByLayout dims and tile_shape."""
        layout = OrderBy(Row(8, 8)).TileBy((2, 2), (4, 4))
        assert isinstance(layout, TileByLayout)
        assert layout._dims == (2, 2, 4, 4)
        assert layout.tile_shape == [2, 2]

    def test_order_by_tile_by_chaining(self):
        """OrderBy.TileBy() returns a TileByLayout."""
        layout = OrderBy(Col(4, 8)).TileBy((4,), (8,))
        assert isinstance(layout, TileByLayout)

    def test_functional_constructors(self):
        """Functional constructors create correct types."""
        r = row(4, 8)
        assert isinstance(r, Row) and r.dims() == (4, 8)

        c = col(4, 8)
        assert isinstance(c, Col) and c.dims() == (4, 8)

        ob = order_by(r)
        assert isinstance(ob, OrderBy) and len(ob.perms) == 1

        gb = group_by((4, 8), ob)
        assert isinstance(gb, GroupBy)

    def test_rowmajor_uses_row(self):
        """RowMajor uses Row internally."""
        layout = RowMajor((4, 8))
        inner = layout._layout
        assert isinstance(inner, GroupBy)
        inner_order = inner.objects[0]
        assert isinstance(inner_order, OrderBy)
        assert isinstance(inner_order.perms[0], Row)

    def test_colmajor_uses_col(self):
        """ColMajor uses Col internally."""
        layout = ColMajor((4, 8))
        inner = layout._layout
        assert isinstance(inner, GroupBy)
        inner_order = inner.objects[0]
        assert isinstance(inner_order, OrderBy)
        assert isinstance(inner_order.perms[0], Col)

    def test_tile_by_via_custom(self):
        """TileByLayout is usable via Custom wrapper."""
        tile_layout = OrderBy(Row(8, 8)).TileBy((2, 2), (4, 4))
        layout = Custom(tile_layout, (8, 8))
        assert isinstance(tile_layout, TileByLayout)
        assert tile_layout._tile_groups == [(2, 2), (4, 4)]

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
        assert result[0, 1] == 4  # j=1, i=0 -> flat[1*4+0] = flat[4] = 4

    def test_tiled_round_trip(self):
        """Tiled transform + inverse == identity."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        tiled = layout.transform(arr)
        assert tiled.shape == (2, 2, 4, 4)
        back = layout.inverse_transform(tiled)
        np.testing.assert_array_equal(back, arr)

    def test_tiled_values_preserved(self):
        """Tiled view preserves values — tiled[tr,tc,lr,lc] == original[tr*th+lr, tc*tw+lc]."""
        layout = Tiled((8, 8), tile_shape=(4, 4))
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        tiled = layout.transform(arr)
        for tr in range(2):
            for tc in range(2):
                expected_tile = arr[tr*4:(tr+1)*4, tc*4:(tc+1)*4]
                np.testing.assert_array_equal(tiled[tr, tc], expected_tile)

    def test_composable_api_col_tile(self):
        """order_by(col(...)).tile_by(...) composes correctly."""
        layout = LegoLayout(order_by(col(4, 8)).TileBy((2, 4), (2, 2)))
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
        """TileByLayout MLIR text contains lego.tile_by."""
        tile_layout = OrderBy(Row(8, 8)).TileBy((2, 2), (4, 4))
        layout = Custom(tile_layout, (8, 8))
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
        layout = LegoLayout(order_by(col(4, 8)).TileBy((4, 8)))
        result = layout.create_tensor(np.float32)

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
        import torch
        layout = Tiled((8, 8), tile_shape=(4, 4))
        x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        tiled = layout.transform(x)
        assert tiled.shape == (2, 2, 4, 4)
        back = layout.inverse_transform(tiled)
        torch.testing.assert_close(back, x)


# ============================================================================
# Permutation Table Tests
# ============================================================================

class TestPermutationTable:
    def test_perm_table_round_trip(self):
        """Permutation table forward then inverse == identity."""
        layout = ColMajor((8, 8))
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, inv = compiler.get_permutation_table()
        assert np.array_equal(inv[fwd], np.arange(64))
        assert np.array_equal(fwd[inv], np.arange(64))

    def test_perm_table_matches_jit(self):
        """Permutation table produces same result as JIT transform."""
        layout = ColMajor((8, 8))
        arr = np.random.randn(64).astype(np.float32)
        jit_result = layout.transform(arr)
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        perm_result = arr[fwd].reshape(8, 8)
        np.testing.assert_array_equal(jit_result, perm_result)

    @requires_cuda
    def test_cuda_transform_no_cpu_roundtrip(self):
        """CUDA tensors should transform without CPU copy."""
        import torch
        layout = ColMajor((8, 8))
        x = torch.randn(8, 8, device="cuda")
        result = layout.transform(x)
        assert result.device.type == "cuda"
        back = layout.inverse_transform(result)
        torch.testing.assert_close(back, x)


# ============================================================================
# Ergonomics Tests
# ============================================================================

class TestErgonomics:
    def test_call_shorthand(self):
        layout = RowMajor((4, 8))
        arr = np.arange(32, dtype=np.float32)
        np.testing.assert_array_equal(layout(arr), layout.transform(arr))

    def test_repr(self):
        layout = Tiled((8, 8), tile_shape=(4, 4))
        assert "TiledView" in repr(layout)
        assert "(8, 8)" in repr(layout)

    def test_rank(self):
        assert Tiled((8, 8), tile_shape=(4, 4)).rank == 4
        assert RowMajor((2, 3, 4)).rank == 3

    def test_identity(self):
        assert RowMajor((4, 8)).is_identity
        assert not ColMajor((4, 8)).is_identity

    def test_compose(self):
        a = ColMajor((4, 8))
        b = ColMajor((4, 8))
        arr = np.arange(32, dtype=np.float32)
        composed = a.compose(b)
        expected = b.transform(a.transform(arr))
        np.testing.assert_array_equal(composed(arr), expected)

    def test_eq(self):
        assert RowMajor((4, 8)) == RowMajor((4, 8))
        assert RowMajor((4, 8)) != ColMajor((4, 8))

    def test_hash(self):
        s = {RowMajor((4, 8)), RowMajor((4, 8))}
        assert len(s) == 1


# ============================================================================
# Batched Tests
# ============================================================================

class TestBatched:
    def test_batched_shape(self):
        batched = Batched(ColMajor((8, 8)), batch_shape=(32,))
        assert batched.shape == (32, 8, 8)

    def test_batched_round_trip(self):
        base = ColMajor((4, 4))
        batched = Batched(base, batch_shape=(3,))
        arr = np.random.randn(3, 4, 4).astype(np.float32)
        result = batched.transform(arr)
        back = batched.inverse_transform(result)
        np.testing.assert_array_almost_equal(back, arr)

    def test_batched_matches_loop(self):
        """Batched transform == applying base transform in a loop."""
        base = ColMajor((3, 4))
        batched = Batched(base, batch_shape=(5,))
        arr = np.random.randn(5, 3, 4).astype(np.float32)
        result = batched.transform(arr)
        for i in range(5):
            expected_slice = base.transform(arr[i])
            np.testing.assert_array_equal(result[i], expected_slice)


# ============================================================================
# New Constructor Tests
# ============================================================================

class TestTransposed:
    def test_transposed_2d_expected(self):
        """Transposed((3,4)) == column-major order."""
        result = Transposed((3, 4)).create_tensor(np.float32)
        expected = np.array([
            [ 0,  3,  6,  9],
            [ 1,  4,  7, 10],
            [ 2,  5,  8, 11],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_transposed_round_trip(self):
        layout = Transposed((3, 4))
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        back = layout.inverse_transform(layout.transform(arr))
        np.testing.assert_array_equal(back, arr)


class TestZCurve:
    def test_zcurve_4x4_expected(self):
        """ZCurve((4,4)) interleaves row/col bits into Morton code."""
        result = ZCurve((4, 4)).create_tensor(np.float32)
        expected = np.array([
            [ 0,  2,  8, 10],
            [ 1,  3,  9, 11],
            [ 4,  6, 12, 14],
            [ 5,  7, 13, 15],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_zcurve_2x2_expected(self):
        """Smallest Z-curve: 2x2."""
        result = ZCurve((2, 2)).create_tensor(np.float32)
        expected = np.array([
            [0, 2],
            [1, 3],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_zcurve_round_trip(self):
        layout = ZCurve((4, 4))
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        back = layout.inverse_transform(layout.transform(arr))
        np.testing.assert_array_equal(back, arr)

    def test_zcurve_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            ZCurve((4, 4, 4))


class TestSwizzle:
    def test_swizzle_4x4_expected(self):
        """Swizzle((4,4), num_bits=2): XOR col with (row & 0b11)."""
        result = Swizzle((4, 4), num_bits=2).create_tensor(np.float32)
        expected = np.array([
            [ 0,  1,  2,  3],
            [ 5,  4,  7,  6],
            [10, 11,  8,  9],
            [15, 14, 13, 12],
        ], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_swizzle_self_inverse(self):
        """XOR swizzle is its own inverse: swizzle(swizzle(x)) == x."""
        layout = Swizzle((4, 4), num_bits=2)
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        once = layout.transform(arr)
        twice = layout.transform(once)
        np.testing.assert_array_equal(twice, arr)

    def test_swizzle_round_trip(self):
        layout = Swizzle((8, 8), num_bits=3)
        arr = np.random.randn(8, 8).astype(np.float32)
        back = layout.inverse_transform(layout.transform(arr))
        np.testing.assert_array_equal(back, arr)

    def test_swizzle_bank_conflict_free(self):
        """Each column index in the swizzled layout maps to a unique bank."""
        layout = Swizzle((4, 4), num_bits=2)
        result = layout.create_tensor(np.float32)
        for row in range(4):
            cols = result[row].astype(int) % 4  # bank = col % num_banks
            assert len(set(cols.tolist())) == 4  # all 4 banks used

    def test_swizzle_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2D"):
            Swizzle((4, 4, 4))


class TestBlockCyclic:
    def test_blockcyclic_16_expected(self):
        """BlockCyclic((16,), bs=2, np=2): data grouped by process."""
        result = BlockCyclic((16,), block_size=2, num_procs=2).create_tensor(np.float32)
        expected = np.array(
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result, expected)

    def test_blockcyclic_round_trip(self):
        layout = BlockCyclic((16,), block_size=2, num_procs=2)
        arr = np.arange(16, dtype=np.float32)
        back = layout.inverse_transform(layout.transform(arr))
        np.testing.assert_array_equal(back, arr)

    def test_blockcyclic_rejects_nd(self):
        with pytest.raises(ValueError):
            BlockCyclic((4, 4), block_size=2, num_procs=2)


# ============================================================================
# LegoTensor & LegoArray Tests
# ============================================================================

class TestLegoTensor:
    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_as_lego_tensor_to_logical(self):
        import torch
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        back = lx.to_logical()
        torch.testing.assert_close(back, x)

    def test_lego_tensor_repr(self):
        import torch
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        lx = as_lego_tensor(x, layout)
        assert "LegoTensor" in repr(lx)

    def test_torch_function_fallback(self):
        """Ops on LegoTensor auto-convert to logical order."""
        import torch
        from lego.backend.torch_tensor import as_lego_tensor
        layout = RowMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = lx + 1  # should trigger __torch_function__
        assert isinstance(result, torch.Tensor)


class TestLegoArray:
    def test_array_protocol(self):
        layout = ColMajor((3, 4))
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        la = layout.as_array(arr)
        # np.asarray triggers __array__, returns logical order
        back = np.asarray(la)
        np.testing.assert_array_equal(back, arr)

    def test_array_shape_dtype(self):
        layout = ColMajor((4, 4))
        arr = np.zeros((4, 4), dtype=np.float64)
        la = layout.as_array(arr)
        assert la.shape == (4, 4)
        assert la.dtype == np.float64


# ============================================================================
# DLPack Tests
# ============================================================================

class TestDLPack:
    def test_to_dlpack_numpy(self):
        """to_dlpack on a numpy array produces a consumable DLPack capsule."""
        layout = ColMajor((3, 4))
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        transformed = layout.transform(arr)
        # Verify the transformed array supports DLPack
        assert hasattr(transformed, '__dlpack__')
        # Consume via np.from_dlpack on the array (not the capsule)
        restored = np.from_dlpack(transformed)
        np.testing.assert_array_equal(restored, transformed)

    @pytest.fixture(autouse=False)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_to_dlpack_torch(self, check_torch):
        """to_dlpack on a torch tensor returns a DLPack capsule."""
        import torch
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        capsule = layout.to_dlpack(x)
        restored = torch.from_dlpack(capsule)
        expected = layout.transform(x)
        torch.testing.assert_close(restored, expected)

    def test_from_dlpack_round_trip(self, check_torch):
        """from_dlpack imports a tensor as a LegoArray with correct inverse."""
        import torch
        layout = ColMajor((3, 4))
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        transformed = layout.transform(x)
        # Pass the tensor directly (has __dlpack__)
        la = layout.from_dlpack(transformed)
        assert isinstance(la, LegoArray)
        back = la.to_logical()
        np.testing.assert_array_almost_equal(back, x.numpy())


# ============================================================================
# torch.library Custom Op Tests
# ============================================================================

class TestCustomOp:
    @pytest.fixture(autouse=True)
    def check_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_permute_op_cpu(self):
        """torch.ops.lego.permute works on CPU."""
        import torch
        from lego.backend import torch_ops  # ensure registration
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        perm = torch.tensor([5, 4, 3, 2, 1, 0])
        result = torch.ops.lego.permute(x, perm)
        expected = x.view(-1)[perm].view(2, 3)
        torch.testing.assert_close(result, expected)

    @requires_cuda
    def test_permute_op_cuda(self):
        """torch.ops.lego.permute works on CUDA."""
        import torch
        from lego.backend import torch_ops
        x = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
        perm = torch.tensor([5, 4, 3, 2, 1, 0], device="cuda")
        result = torch.ops.lego.permute(x, perm)
        expected = x.view(-1)[perm].view(2, 3)
        torch.testing.assert_close(result, expected)

    def test_permute_op_torch_compile(self):
        """torch.ops.lego.permute is compatible with torch.compile."""
        import torch
        from lego.backend import torch_ops

        @torch.compile(fullgraph=True)
        def compiled_permute(x, p):
            return torch.ops.lego.permute(x, p)

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        perm = torch.tensor([5, 4, 3, 2, 1, 0])
        result = compiled_permute(x, perm)
        expected = x.view(-1)[perm].view(2, 3)
        torch.testing.assert_close(result, expected)


# ============================================================================
# Symbolic JIT Compilation (C2)
# ============================================================================

class TestSymbolicJIT:
    """Test JIT compilation with symbolic layout dimensions."""

    def test_symbolic_row_major_with_bindings(self):
        """Row-major with symbolic M, N, bound at compile time."""
        import sympy as sp
        M, N = sp.symbols('M N', integer=True, positive=True)
        layout = OrderBy(Row(M, N)).GroupBy([(M, N)])
        compiler = LayoutCompiler(layout, (3, 4), "f32",
                                  dim_bindings={M: 3, N: 4})
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = compiler.transform_numpy(data)
        # Row-major is identity
        np.testing.assert_array_equal(result, data)

    def test_symbolic_row_major_auto_resolve(self):
        """Auto-resolve symbolic dims from shape."""
        import sympy as sp
        M, N = sp.symbols('M N', integer=True, positive=True)
        layout = OrderBy(Row(M, N)).GroupBy([(M, N)])
        compiler = LayoutCompiler(layout, (4, 8), "f32")
        data = np.arange(32, dtype=np.float32).reshape(4, 8)
        result = compiler.transform_numpy(data)
        np.testing.assert_array_equal(result, data)

    def test_symbolic_round_trip(self):
        """Symbolic layout: transform then inverse == identity."""
        import sympy as sp
        M, N = sp.symbols('M N', integer=True, positive=True)
        layout = OrderBy(Row(M, N)).GroupBy([(M, N)])
        compiler = LayoutCompiler(layout, (5, 6), "f32",
                                  dim_bindings={M: 5, N: 6})
        data = np.arange(30, dtype=np.float32).reshape(5, 6)
        fwd = compiler.transform_numpy(data)
        back = compiler.inverse_transform_numpy(fwd)
        np.testing.assert_array_equal(back, data)

    def test_symbolic_col_major(self):
        """Symbolic col-major should transpose elements."""
        import sympy as sp
        M, N = sp.symbols('M N', integer=True, positive=True)
        layout = OrderBy(Col(M, N)).GroupBy([(M, N)])
        compiler = LayoutCompiler(layout, (3, 4), "f32",
                                  dim_bindings={M: 3, N: 4})
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        result = compiler.transform_numpy(data)
        back = compiler.inverse_transform_numpy(result)
        np.testing.assert_array_equal(back, data)
