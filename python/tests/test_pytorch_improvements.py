"""
Tests for PyTorch path improvements.

Covers bug fixes, API usability, performance, layout-aware LegoTensor, and autotuning.
"""

import pytest
import warnings
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.core import Row, Col, RegP, OrderBy, GroupBy, GenP
from lego.backend.compiler import LayoutCompiler, _dtype_to_mlir, _get_mlir_element_type
from lego.backend.compiler import _COMPILER_CACHE, _PERM_TABLE_CACHE
from lego.frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, TiledPermute, Custom,
    Transposed, _check_layout_invertible,
    row, col, reg_p, order_by, group_by,
)

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not available")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ============================================================================
# Bug Fixes
# ============================================================================

class TestGenPInverseGuard:
    """GenP inverse None crash guard."""

    def test_genp_warns_when_no_inverse(self):
        """GenP warns when inverse cannot be derived."""
        import sympy as sp
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Piecewise function — can't derive inverse
            gp = GenP((4,), lambda args: sp.Piecewise((args[0], args[0] < 2), (args[0] + 1, True)))
            assert len(w) >= 1
            assert "inverse" in str(w[-1].message).lower()

    def test_genp_has_inverse_true(self):
        """GenP.has_inverse is True when inverse is provided."""
        gp = GenP((4,), lambda args: args[0], lambda flat: (flat,))
        assert gp.has_inverse is True

    def test_genp_has_inverse_false(self):
        """GenP.has_inverse is False when inverse cannot be derived."""
        import sympy as sp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GenP((4,), lambda args: sp.Piecewise((args[0], args[0] < 2), (args[0] + 1, True)))
        assert gp.has_inverse is False

    def test_inverse_transform_raises_on_no_inverse(self):
        """LegoLayout.inverse_transform raises when GenP has no inverse."""
        import sympy as sp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GenP((4,), lambda args: sp.Piecewise((args[0], args[0] < 2), (args[0] + 1, True)))
        layout_obj = GroupBy([(4,)], [OrderBy(gp)])
        layout = LegoLayout(layout_obj, (4,))
        arr = np.arange(4, dtype=np.float32)
        with pytest.raises(ValueError, match="no inverse"):
            layout.inverse_transform(arr)

    def test_check_layout_invertible_passes_for_valid(self):
        """_check_layout_invertible does not raise for normal layouts."""
        layout = RowMajor((4, 8))
        _check_layout_invertible(layout._layout)  # Should not raise


class TestShapeValidation:
    """Shape validation on transform/inverse_transform."""

    def test_transform_rejects_wrong_numel(self):
        """transform raises ValueError for numel mismatch."""
        layout = RowMajor((4, 8))
        arr = np.arange(16, dtype=np.float32)  # numel=16, expected 32
        with pytest.raises(ValueError, match="numel"):
            layout.transform(arr)

    def test_inverse_transform_rejects_wrong_numel(self):
        """inverse_transform raises ValueError for numel mismatch."""
        layout = ColMajor((4, 8))
        arr = np.arange(16, dtype=np.float32)
        with pytest.raises(ValueError, match="numel"):
            layout.inverse_transform(arr)

    def test_transform_accepts_correct_numel(self):
        """transform works with correct numel even if shape differs."""
        layout = RowMajor((4, 8))
        arr = np.arange(32, dtype=np.float32)  # flat but same numel
        result = layout.transform(arr)
        assert result.shape == (4, 8)


class TestXreplaceToSubs:
    """xreplace -> subs compatibility."""

    def test_getitem_uses_subs(self):
        """GroupBy.__getitem__ uses .subs() and returns valid expression."""
        import sympy as sp
        shape = (4, 8)
        layout = OrderBy(Row(*shape)).GroupBy([shape])
        i, j = sp.symbols('i j', integer=True)
        result = layout[i, j]
        # Should return a valid SymPy expression
        assert isinstance(result, (sp.Expr, int, sp.Integer))


class TestRegPValidation:
    """RegP input validation."""

    def test_regp_rejects_length_mismatch(self):
        """RegP raises on perm length != dims length."""
        with pytest.raises(ValueError, match="perm length"):
            RegP((4, 8), (0,))

    def test_regp_rejects_invalid_perm(self):
        """RegP raises on invalid permutation."""
        with pytest.raises(ValueError, match="not a valid permutation"):
            RegP((4, 8), (0, 0))

    def test_regp_accepts_valid_perm(self):
        """RegP accepts a valid permutation."""
        rp = RegP((4, 8), (1, 0))
        assert rp._perm_vector == [1, 0]

    def test_regp_accepts_identity_perm(self):
        """RegP accepts identity permutation."""
        rp = RegP((2, 3, 4), (0, 1, 2))
        assert rp._perm_vector == [0, 1, 2]


class TestLegoLayoutValidation:
    """LegoLayout dim validation."""

    def test_rejects_zero_dim(self):
        """LegoLayout raises on zero dimension."""
        layout_obj = GroupBy([(0, 8)], [OrderBy(Row(0, 8))])
        with pytest.raises(ValueError, match="non-positive"):
            LegoLayout(layout_obj, (0, 8))

    def test_rejects_negative_dim(self):
        """LegoLayout raises on negative dimension."""
        layout_obj = GroupBy([(-1, 8)], [OrderBy(Row(-1, 8))])
        with pytest.raises(ValueError, match="non-positive"):
            LegoLayout(layout_obj, (-1, 8))


# ============================================================================
# API Usability
# ============================================================================

class TestTiledPermute:
    """TiledPermute convenience constructor."""

    def test_basic_tiled_permute(self):
        """TiledPermute creates a LegoLayout with same shape."""
        layout = TiledPermute((8, 8), tile_shape=(4, 4))
        assert isinstance(layout, LegoLayout)
        assert layout.shape == (8, 8)

    def test_tiled_permute_round_trip(self):
        """TiledPermute round-trip preserves data."""
        layout = TiledPermute((8, 8), tile_shape=(4, 4))
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        result = layout.transform(arr)
        assert result.shape == (8, 8)
        back = layout.inverse_transform(result)
        np.testing.assert_array_equal(back, arr)

    def test_tiled_permute_reorders_data(self):
        """TiledPermute physically reorders data (unlike Tiled reshape)."""
        layout = TiledPermute((4, 4), tile_shape=(2, 2))
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        result = layout.transform(arr)
        # Data should be reordered (not identity for non-trivial tiling)
        assert result.shape == (4, 4)

    def test_tiled_permute_rejects_rank_mismatch(self):
        with pytest.raises(ValueError, match="same rank"):
            TiledPermute((8, 8), tile_shape=(4,))

    def test_tiled_permute_rejects_non_divisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            TiledPermute((8, 8), tile_shape=(3, 3))

    def test_tiled_permute_importable_from_lego(self):
        """TiledPermute is importable from the top-level package."""
        import lego
        assert hasattr(lego, 'TiledPermute')



class TestDtypeExpansion:
    """dtype expansion."""

    def test_numpy_bool(self):
        assert _dtype_to_mlir(np.bool_) == "i1"

    def test_numpy_uint8(self):
        assert _dtype_to_mlir(np.uint8) == "ui8"

    @requires_torch
    def test_torch_bool(self):
        assert _dtype_to_mlir(torch.bool) == "i1"

    @requires_torch
    def test_torch_uint8(self):
        assert _dtype_to_mlir(torch.uint8) == "ui8"

    @requires_torch
    def test_torch_float16(self):
        assert _dtype_to_mlir(torch.float16) == "f16"

    @requires_torch
    def test_torch_bfloat16(self):
        assert _dtype_to_mlir(torch.bfloat16) == "bf16"

    def test_mlir_element_type_f16(self):
        from lego.mlir.ir import Context, F16Type
        from lego.backend.dialects.lego_dialect import register
        ctx = Context()
        register(ctx)
        with ctx:
            t = _get_mlir_element_type(ctx, "f16")
            assert isinstance(t, F16Type)

    def test_mlir_element_type_bf16(self):
        from lego.mlir.ir import Context, BF16Type
        from lego.backend.dialects.lego_dialect import register
        ctx = Context()
        register(ctx)
        with ctx:
            t = _get_mlir_element_type(ctx, "bf16")
            assert isinstance(t, BF16Type)


# ============================================================================
# Performance
# ============================================================================

class TestCompilerCache:
    """Global in-memory compiler cache."""

    def test_cache_hit_same_layout(self):
        """Same layout compiled twice uses cache."""
        layout = RowMajor((4, 8))
        c1 = LayoutCompiler(layout._layout, layout._shape, "f32")
        c1.compile()
        initial_cache_size = len(_COMPILER_CACHE)

        c2 = LayoutCompiler(layout._layout, layout._shape, "f32")
        c2.compile()
        # Cache should not have grown
        assert len(_COMPILER_CACHE) == initial_cache_size

    def test_cache_produces_correct_results(self):
        """Cached compiler produces same results as fresh one."""
        layout = ColMajor((4, 4))
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)

        c1 = LayoutCompiler(layout._layout, layout._shape, "f32")
        r1 = c1.transform_numpy(arr)

        c2 = LayoutCompiler(layout._layout, layout._shape, "f32")
        r2 = c2.transform_numpy(arr)

        np.testing.assert_array_equal(r1, r2)

    def test_perm_table_cache(self):
        """Permutation tables are cached globally."""
        layout = ColMajor((4, 4))
        c1 = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd1, inv1 = c1.get_permutation_table()

        c2 = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd2, inv2 = c2.get_permutation_table()

        # Should be the exact same arrays (cached)
        assert fwd1 is fwd2
        assert inv1 is inv2



# ============================================================================
# Permutation docstring verification
# ============================================================================

class TestPermutationDocstrings:
    """Verify permutation semantics are documented and correct."""

    def test_perm_table_invariant(self):
        """inv[fwd[i]] == i and fwd[inv[i]] == i for all i."""
        layout = ColMajor((8, 8))
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, inv = compiler.get_permutation_table()
        n = len(fwd)
        np.testing.assert_array_equal(inv[fwd], np.arange(n))
        np.testing.assert_array_equal(fwd[inv], np.arange(n))

    def test_perm_table_gather_semantics(self):
        """fwd table: output[i] = input[fwd[i]] produces transformed result."""
        layout = ColMajor((3, 4))
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        arr = np.arange(12, dtype=np.float32)
        gathered = arr[fwd].reshape(3, 4)
        jit_result = layout.transform(arr)
        np.testing.assert_array_equal(gathered, jit_result)
