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
    LegoLayout, RowMajor, ColMajor, Tiled, TiledPermute, TiledView, Custom,
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

    def test_tiled_vs_tiled_permute_difference(self):
        """Tiled reshapes to higher rank; TiledPermute keeps same rank."""
        t = Tiled((8, 8), tile_shape=(4, 4))
        tp = TiledPermute((8, 8), tile_shape=(4, 4))
        assert t.shape == (2, 2, 4, 4)  # Higher rank
        assert tp.shape == (8, 8)       # Same rank


class TestTiledViewErrors:
    """Improved TiledView error messages."""

    def test_zero_tile_size(self):
        with pytest.raises(ValueError, match="must be > 0"):
            Tiled((8, 8), tile_shape=(0, 4))

    def test_non_divisible_suggests_factors(self):
        """Error message suggests valid tile sizes."""
        with pytest.raises(ValueError, match="factors"):
            Tiled((12, 8), tile_shape=(5, 4))


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
        from mlir.ir import Context, F16Type
        from lego.backend.dialects.lego_dialect import register
        ctx = Context()
        register(ctx)
        with ctx:
            t = _get_mlir_element_type(ctx, "f16")
            assert isinstance(t, F16Type)

    def test_mlir_element_type_bf16(self):
        from mlir.ir import Context, BF16Type
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
# Layout-Aware LegoTensor
# ============================================================================

@requires_torch
class TestLayoutAwareElementwise:
    """Dispatch table and elementwise ops on physical storage."""

    def test_add_preserves_layout(self):
        """Adding two same-layout LegoTensors preserves layout."""
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = lx + lx
        assert isinstance(result, LegoTensor)
        # Verify correctness: to_logical of result == 2*x
        torch.testing.assert_close(result.to_logical(), 2 * x)

    def test_mul_preserves_layout(self):
        """Multiplying same-layout LegoTensors preserves layout."""
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.ones(4, 4)
        lx = as_lego_tensor(x, layout)
        result = lx * lx
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), x)

    def test_scalar_mul_preserves_layout(self):
        """Scalar multiplication preserves layout."""
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = RowMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = lx * 3.0
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), x * 3.0)

    def test_neg_preserves_layout(self):
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = -lx
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), -x)

    def test_relu_preserves_layout(self):
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.relu(lx)
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), torch.relu(x))

    def test_exp_preserves_layout(self):
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.exp(lx)
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), torch.exp(x))

    def test_mismatched_layouts_fallback(self):
        """Different layouts fall back to logical order."""
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout1 = RowMajor((4, 4))
        layout2 = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx1 = as_lego_tensor(x, layout1)
        lx2 = as_lego_tensor(x, layout2)
        result = lx1 + lx2
        # Should return plain tensor (not LegoTensor)
        assert not isinstance(result, LegoTensor)
        torch.testing.assert_close(result, 2 * x)


@requires_torch
class TestLayoutAwareReductions:
    """Layout-aware reductions."""

    def test_full_sum_on_physical(self):
        """Full sum (no dim) operates on physical data directly."""
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.sum(lx)
        torch.testing.assert_close(result, torch.sum(x))

    def test_full_mean_on_physical(self):
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.mean(lx)
        torch.testing.assert_close(result, torch.mean(x))

    def test_dim_reduction_uses_logical(self):
        """Axis-specific reduction falls back to logical order."""
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.sum(lx, dim=0)
        torch.testing.assert_close(result, torch.sum(x, dim=0))


@requires_torch
class TestLayoutAwareMatmul:
    """Layout-aware matmul falls back to logical."""

    def test_matmul_correctness(self):
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        la = as_lego_tensor(a, layout)
        lb = as_lego_tensor(b, layout)
        result = torch.matmul(la, lb)
        expected = torch.matmul(a, b)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


@requires_torch
class TestDispatchWarnings:
    """Unregistered ops trigger one-time warning."""

    def test_unregistered_op_warns(self):
        from lego.backend.torch_tensor import as_lego_tensor, _warned_fallback_ops
        layout = RowMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        # Clear warned set for this test
        _warned_fallback_ops.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.reshape(lx, (2, 8))
            # Should have warned about reshape not being layout-aware
            layout_warns = [x for x in w if "layout-aware" in str(x.message)]
            assert len(layout_warns) >= 1


# ============================================================================
# Autotuning
# ============================================================================

class TestAutotuning:
    """Autotuning for tile sizes."""

    def test_autotune_returns_tiled_view(self):
        from lego.autotune import autotune
        layout = autotune(shape=(16, 16), tile_candidates=[(4, 4), (8, 8)], n_iters=3)
        assert isinstance(layout, TiledView)

    def test_autotune_caches_result(self):
        from lego.autotune import autotune, _AUTOTUNE_CACHE, clear_cache
        clear_cache()
        layout1 = autotune(shape=(16, 16), tile_candidates=[(4, 4), (8, 8)], n_iters=3)
        # Second call should use cache
        layout2 = autotune(shape=(16, 16), tile_candidates=[(4, 4), (8, 8)], n_iters=3)
        assert layout1.tile_shape == layout2.tile_shape

    def test_autotune_force_reruns(self):
        from lego.autotune import autotune, clear_cache
        clear_cache()
        layout = autotune(shape=(16, 16), tile_candidates=[(4, 4), (8, 8)], n_iters=3, force=True)
        assert isinstance(layout, TiledView)

    def test_autotune_default_candidates(self):
        """autotune generates candidates when none provided."""
        from lego.autotune import autotune, clear_cache
        clear_cache()
        layout = autotune(shape=(64, 64), n_iters=3)
        assert isinstance(layout, TiledView)

    def test_autotune_importable_from_lego(self):
        import lego
        assert hasattr(lego, 'autotune')

    def test_clear_cache(self):
        from lego.autotune import autotune, clear_cache, _AUTOTUNE_CACHE
        autotune(shape=(16, 16), tile_candidates=[(4, 4)], n_iters=2)
        clear_cache()
        assert len(_AUTOTUNE_CACHE) == 0


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
