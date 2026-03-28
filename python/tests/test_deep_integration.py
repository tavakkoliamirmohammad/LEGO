"""
Tests for deep PyTorch integration.

Covers:
  - SymPy-to-PyTorch compiler (sympy_to_torch)
  - Compiled layout transforms matching MLIR JIT output
  - LegoTensor minimal state (no _fwd_perm/_inv_perm)
  - Name-based dispatch
  - Autograd through compiled transforms
  - torch.compile backend registration
"""

import pytest
import warnings
import numpy as np
import sympy as sp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lego.core import Row, Col, RegP, OrderBy, GroupBy, GenP
from lego.frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, Tiled, TiledPermute, Custom,
    Transposed, ZCurve, Swizzle, BlockCyclic,
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
# SymPy-to-PyTorch Compiler
# ============================================================================

@requires_torch
class TestSympyToTorch:
    """Test sympy_to_torch compiles expressions correctly."""

    def test_linear(self):
        from lego.backend.torch_layout import sympy_to_torch
        i, j = sp.symbols('i j')
        fn = sympy_to_torch(8*i + j, [i, j])
        x = torch.tensor([0, 1, 2, 3])
        y = torch.tensor([0, 1, 2, 3])
        result = fn(x, y)
        expected = 8 * x + y
        torch.testing.assert_close(result, expected)

    def test_floor_div(self):
        from lego.backend.torch_layout import sympy_to_torch
        x = sp.Symbol('x')
        fn = sympy_to_torch(sp.floor(x / 8), [x])
        vals = torch.tensor([0, 7, 8, 15, 16, 31])
        expected = torch.tensor([0, 0, 1, 1, 2, 3])
        torch.testing.assert_close(fn(vals), expected)

    def test_mod(self):
        from lego.backend.torch_layout import sympy_to_torch
        x = sp.Symbol('x')
        fn = sympy_to_torch(sp.Mod(x, 4), [x])
        vals = torch.tensor([0, 1, 4, 5, 11])
        expected = torch.tensor([0, 1, 0, 1, 3])
        torch.testing.assert_close(fn(vals), expected)

    def test_piecewise(self):
        from lego.backend.torch_layout import sympy_to_torch
        x = sp.Symbol('x')
        expr = sp.Piecewise((x, x < 4), (x + 10, True))
        fn = sympy_to_torch(expr, [x])
        vals = torch.tensor([0, 2, 4, 6])
        expected = torch.tensor([0, 2, 14, 16])
        torch.testing.assert_close(fn(vals), expected)

    def test_constant(self):
        from lego.backend.torch_layout import sympy_to_torch
        x = sp.Symbol('x')
        fn = sympy_to_torch(sp.Integer(42), [x])
        vals = torch.tensor([1, 2, 3])
        result = fn(vals)
        assert result == 42


# ============================================================================
# Compiled Transforms Match MLIR
# ============================================================================

@requires_torch
class TestCompiledTransformCorrectness:
    """Verify compiled PyTorch transforms produce identical results to MLIR JIT."""

    def _compare(self, layout, shape):
        """Compare compiled PyTorch path vs MLIR numpy path."""
        arr = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        mlir_fwd = layout.transform(arr)

        t = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        torch_fwd = layout.transform(t)

        np.testing.assert_array_equal(mlir_fwd, torch_fwd.numpy())

        # Test round-trip
        mlir_inv = layout.inverse_transform(mlir_fwd)
        torch_inv = layout.inverse_transform(torch_fwd)
        np.testing.assert_array_equal(mlir_inv, torch_inv.numpy())
        np.testing.assert_array_equal(arr, torch_inv.numpy())

    def test_row_major(self):
        self._compare(RowMajor((4, 8)), (4, 8))

    def test_col_major(self):
        self._compare(ColMajor((4, 8)), (4, 8))

    def test_transposed(self):
        self._compare(Transposed((3, 4)), (3, 4))

    def test_tiled_permute(self):
        self._compare(TiledPermute((8, 8), (4, 4)), (8, 8))

    def test_zcurve(self):
        self._compare(ZCurve((4, 4)), (4, 4))

    def test_swizzle(self):
        self._compare(Swizzle((4, 4), num_bits=2), (4, 4))

    def test_blockcyclic(self):
        self._compare(BlockCyclic((16,), block_size=2, num_procs=2), (16,))

    def test_3d_layout(self):
        layout = LegoLayout(GroupBy([(2, 3, 4)], [OrderBy(Row(2, 3, 4))]))
        self._compare(layout, (2, 3, 4))

    def test_col_3d(self):
        layout = LegoLayout(GroupBy([(2, 3, 4)], [OrderBy(Col(2, 3, 4))]))
        self._compare(layout, (2, 3, 4))


# ============================================================================
# LegoTensor Minimal State
# ============================================================================

@requires_torch
class TestLegoTensorMinimal:
    """Verify LegoTensor stores only layout reference, no perm tables."""

    def test_no_fwd_perm_attribute(self):
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        assert not hasattr(lx, '_fwd_perm')
        assert not hasattr(lx, '_inv_perm')

    def test_has_layout_ref(self):
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        assert lx._lego_layout is layout

    def test_to_logical_roundtrip(self):
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        back = lx.to_logical()
        torch.testing.assert_close(back, x)


# ============================================================================
# Name-Based Dispatch
# ============================================================================

@requires_torch
class TestNameBasedDispatch:
    """Verify name-based op classification works correctly."""

    def test_elementwise_preserves_layout(self):
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = lx + lx
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), 2 * x)

    def test_relu_preserves_layout(self):
        from lego.backend.torch_tensor import as_lego_tensor, LegoTensor
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.relu(lx)
        assert isinstance(result, LegoTensor)
        torch.testing.assert_close(result.to_logical(), torch.relu(x))

    def test_full_reduction(self):
        from lego.backend.torch_tensor import as_lego_tensor
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        result = torch.sum(lx)
        torch.testing.assert_close(result, torch.sum(x))

    def test_fallback_warns(self):
        from lego.backend.torch_tensor import as_lego_tensor, _warned_fallback_ops
        layout = RowMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        lx = as_lego_tensor(x, layout)
        _warned_fallback_ops.clear()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.reshape(lx, (2, 8))
            layout_warns = [x for x in w if "layout-aware" in str(x.message)]
            assert len(layout_warns) >= 1


# ============================================================================
# Autograd Through Compiled Transforms
# ============================================================================

@requires_torch
class TestAutogradCompiledTransform:
    """Verify gradients flow through compiled layout transforms."""

    def test_transform_gradient(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4, requires_grad=True)
        transformed = layout.transform(x)
        loss = transformed.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 4)

    def test_roundtrip_gradient(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4, requires_grad=True)
        transformed = layout.transform(x)
        back = layout.inverse_transform(transformed)
        loss = back.sum()
        loss.backward()
        # Round-trip should produce all-ones gradient
        torch.testing.assert_close(x.grad, torch.ones(4, 4))

    @requires_cuda
    def test_cuda_transform(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4, device="cuda")
        result = layout.transform(x)
        assert result.device.type == "cuda"
        back = layout.inverse_transform(result)
        torch.testing.assert_close(back, x)


# ============================================================================
# torch.compile Backend
# ============================================================================

@requires_torch
class TestCompileBackend:
    """Verify torch.compile "lego" backend is registered and functional."""

    def test_backend_registered(self):
        """The 'lego' backend is available after import lego."""
        import lego  # triggers registration
        # torch._dynamo has the backend
        from torch._dynamo import list_backends
        backends = list_backends()
        assert 'lego' in backends

    def test_compile_basic(self):
        """torch.compile with lego backend produces correct results."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        @torch.compile(backend="lego")
        def fn(t):
            return t * 2

        result = fn(x)
        torch.testing.assert_close(result, x * 2)
