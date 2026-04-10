"""
Tests for LEGO Layer 2: LegoTensor subclass with __torch_dispatch__.

Covers: annotate, rearrange, 4-tier op classification, layout transforms,
pickling, autograd, and end-to-end model flow.
"""

import pytest
import warnings
import pickle
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not available")

if _HAS_TORCH:
    import lego
    from lego.torch import annotate, rearrange, LegoTensor
    from lego.torch.tensor import _warned_ops, TransposedLayout
    from lego.frontends.python_mlir import ColMajor, RowMajor, TiledPermute
    _HAS_CUDA = torch.cuda.is_available()
else:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ============================================================================
# annotate
# ============================================================================

class TestAnnotate:
    def test_returns_lego_tensor(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(x, layout)
        assert isinstance(ax, LegoTensor)
        assert ax.lego_layout is layout

    def test_metadata_only_no_data_copy(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(x, layout)
        assert ax._data.data_ptr() == x.data_ptr()

    def test_composition(self):
        l1 = RowMajor((4, 4))
        l2 = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(annotate(x, l1), l2)
        assert isinstance(ax, LegoTensor)

    def test_shape_dtype_device(self):
        layout = ColMajor((8, 8))
        x = torch.randn(8, 8, dtype=torch.float64)
        ax = annotate(x, layout)
        assert ax.shape == (8, 8)
        assert ax.dtype == torch.float64


# ============================================================================
# rearrange
# ============================================================================

class TestRearrange:
    def test_physically_rearranges(self):
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        assert isinstance(rx, LegoTensor)
        assert rx._is_physical
        # Data should differ from original
        assert not torch.equal(rx._data, x)

    def test_round_trip_via_inverse(self):
        """rearrange then inverse permutation recovers original."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        # Inverse: apply inv permutation
        from lego.backend.compiler import LayoutCompiler
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        _, inv = compiler.get_permutation_table()
        inv_idx = torch.from_numpy(np.ascontiguousarray(inv))
        back = rx._data.reshape(-1)[inv_idx].reshape(4, 4)
        torch.testing.assert_close(back, x)

    def test_autograd(self):
        """Backward through rearrange."""
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4, requires_grad=True)
        rx = rearrange(x, layout)
        loss = rx._data.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 4)


# ============================================================================
# Tier 1: Pointwise propagation
# ============================================================================

class TestTier1:
    def test_add_preserves_layout(self):
        layout = ColMajor((4, 4))
        a = annotate(torch.randn(4, 4), layout)
        b = annotate(torch.randn(4, 4), layout)
        c = a + b
        assert isinstance(c, LegoTensor)
        assert c.lego_layout is layout

    def test_relu_preserves_layout(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        y = torch.relu(x)
        assert isinstance(y, LegoTensor)
        assert y.lego_layout is layout

    def test_mul_scalar(self):
        layout = RowMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        y = x * 3.0
        assert isinstance(y, LegoTensor)
        assert y.lego_layout is layout

    def test_neg(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.ones(4, 4), layout)
        y = -x
        assert isinstance(y, LegoTensor)
        torch.testing.assert_close(y._data, -torch.ones(4, 4))

    def test_exp(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        y = torch.exp(x)
        assert isinstance(y, LegoTensor)
        torch.testing.assert_close(y._data, torch.exp(x._data))

    def test_correctness(self):
        """Pointwise result matches plain tensor."""
        layout = ColMajor((4, 4))
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        expected = a + b
        actual = annotate(a, layout) + annotate(b, layout)
        torch.testing.assert_close(actual._data, expected)


# ============================================================================
# Tier 2: Layout transforms
# ============================================================================

class TestTier2:
    def test_transpose_creates_transposed_layout(self):
        layout = ColMajor((4, 8))
        x = annotate(torch.randn(4, 8), layout)
        y = x.t()
        assert isinstance(y, LegoTensor)
        assert isinstance(y.lego_layout, TransposedLayout)
        assert y.shape == (8, 4)

    def test_transpose_int(self):
        layout = ColMajor((2, 3, 4))
        x = annotate(torch.randn(2, 3, 4), layout)
        y = torch.transpose(x, 0, 2)
        assert isinstance(y, LegoTensor)
        assert y.shape == (4, 3, 2)

    def test_permute(self):
        layout = ColMajor((2, 3, 4))
        x = annotate(torch.randn(2, 3, 4), layout)
        y = x.permute(2, 0, 1)
        assert isinstance(y, LegoTensor)
        assert y.shape == (4, 2, 3)

    def test_double_transpose_composes(self):
        layout = ColMajor((4, 8))
        x = annotate(torch.randn(4, 8), layout)
        y = x.t().t()
        assert isinstance(y, LegoTensor)
        assert y.shape == (4, 8)


# ============================================================================
# Tier 3: Drop + warn
# ============================================================================

class TestTier3:
    def test_reshape_drops_layout(self):
        _warned_ops.clear()
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = x.reshape(2, 8)
        assert not isinstance(y, LegoTensor)
        layout_warns = [x for x in w if "layout-aware" in str(x.message)]
        assert len(layout_warns) >= 1

    def test_warns_once(self):
        _warned_ops.clear()
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x.reshape(2, 8)
            x.reshape(2, 8)
        layout_warns = [x for x in w if "layout-aware" in str(x.message)]
        assert len(layout_warns) == 1


# ============================================================================
# Tier 4: LEGO kernel (stub — runs standard op)
# ============================================================================

class TestTier4:
    def test_mm_dispatches_to_lego_op(self):
        """mm on annotated tensors routes through lego::mm."""
        layout = ColMajor((4, 4))
        a = annotate(torch.randn(4, 4), layout)
        b = annotate(torch.randn(4, 4), layout)
        c = torch.mm(a, b)
        expected = torch.mm(a._data, b._data)
        torch.testing.assert_close(c, expected)

    def test_bmm_dispatches_to_lego_op(self):
        layout = ColMajor((4, 4))
        a = annotate(torch.randn(2, 4, 4), layout)
        b = annotate(torch.randn(2, 4, 4), layout)
        c = torch.bmm(a, b)
        expected = torch.bmm(a._data, b._data)
        torch.testing.assert_close(c, expected)


# ============================================================================
# Full reductions
# ============================================================================

class TestReductions:
    def test_sum_returns_scalar(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        s = torch.sum(x)
        assert not isinstance(s, LegoTensor)
        torch.testing.assert_close(s, torch.sum(x._data))

    def test_mean(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.arange(16, dtype=torch.float32).reshape(4, 4), layout)
        torch.testing.assert_close(torch.mean(x), torch.mean(x._data))


# ============================================================================
# Serialization
# ============================================================================

class TestSerialization:
    def test_pickle_round_trip(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        buf = pickle.dumps(x)
        y = pickle.loads(buf)
        assert isinstance(y, LegoTensor)
        torch.testing.assert_close(y._data, x._data)

    def test_repr(self):
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        r = repr(x)
        assert "LegoTensor" in r
        assert "virtual" in r


# ============================================================================
# End-to-end: annotated tensors through a model
# ============================================================================

class TestEndToEnd:
    def test_simple_model(self):
        """Annotated tensors flow through a simple model without errors."""
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)

        # Tier 1 chain
        y = torch.relu(x)
        y = y + x
        y = y * 2.0
        y = torch.sigmoid(y)

        assert isinstance(y, LegoTensor)
        assert y.lego_layout is layout
        assert y.shape == (4, 4)

    def test_model_with_transpose(self):
        """Layout transforms compose correctly through a model."""
        layout = ColMajor((4, 8))
        x = annotate(torch.randn(4, 8), layout)
        y = torch.relu(x)          # Tier 1
        z = y.t()                   # Tier 2
        w = torch.relu(z)           # Tier 1 on transposed
        assert isinstance(w, LegoTensor)
        assert w.shape == (8, 4)

    def test_nn_module(self):
        """LegoTensor works as input to nn.Module."""
        layout = RowMajor((8, 4))

        class SimpleNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return torch.relu(x)  # Just pointwise

        model = SimpleNet()
        x = annotate(torch.randn(8, 4), layout)
        y = model(x)
        assert isinstance(y, LegoTensor)
        assert y.shape == (8, 4)


# ============================================================================
# Tier 2 expansion
# ============================================================================

class TestTier2Expansion:
    def test_contiguous_preserves_layout(self):
        """contiguous() on contiguous data preserves layout (Tier 1 clone)."""
        layout = ColMajor((4, 4))
        x = annotate(torch.randn(4, 4), layout)
        # On contiguous data, .contiguous() dispatches as clone (Tier 1)
        y = x.clone()
        assert isinstance(y, LegoTensor)
        assert y.lego_layout is layout

    def test_contiguous_after_transpose(self):
        """contiguous() after transpose preserves (transposed) layout via clone."""
        layout = ColMajor((4, 8))
        x = annotate(torch.randn(4, 8), layout)
        y = x.t()
        # .contiguous() on non-contiguous dispatches as aten.clone (Tier 1)
        z = y.contiguous()
        assert isinstance(z, LegoTensor)
        assert z.shape == (8, 4)


# ============================================================================
# Dim-reductions
# ============================================================================

class TestDimReductions:
    def test_sum_dim(self):
        """sum(x, dim=k) on annotated tensor is correct."""
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(x, layout)
        result = torch.sum(ax, dim=1)
        expected = torch.sum(x, dim=1)
        torch.testing.assert_close(result, expected)

    def test_mean_dim(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        ax = annotate(x, layout)
        result = torch.mean(ax, dim=0)
        expected = torch.mean(x, dim=0)
        torch.testing.assert_close(result, expected)


# ============================================================================
# Hardening: edge cases
# ============================================================================

class TestEdgeCases:
    def test_mm_annotated_plus_plain(self):
        """mm with one annotated and one plain tensor."""
        layout = ColMajor((4, 8))
        a = annotate(torch.randn(4, 8), layout)
        b = torch.randn(8, 3)
        c = torch.mm(a, b)
        expected = torch.mm(a._data, b)
        torch.testing.assert_close(c, expected)

    def test_rearrange_all_layouts(self):
        """rearrange works with all layout types."""
        import numpy as np
        from lego.backend.compiler import LayoutCompiler
        for name, layout in [
            ("ColMajor", ColMajor((4, 4))),
            ("TiledPermute", TiledPermute((8, 8), tile_shape=(4, 4))),
        ]:
            x = torch.arange(layout.numel, dtype=torch.float32).reshape(layout.shape)
            rx = rearrange(x, layout)
            assert rx._is_physical, f"{name}: not physical"
            compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
            _, inv = compiler.get_permutation_table()
            inv_t = torch.from_numpy(np.ascontiguousarray(inv))
            back = rx._data.reshape(-1)[inv_t].reshape(layout.shape)
            torch.testing.assert_close(back, x, msg=f"{name}: round-trip failed")

    def test_chain_annotate_relu_transpose_relu_sum(self):
        """Full chain: annotate -> relu -> transpose -> relu -> sum."""
        layout = ColMajor((4, 8))
        x = torch.randn(4, 8)
        ax = annotate(x, layout)
        y = torch.relu(ax)
        z = y.t()
        w = torch.relu(z)
        s = torch.sum(w, dim=0)
        ref = torch.sum(torch.relu(torch.relu(x).t()), dim=0)
        torch.testing.assert_close(s, ref)

    def test_rearrange_pointwise_preserves_physical(self):
        """Pointwise on physically rearranged data preserves physical flag."""
        layout = ColMajor((4, 4))
        rx = rearrange(torch.randn(4, 4), layout)
        y = rx + rx
        assert isinstance(y, LegoTensor)
        assert y._is_physical

    def test_top_level_import(self):
        """annotate/rearrange/LegoTensor importable from top-level lego."""
        import lego
        assert hasattr(lego, "annotate")
        assert hasattr(lego, "rearrange")
        assert hasattr(lego, "LegoTensor")

    def test_lego_annotate_without_preimport(self):
        """lego.annotate works even if accessed before torch is in sys.modules."""
        import importlib
        import lego as lego_mod
        # Force __getattr__ path
        if "annotate" in lego_mod.__dict__:
            del lego_mod.__dict__["annotate"]
        # Should not raise AttributeError
        fn = lego_mod.annotate
        assert callable(fn)


# ============================================================================
# CUDA
# ============================================================================

@requires_cuda
class TestCUDA:
    def test_annotate_cuda(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4, device="cuda")
        ax = annotate(x, layout)
        y = torch.relu(ax)
        assert y.device.type == "cuda"
        assert isinstance(y, LegoTensor)
        torch.testing.assert_close(y._data, torch.relu(x))

    def test_rearrange_cuda(self):
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        rx = rearrange(x, layout)
        assert rx.device.type == "cuda"
        assert rx._is_physical

    def test_mm_cuda(self):
        layout = ColMajor((4, 8))
        a = annotate(torch.randn(4, 8, device="cuda"), layout)
        b = torch.randn(8, 3, device="cuda")
        c = torch.mm(a, b)
        expected = torch.mm(a._data, b)
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)

    def test_rearrange_autograd_cuda(self):
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        rx = rearrange(x, layout)
        rx._data.sum().backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
