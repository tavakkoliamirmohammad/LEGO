"""Tests for LEGO torch autotune."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not available")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


class TestAutotune:
    def test_autotune_returns_layout(self):
        from lego.torch.autotune import autotune
        layout = autotune(shape=(16, 16), device="cpu", n_iters=2)
        from lego.frontends.python_mlir import LegoLayout
        assert isinstance(layout, LegoLayout)

    def test_autotune_candidates(self):
        from lego.torch.autotune import autotune
        layout = autotune(
            shape=(16, 16),
            tile_candidates=[(4, 4), (8, 8)],
            device="cpu",
            n_iters=2,
        )
        from lego.frontends.python_mlir import LegoLayout
        assert isinstance(layout, LegoLayout)

    def test_autotune_cache(self):
        from lego.torch.autotune import autotune, clear_cache
        clear_cache()
        l1 = autotune(shape=(16, 16), device="cpu", n_iters=2)
        l2 = autotune(shape=(16, 16), device="cpu", n_iters=2)
        assert l1._shape == l2._shape

    @requires_cuda
    def test_autotune_cuda(self):
        from lego.torch.autotune import autotune
        layout = autotune(shape=(64, 64), device="cuda", n_iters=3)
        from lego.frontends.python_mlir import LegoLayout
        assert isinstance(layout, LegoLayout)
