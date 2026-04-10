"""
Tests for LEGO Layer 1: torch.library custom ops (lego::mm, lego::bmm).

Covers: registration, eager correctness, fake tensors, autograd.
"""

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


@pytest.fixture(autouse=True)
def _register_ops():
    """Ensure lego::mm and lego::bmm are registered."""
    import lego.torch.ops  # noqa: F401


class TestLegoMM:
    def test_eager_cpu(self):
        a = torch.randn(4, 8)
        b = torch.randn(8, 3)
        result = torch.ops.lego.mm(a, b)
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected)

    @requires_cuda
    def test_eager_cuda(self):
        a = torch.randn(4, 8, device="cuda")
        b = torch.randn(8, 3, device="cuda")
        result = torch.ops.lego.mm(a, b)
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected)

    def test_autograd(self):
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(8, 3, requires_grad=True)
        c = torch.ops.lego.mm(a, b)
        loss = c.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == (4, 8)
        assert b.grad.shape == (8, 3)

    def test_autograd_correctness(self):
        """Gradients match torch.mm."""
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(8, 3, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        torch.ops.lego.mm(a, b).sum().backward()
        torch.mm(a2, b2).sum().backward()

        torch.testing.assert_close(a.grad, a2.grad)
        torch.testing.assert_close(b.grad, b2.grad)

    def test_fake_tensor(self):
        """Fake-tensor impl infers correct shape/dtype."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            a = torch.randn(4, 8)
            b = torch.randn(8, 3)
            c = torch.ops.lego.mm(a, b)
            assert c.shape == (4, 3)
            assert c.dtype == torch.float32


class TestLegoBMM:
    def test_eager_cpu(self):
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 8, 3)
        result = torch.ops.lego.bmm(a, b)
        expected = torch.bmm(a, b)
        torch.testing.assert_close(result, expected)

    def test_autograd(self):
        a = torch.randn(2, 4, 8, requires_grad=True)
        b = torch.randn(2, 8, 3, requires_grad=True)
        c = torch.ops.lego.bmm(a, b)
        c.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_fake_tensor(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            a = torch.randn(2, 4, 8)
            b = torch.randn(2, 8, 3)
            c = torch.ops.lego.bmm(a, b)
            assert c.shape == (2, 4, 3)


class TestTorchOp:
    def test_register_custom_op(self):
        """@lego.torch_op registers a callable custom op."""
        from lego.torch import torch_op

        @torch_op("lego::test_add")
        def test_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        result = torch.ops.lego.test_add(torch.ones(2), torch.ones(2))
        torch.testing.assert_close(result, torch.full((2,), 2.0))

    def test_torch_op_fake_tensor(self):
        """Custom op works with FakeTensorMode via shape inference."""
        from lego.torch import torch_op

        @torch_op("lego::test_scale")
        def test_scale(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.randn(3, 4)
            y = torch.ops.lego.test_scale(x)
            assert y.shape == (3, 4)


class TestTorchOpFake:
    def test_torch_op_with_explicit_fake(self):
        """@torch_op with explicit fake= parameter."""
        from lego.torch import torch_op

        @torch_op("lego::test_custom_fake", fake=lambda x: torch.empty_like(x))
        def test_custom_fake(x: torch.Tensor) -> torch.Tensor:
            return x * 2.0 + 1.0

        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.randn(3, 4)
            y = torch.ops.lego.test_custom_fake(x)
            assert y.shape == (3, 4)

    def test_torch_op_default_fake_still_works(self):
        """@torch_op without fake= still works for simple ops."""
        from lego.torch import torch_op

        @torch_op("lego::test_simple_default2")
        def test_simple_default2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            c = torch.ops.lego.test_simple_default2(a, b)
            assert c.shape == (2, 3)
