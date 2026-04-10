"""
Stress tests for LEGO PyTorch backend.

Covers: large tensors, mixed layouts, physical data through multi-op chains,
rearrange inside torch.compile, BatchedLayout, and end-to-end model training.
"""

import pytest
import sys
import os
import numpy as np

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

if _HAS_TORCH:
    from lego.torch import annotate, rearrange, LegoTensor
    from lego.frontends.python_mlir import (
        ColMajor, RowMajor, TiledPermute,
    )


class TestPhysicalMultiOpChain:
    def test_rearrange_relu_add_sum(self):
        layout = ColMajor((8, 8))
        x = torch.randn(8, 8)
        rx = rearrange(x, layout)
        y = torch.relu(rx)
        z = y + rx
        s = torch.sum(z)
        expected = torch.sum(torch.relu(x) + x)
        torch.testing.assert_close(s, expected)

    def test_rearrange_chain_all_tiers(self):
        layout = ColMajor((4, 8))
        x = torch.randn(4, 8)
        rx = rearrange(x, layout)
        y = torch.sigmoid(rx)
        z = y.t()
        w = torch.relu(z)
        s = torch.sum(w)
        expected = torch.sum(torch.relu(torch.sigmoid(x).t()))
        torch.testing.assert_close(s, expected)

    def test_physical_pointwise_values_correct(self):
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        y = rx * 2.0
        assert isinstance(y, LegoTensor)
        assert y._is_physical
        torch.testing.assert_close(y._data, rx._data * 2.0)


class TestPhysicalDimReductionsStress:
    @pytest.mark.parametrize("layout_fn,shape", [
        (ColMajor, (4, 4)),
        (ColMajor, (8, 16)),
        (lambda s: TiledPermute(s, tile_shape=(4, 4)), (8, 8)),
    ])
    @pytest.mark.parametrize("dim", [0, 1])
    def test_sum_dim_parametric(self, layout_fn, shape, dim):
        layout = layout_fn(shape)
        x = torch.randn(*shape)
        rx = rearrange(x, layout)
        result = torch.sum(rx, dim=dim)
        expected = torch.sum(x, dim=dim)
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_mean_dim_physical(self, dim):
        layout = ColMajor((8, 4))
        x = torch.randn(8, 4)
        rx = rearrange(x, layout)
        result = torch.mean(rx, dim=dim)
        expected = torch.mean(x, dim=dim)
        torch.testing.assert_close(result, expected)


try:
    import triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

requires_triton = pytest.mark.skipif(not _HAS_TRITON, reason="Triton not available")


class TestTritonKernelStress:
    @requires_triton
    @requires_cuda
    def test_large_mm(self):
        from lego.torch.triton_kernels import triton_lego_mm
        a = torch.randn(256, 128, device="cuda")
        b = torch.randn(128, 256, device="cuda")
        result = triton_lego_mm(a, b)
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

    @requires_triton
    @requires_cuda
    def test_mm_with_tiled_layout(self):
        from lego.torch.triton_kernels import triton_lego_mm
        layout = TiledPermute((128, 64), tile_shape=(32, 32))
        a_data = torch.randn(128, 64, device="cuda")

        from lego.backend.compiler import LayoutCompiler
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to("cuda")
        a_phys = a_data.reshape(-1)[fwd_t].reshape(128, 64)

        b = torch.randn(64, 96, device="cuda")
        result = triton_lego_mm(a_phys, b, a_layout=layout)
        expected = torch.mm(a_data, b)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


class TestRearrangeInCompile:
    def test_rearrange_outside_compile(self):
        import lego.torch.compile  # noqa: F401
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)
        rx = rearrange(x, layout)

        def fn(a):
            return torch.relu(a) + 1.0

        compiled = torch.compile(fn, backend="lego")
        result = compiled(rx)
        expected = torch.relu(rx._data) + 1.0
        torch.testing.assert_close(result, expected)


class TestBatchedLayoutTorch:
    def test_batched_transform_roundtrip(self):
        from lego.frontends.python_mlir import Batched
        base = ColMajor((4, 4))
        batched = Batched(base, batch_shape=(3,))
        x = torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)
        transformed = batched.transform(x)
        back = batched.inverse_transform(transformed)
        torch.testing.assert_close(back, x)


class TestEndToEndTraining:
    def test_annotated_linear_forward_backward(self):
        layout = RowMajor((8, 4))
        model = torch.nn.Linear(4, 3)
        x = annotate(torch.randn(8, 4), layout)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert model.weight.grad is not None
        assert model.weight.grad.shape == (3, 4)

    def test_multi_layer_model(self):
        layout = RowMajor((16, 8))

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(8, 16)
                self.fc2 = torch.nn.Linear(16, 4)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = Net()
        x = annotate(torch.randn(16, 8), layout)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert model.fc1.weight.grad is not None
        assert model.fc2.weight.grad is not None

    @requires_cuda
    def test_cuda_training_loop(self):
        layout = RowMajor((32, 16))
        model = torch.nn.Linear(16, 4).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(5):
            x = annotate(torch.randn(32, 16, device="cuda"), layout)
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert model.weight.grad is not None


class TestPhysicalMixedOps:
    """Tests for physical data mixed with plain tensors (correctness bug fixes)."""

    def test_physical_plus_plain(self):
        """Physical tensor + plain tensor must give correct result."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        y = torch.ones(4, 4) * 10
        result = rx + y
        expected = x + y
        # Result should NOT be LegoTensor since one operand was plain
        if isinstance(result, LegoTensor):
            result = result._data
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_physical_mul_plain(self):
        """Physical tensor * plain tensor must give correct result."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        y = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        result = rx * y
        expected = x * y
        if isinstance(result, LegoTensor):
            result = result._data
        torch.testing.assert_close(result, expected)

    def test_tier3_on_physical_returns_logical_order(self):
        """Tier 3 op on physical data should return data in logical order."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        import warnings
        from lego.torch.tensor import _warned_ops
        _warned_ops.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rx.reshape(2, 8)
        # Result should be in logical order (same as x.reshape(2,8))
        torch.testing.assert_close(result, x.reshape(2, 8))

    def test_amax_dim_on_physical(self):
        """amax with dim on physical data must give correct result."""
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)
        result = torch.amax(rx, dim=0)
        expected = torch.amax(x, dim=0)
        torch.testing.assert_close(result, expected)
