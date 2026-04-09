"""
Tests for LEGO Layer 3: torch.compile backend.

Covers: backend registration, compiled execution with annotated tensors,
layout planner propagation, and fusion infrastructure.
"""

import pytest
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
    from lego.torch import annotate, LegoTensor
    from lego.torch.fusion import make_index_function, make_index_tensor
    from lego.frontends.python_mlir import ColMajor, RowMajor


# ============================================================================
# Backend registration
# ============================================================================

class TestBackendRegistration:
    def test_backend_registered(self):
        """The 'lego' backend is discoverable by torch.compile."""
        import lego.torch.compile  # noqa: F401  — triggers registration
        # torch._dynamo lists registered backends
        from torch._dynamo import list_backends
        assert "lego" in list_backends()


# ============================================================================
# Compiled execution
# ============================================================================

class TestCompiledExecution:
    def test_pointwise_compiled(self):
        """Compiled pointwise on annotated tensor produces correct result."""
        import lego.torch.compile  # noqa: F401

        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)

        def fn(a):
            return torch.relu(a) + 1.0

        compiled = torch.compile(fn, backend="lego")
        # Run on plain tensor first for reference
        expected = fn(x)
        # Run on annotated tensor
        ax = annotate(x, layout)
        result = compiled(ax)
        torch.testing.assert_close(result, expected)

    def test_multi_op_compiled(self):
        """Chain of ops through compiled backend."""
        import lego.torch.compile  # noqa: F401

        layout = RowMajor((8, 4))
        x = torch.randn(8, 4)

        def fn(a):
            y = torch.sigmoid(a)
            y = y * 2.0
            return y + a

        compiled = torch.compile(fn, backend="lego")
        expected = fn(x)
        result = compiled(annotate(x, layout))
        torch.testing.assert_close(result, expected)


# ============================================================================
# Fusion infrastructure (Path C)
# ============================================================================

class TestFusion:
    def test_make_index_function(self):
        """Index function returns valid permutation tables."""
        layout = ColMajor((4, 4))
        fwd, inv = make_index_function(layout)
        # fwd and inv are inverse of each other
        import numpy as np
        assert np.array_equal(inv[fwd], np.arange(16))
        assert np.array_equal(fwd[inv], np.arange(16))

    def test_make_index_tensor(self):
        """Index tensors are on the correct device."""
        layout = ColMajor((4, 4))
        fwd_t, inv_t = make_index_tensor(layout, device="cpu")
        assert fwd_t.device == torch.device("cpu")
        assert fwd_t.shape == (16,)


# ============================================================================
# Layout planner
# ============================================================================

class TestPlanner:
    def test_runs_without_error(self):
        """Planner executes on a traced graph without crashing."""
        from lego.torch.planner import plan_layouts

        layout = ColMajor((4, 4))

        def fn(x):
            return torch.relu(x) + 1.0

        gm = torch.fx.symbolic_trace(fn)
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        layout_map = {placeholders[0].name: layout}
        # Should not raise
        plan_layouts(gm, layout_map)

    def test_propagates_aten_ops(self):
        """Planner propagates layout through decomposed aten ops."""
        from lego.torch.planner import plan_layouts
        from lego.torch.tensor import _TIER1

        layout = ColMajor((4, 4))

        # Build a graph with aten ops directly
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu_node = graph.call_function(torch.ops.aten.relu.default, (x,))
        graph.output(relu_node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layout_map = {"x": layout}
        plan_layouts(gm, layout_map)
        assert relu_node.name in layout_map
