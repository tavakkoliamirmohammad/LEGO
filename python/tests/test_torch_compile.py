"""
Tests for LEGO Layer 3: torch.compile backend.

Covers: backend registration, compiled execution with annotated tensors,
layout planner propagation, fusion infrastructure, Path B/C, and the
spec's end-to-end example.
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
    from lego.frontends.python_mlir import ColMajor, RowMajor, TiledPermute


# ============================================================================
# Backend registration
# ============================================================================

class TestBackendRegistration:
    def test_backend_registered(self):
        """The 'lego' backend is discoverable by torch.compile."""
        import lego.torch.compile  # noqa: F401
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
        expected = fn(x)
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
        import numpy as np
        layout = ColMajor((4, 4))
        fwd, inv = make_index_function(layout)
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
        plan_layouts(gm, layout_map)

    def test_propagates_aten_ops(self):
        """Planner propagates layout through decomposed aten ops."""
        from lego.torch.planner import plan_layouts

        layout = ColMajor((4, 4))

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu_node = graph.call_function(torch.ops.aten.relu.default, (x,))
        graph.output(relu_node)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layout_map = {"x": layout}
        plan_layouts(gm, layout_map)
        assert relu_node.name in layout_map


# ============================================================================
# Planner cost model
# ============================================================================

class TestPlannerCostModel:
    def test_cost_model_identity_layout(self):
        """Identity layout (RowMajor) has zero rearrangement cost."""
        from lego.torch.planner import layout_cost
        layout = RowMajor((4, 4))
        assert layout_cost(layout) == 0

    def test_cost_model_nonidentity(self):
        """Non-identity layout has positive cost."""
        from lego.torch.planner import layout_cost
        layout = ColMajor((4, 4))
        assert layout_cost(layout) > 0

    def test_planner_propagates_layout_through_tier1(self):
        """Tier 1 ops propagate layout in planner."""
        from lego.torch.planner import plan_layouts
        layout = ColMajor((4, 4))

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, (x,))
        add = graph.call_function(torch.ops.aten.add.Tensor, (relu, relu))
        graph.output(add)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        layout_map = {"x": layout}
        plan_layouts(gm, layout_map)
        assert relu.name in layout_map
        assert add.name in layout_map


# ============================================================================
# Path B: lego::mm inductor lowering
# ============================================================================

class TestPathB:
    def test_lego_mm_compiles(self):
        """lego::mm is compilable through torch.compile."""
        import lego.torch.compile  # noqa: F401
        import lego.torch.ops  # noqa: F401

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)

        def fn(x, y):
            return torch.ops.lego.mm(x, y)

        compiled = torch.compile(fn, backend="lego")
        result = compiled(a, b)
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected)


# ============================================================================
# Path C: Layout materialization
# ============================================================================

class TestPathC:
    def test_virtual_annotation_compiled_correctness(self):
        """Virtually-annotated tensor produces correct compiled result."""
        import lego.torch.compile  # noqa: F401
        layout = ColMajor((4, 4))
        x = torch.randn(4, 4)

        def fn(a):
            return torch.relu(a) * 2.0

        compiled = torch.compile(fn, backend="lego")
        expected = fn(x)
        result = compiled(annotate(x, layout))
        torch.testing.assert_close(result, expected)

    def test_physical_rearrange_compiled(self):
        """Physically-rearranged tensor flows through compiled graph."""
        import lego.torch.compile  # noqa: F401
        from lego.torch import rearrange
        layout = ColMajor((4, 4))
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        rx = rearrange(x, layout)

        def fn(a):
            return a + 1.0

        compiled = torch.compile(fn, backend="lego")
        result = compiled(rx)
        expected = rx._data + 1.0
        torch.testing.assert_close(result, expected)


# ============================================================================
# Spec end-to-end example
# ============================================================================

class TestSpecEndToEnd:
    def test_eager_mode(self):
        """Spec end-to-end example: eager mode (Layers 1 + 2)."""
        A = annotate(torch.randn(64, 32),
                     layout=TiledPermute((64, 32), tile_shape=(8, 8)))
        B = torch.randn(32, 48)

        class Model(torch.nn.Module):
            def forward(self, a, b):
                x = torch.mm(a, b)        # Tier 4 -> lego::mm
                x = torch.relu(x)         # Tier 1 -> propagate layout
                x = torch.sum(x, dim=1)   # dim-reduction
                return x

        model = Model()
        out = model(A, B)
        expected = model(A._data, B)
        torch.testing.assert_close(out, expected)

    def test_compiled_mode(self):
        """Spec end-to-end example: compiled mode (all 3 layers)."""
        import lego.torch.compile  # noqa: F401

        A_data = torch.randn(64, 32)
        A = annotate(A_data, layout=TiledPermute((64, 32), tile_shape=(8, 8)))
        B = torch.randn(32, 48)

        def fn(a, b):
            x = torch.relu(a)
            return x + a

        compiled = torch.compile(fn, backend="lego")
        expected = fn(A_data, B)
        result = compiled(A, B)
        torch.testing.assert_close(result, expected)
