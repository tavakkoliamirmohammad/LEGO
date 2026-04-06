"""Tests for the ROCDL/AMD GPU compilation backend.

Tests verify that LEGO layouts compile through the lego-to-rocdl pipeline
to produce valid AMDGPU assembly.  No AMD GPU hardware required — all tests
are pure cross-compilation.
"""

import os
import pytest

from lego.core import Row, Col, OrderBy
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _has_lego_to_rocdl_pipeline():
    """Check if the lego-to-rocdl pipeline is available."""
    try:
        from lego.mlir.ir import Context, Location
        from lego.mlir.passmanager import PassManager
        from lego.backend.dialects.lego_dialect import register
        ctx = Context()
        register(ctx)
        with ctx, Location.unknown():
            pm = PassManager.parse("builtin.module(lego-to-rocdl)")
        return True
    except Exception:
        return False


requires_rocdl = pytest.mark.skipif(
    not _has_lego_to_rocdl_pipeline(),
    reason="lego-to-rocdl pipeline not available (AMDGPU target not built)",
)


# ══════════════════════════════════════════════════════════════════════════
# Tier 1: Pipeline lowering — lego-to-rocdl produces AMDGPU binary
# ══════════════════════════════════════════════════════════════════════════

class TestROCDLPipeline:
    """Test that the lego-to-rocdl pipeline produces valid AMDGPU output."""

    @requires_rocdl
    def test_row_layout_compiles(self):
        """Simple Row layout → ROCDL."""
        import lego
        result = lego.compile(Row(256), shape=(256,), target="rocm")
        assert isinstance(result, lego.CompileResult)
        assert result.target == "rocm"
        assert "gpu.binary" in result.kernel_source
        assert "rocdl.target" in result.kernel_source
        assert "amdgcn" in result.kernel_source

    @requires_rocdl
    def test_col_layout_compiles(self):
        """Col layout → ROCDL."""
        import lego
        result = lego.compile(Col(16, 16), shape=(16, 16), target="rocm")
        assert result.target == "rocm"
        assert "gpu.binary" in result.kernel_source
        assert "rocdl.target" in result.kernel_source

    @requires_rocdl
    def test_2d_row_layout_compiles(self):
        """2D Row layout → ROCDL."""
        import lego
        result = lego.compile(Row(8, 8), shape=(8, 8), target="rocm")
        assert result.target == "rocm"

    @requires_rocdl
    def test_tileby_layout_compiles(self):
        """TileBy layout → ROCDL."""
        import lego
        layout = OrderBy(Row(8, 8)).TileBy([4, 4], [2, 2])
        result = lego.compile(layout, shape=(8, 8), target="rocm")
        assert result.target == "rocm"
        assert result.kernel_source

    @requires_rocdl
    def test_output_contains_amdgcn_target(self):
        """The compiled output should reference the AMDGPU target."""
        import lego
        result = lego.compile(Row(64), shape=(64,), target="rocm")
        assert "amdgcn" in result.kernel_source or "rocdl" in result.kernel_source

    @requires_rocdl
    def test_metadata_contains_pipeline(self):
        """CompileResult metadata should record the pipeline used."""
        import lego
        result = lego.compile(Row(64), shape=(64,), target="rocm")
        assert result.metadata.get("pipeline") == "lego-to-rocdl"


# ══════════════════════════════════════════════════════════════════════════
# Tier 2: KernelBuilder — multi-buffer kernels via ROCDL
# ══════════════════════════════════════════════════════════════════════════

class TestROCDLKernelBuilder:
    """Test KernelBuilder compiles user-defined kernels through ROCDL."""

    @requires_rocdl
    def test_vecadd_compiles(self):
        """Vector addition kernel → ROCDL."""
        from lego.backend.gpu_builder import make_elementwise_kernel

        a = LayoutBuffer(Row(64), shape=(64,))
        b = LayoutBuffer(Row(64), shape=(64,))
        c = LayoutBuffer(Row(64), shape=(64,))

        def vecadd(ctx, gid):
            va = ctx.load_flat(0, gid)
            vb = ctx.load_flat(1, gid)
            ctx.store_flat(ctx.addf(va, vb), 2, gid)

        builder = make_elementwise_kernel([a, b, c], vecadd, name="vecadd")
        result = builder.compile(target="rocm")
        assert result.target == "rocm"
        assert result.kernel_source

    @requires_rocdl
    def test_transpose_kernel_compiles(self):
        """Transpose-style kernel: load Row, store Col → ROCDL."""
        src = LayoutBuffer(Row(4, 4), shape=(4, 4))
        dst = LayoutBuffer(Col(4, 4), shape=(4, 4))

        def transpose_body(ctx):
            gid = KernelBuilder.global_id_1d(ctx)
            val = ctx.load_flat(0, gid)
            i, j = ctx.apply_inverse(Row(4, 4), gid)
            ctx.store(val, 1, [i, j])

        builder = KernelBuilder(
            buffers=[src, dst], kernel_body=transpose_body, name="transpose",
        )
        result = builder.compile(target="rocm")
        assert result.target == "rocm"
        assert os.path.exists(result.kernel_path)

    @requires_rocdl
    def test_custom_output_dir(self):
        """Compile with custom output directory."""
        import tempfile
        import lego
        with tempfile.TemporaryDirectory() as d:
            result = lego.compile(
                Row(64), shape=(64,), target="rocm",
                output_dir=d, name="my_amd_kernel",
            )
            assert "my_amd_kernel" in result.kernel_path
            assert os.path.isfile(result.kernel_path)
