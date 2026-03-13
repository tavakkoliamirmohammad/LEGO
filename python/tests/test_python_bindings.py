"""
Tests for LEGO MLIR Python bindings.

Tests the MLIR Python binding infrastructure:
  - Dialect registration
  - Op construction via Python bindings
  - IR serialization round-trip
"""

import pytest
import sys
import os

# Add python dir to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDialectRegistration:
    """Test that the LEGO dialect can be registered with MLIR Python bindings."""

    @pytest.fixture
    def mlir_ctx(self):
        """Create an MLIR context with the LEGO dialect registered."""
        try:
            from mlir.ir import Context
            from lego.dialects.lego_dialect import register
        except ImportError:
            pytest.skip("MLIR Python bindings not available")

        ctx = Context()
        register(ctx)
        return ctx

    def test_register_dialect(self, mlir_ctx):
        """Test that lego dialect is loaded after registration."""
        assert mlir_ctx is not None

    def test_parse_lego_type(self, mlir_ctx):
        """Test parsing !lego.layout type."""
        from mlir.ir import Type
        with mlir_ctx:
            ty = Type.parse("!lego.layout")
            assert "lego.layout" in str(ty)

    def test_parse_lego_view_type(self, mlir_ctx):
        """Test parsing !lego.view<f32> type."""
        from mlir.ir import Type
        with mlir_ctx:
            ty = Type.parse("!lego.view<f32>")
            assert "lego.view" in str(ty)


class TestOpConstruction:
    """Test constructing LEGO ops via Python bindings."""

    @pytest.fixture
    def setup(self):
        try:
            from mlir.ir import (
                Context, Location, Module, InsertionPoint,
                IndexType, IntegerType, IntegerAttr, ArrayAttr,
                FunctionType, Operation, Type,
            )
            from mlir.dialects import func, arith
            from lego.dialects.lego_dialect import register
        except ImportError:
            pytest.skip("MLIR Python bindings not available")

        ctx = Context()
        register(ctx)
        return ctx

    def test_build_row_op(self, setup):
        """Test building a lego.row op."""
        from mlir.ir import (
            Location, Module, InsertionPoint, IndexType,
            IntegerAttr, Operation, Type,
        )
        from mlir.dialects import arith

        ctx = setup
        with ctx, Location.unknown():
            module = Module.create()
            idx_ty = IndexType.get()
            layout_ty = Type.parse("!lego.layout")

            with InsertionPoint(module.body):
                from mlir.ir import FunctionType
                from mlir.dialects import func

                func_ty = FunctionType.get([], [layout_ty])
                f = func.FuncOp("test_row", func_ty)
                entry = f.add_entry_block()

                with InsertionPoint(entry):
                    c4 = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 4))
                    c8 = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 8))

                    row = Operation.create(
                        "lego.row",
                        results=[layout_ty],
                        operands=[c4.result, c8.result],
                    )
                    func.ReturnOp([row.result])

            module_text = str(module)
            assert "lego.row" in module_text
            assert "!lego.layout" in module_text

    def test_build_reg_p_op(self, setup):
        """Test building a lego.reg_p op."""
        from mlir.ir import (
            Location, Module, InsertionPoint, IndexType,
            IntegerType, IntegerAttr, ArrayAttr, Operation, Type,
        )
        from mlir.dialects import arith

        ctx = setup
        with ctx, Location.unknown():
            module = Module.create()
            idx_ty = IndexType.get()
            i64_ty = IntegerType.get_signless(64)
            layout_ty = Type.parse("!lego.layout")

            with InsertionPoint(module.body):
                from mlir.ir import FunctionType
                from mlir.dialects import func

                func_ty = FunctionType.get([], [layout_ty])
                f = func.FuncOp("test_regp", func_ty)
                entry = f.add_entry_block()

                with InsertionPoint(entry):
                    c4 = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 4))
                    c8 = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 8))

                    perm = ArrayAttr.get([
                        IntegerAttr.get(i64_ty, 1),
                        IntegerAttr.get(i64_ty, 0),
                    ])

                    regp = Operation.create(
                        "lego.reg_p",
                        results=[layout_ty],
                        operands=[c4.result, c8.result],
                        attributes={"perm": perm},
                    )
                    func.ReturnOp([regp.result])

            module_text = str(module)
            assert "lego.reg_p" in module_text
            assert "perm" in module_text

    def test_build_apply_op(self, setup):
        """Test building lego.apply op."""
        from mlir.ir import (
            Location, Module, InsertionPoint, IndexType,
            IntegerAttr, Operation, Type, FunctionType,
        )
        from mlir.dialects import arith, func

        ctx = setup
        with ctx, Location.unknown():
            module = Module.create()
            idx_ty = IndexType.get()
            layout_ty = Type.parse("!lego.layout")

            with InsertionPoint(module.body):
                func_ty = FunctionType.get([idx_ty, idx_ty], [idx_ty])
                f = func.FuncOp("test_apply", func_ty)
                entry = f.add_entry_block()
                i, j = entry.arguments

                with InsertionPoint(entry):
                    c4 = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 4))
                    c8 = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 8))

                    row = Operation.create(
                        "lego.row",
                        results=[layout_ty],
                        operands=[c4.result, c8.result],
                    )
                    apply_op = Operation.create(
                        "lego.apply",
                        results=[idx_ty],
                        operands=[row.result, i, j],
                    )
                    func.ReturnOp([apply_op.result])

            module_text = str(module)
            assert "lego.apply" in module_text


class TestIRBuilderIntegration:
    """Test the IRBuilder from compiler.py."""

    def test_ir_builder_produces_valid_mlir(self):
        """Test that IRBuilder generates valid MLIR text."""
        try:
            from lego.compiler import IRBuilder
        except ImportError:
            pytest.skip("MLIR Python bindings not available")

        from lego.lego import Row, OrderBy, GroupBy

        # Build a simple row-major layout
        L = OrderBy(Row(4, 8)).GroupBy([(4, 8)])
        builder = IRBuilder(L, (4, 8), "f32")
        ctx, module = builder.build_module()
        with ctx:
            mlir_text = str(module)

        assert "func.func" in mlir_text
        assert "transform" in mlir_text
        assert "inverse_transform" in mlir_text
        assert "lego." in mlir_text
        assert "scf.for" in mlir_text

    def test_ir_builder_col_major(self):
        """Test IRBuilder with column-major layout."""
        try:
            from lego.compiler import IRBuilder
        except ImportError:
            pytest.skip("MLIR Python bindings not available")

        from lego.lego import Col, OrderBy

        L = OrderBy(Col(4, 8)).GroupBy([(4, 8)])
        builder = IRBuilder(L, (4, 8), "f32")
        ctx, module = builder.build_module()
        with ctx:
            mlir_text = str(module)

        assert "lego.reg_p" in mlir_text
        assert "transform" in mlir_text
