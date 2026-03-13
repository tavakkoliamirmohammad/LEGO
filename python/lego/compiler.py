"""
LEGO Layout Compiler

Builds LEGO MLIR IR programmatically using the MLIR Python bindings, then
JIT-compiles via the lego-to-llvm pipeline for efficient tensor transforms.

Uses mlir.execution_engine.ExecutionEngine for JIT (no custom C++ wrapper).
"""

import ctypes
import numpy as np
from .lego import GroupBy, OrderBy, RegP, GenP

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, MemRefType, FunctionType, IntegerAttr, StringAttr,
    IntegerType, ArrayAttr, F32Type, F64Type, UnitAttr,
)
from mlir.dialects import func as func_dialect
from mlir.dialects import arith as arith_dialect
from mlir.dialects import scf as scf_dialect
from mlir.dialects import memref as memref_dialect
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from lego.dialects.lego_dialect import register as register_lego


def _dtype_to_mlir(dtype):
    """Map numpy/torch dtype to MLIR element type string."""
    dtype_map = {
        np.float32: "f32",
        np.float64: "f64",
        np.int32: "i32",
        np.int64: "i64",
        np.int16: "i16",
        np.int8: "i8",
    }
    if hasattr(dtype, 'numpy_dtype'):
        dtype = dtype.numpy_dtype
    if hasattr(dtype, 'type'):
        dtype = dtype.type
    return dtype_map.get(dtype, "f32")


def _get_mlir_element_type(ctx, dtype_str):
    """Get MLIR Type for a dtype string within the given context."""
    type_map = {
        "f32": lambda: F32Type.get(ctx),
        "f64": lambda: F64Type.get(ctx),
        "i32": lambda: IntegerType.get_signless(32, ctx),
        "i64": lambda: IntegerType.get_signless(64, ctx),
        "i16": lambda: IntegerType.get_signless(16, ctx),
        "i8": lambda: IntegerType.get_signless(8, ctx),
    }
    return type_map.get(dtype_str, type_map["f32"])()


_MLIR_DTYPE_TO_CTYPE = {
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "i16": ctypes.c_int16,
    "i8": ctypes.c_int8,
}


# ============================================================================
# MLIR IR Builder — constructs LEGO ops using Python bindings
# ============================================================================

class IRBuilder:
    """Builds MLIR IR for LEGO layout transforms using the Python bindings.

    Constructs a complete module with @transform and @inverse_transform
    functions that use LEGO dialect ops (reg_p, order_by, group_by, apply,
    apply_inverse) and standard dialects (func, arith, scf, memref).
    """

    def __init__(self, group_by, shape, dtype_str="f32"):
        self._layout = group_by
        self._shape = shape
        self._dtype_str = dtype_str
        self._total = 1
        for s in shape:
            self._total *= int(s)

    def build_module(self):
        """Build and return (context, module) tuple.

        The context must be kept alive while the module is in use.
        """
        ctx = Context()
        register_lego(ctx)

        with ctx, Location.unknown():
            module = Module.create()
            idx_ty = IndexType.get()
            elem_ty = _get_mlir_element_type(ctx, self._dtype_str)
            memref_ty = MemRefType.get([self._total], elem_ty)

            # Build @transform
            self._build_function(
                module, "transform", idx_ty, memref_ty, forward=True
            )
            # Build @inverse_transform
            self._build_function(
                module, "inverse_transform", idx_ty, memref_ty, forward=False
            )

        return ctx, module

    def _build_function(self, module, name, idx_ty, memref_ty, forward):
        """Build a single transform function inside the module."""
        func_ty = FunctionType.get(
            [memref_ty, memref_ty, idx_ty], []
        )

        with InsertionPoint(module.body):
            f = func_dialect.FuncOp(name, func_ty)
            f.sym_visibility = StringAttr.get("public")
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        entry = f.add_entry_block()
        src, dst, n = entry.arguments

        with InsertionPoint(entry):
            # Create dimension constants
            dim_vals = []
            for s in self._shape:
                c = arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(s)))
                dim_vals.append(c.result)

            # Build the identity (row-major) layout for delinearization
            identity_perm = list(range(len(self._shape)))
            identity_reg_p = _emit_reg_p_raw(identity_perm, dim_vals, idx_ty)
            identity_layout = _emit_order_by_op([identity_reg_p])
            identity_group = _emit_group_by_op(dim_vals, [identity_layout])

            # Build the custom layout
            layout_val = self._emit_layout(self._layout, dim_vals, idx_ty)

            # Loop: scf.for %i = 0 to %n step 1
            c0 = arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 0))
            c1 = arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))

            loop = scf_dialect.ForOp(c0.result, n, c1.result)
            with InsertionPoint(loop.body):
                iv = loop.induction_variable

                if forward:
                    # Forward: row-major → custom layout
                    # Delinearize with identity, relinearize with custom
                    val = memref_dialect.LoadOp(src, [iv])
                    indices = _emit_apply_inverse(identity_group, iv, len(self._shape))
                    new_idx = _emit_apply(layout_val, indices)
                    memref_dialect.StoreOp(val.result, dst, [new_idx])
                else:
                    # Inverse: custom layout → row-major
                    # Delinearize with custom, relinearize with identity
                    val = memref_dialect.LoadOp(src, [iv])
                    indices = _emit_apply_inverse(layout_val, iv, len(self._shape))
                    new_idx = _emit_apply(identity_group, indices)
                    memref_dialect.StoreOp(val.result, dst, [new_idx])

                scf_dialect.YieldOp([])

            func_dialect.ReturnOp([])

    def _emit_layout(self, block, dim_vals, idx_ty):
        """Emit LEGO dialect ops for a layout block. Returns the layout SSA value."""
        if isinstance(block, RegP):
            return _emit_reg_p(block, dim_vals, idx_ty)
        elif isinstance(block, OrderBy):
            return self._emit_order_by(block, dim_vals, idx_ty)
        elif isinstance(block, GroupBy):
            return self._emit_group_by(block, dim_vals, idx_ty)
        else:
            raise TypeError(f"Unsupported layout block type: {type(block).__name__}")

    def _emit_order_by(self, order_by, dim_vals, idx_ty):
        """Emit lego.order_by with its sub-permutation blocks."""
        offset = 0
        perm_vals = []
        for perm in order_by.perms:
            perm_dims = perm.dims()
            count = len(perm_dims)
            sub_dims = dim_vals[offset:offset + count]
            offset += count
            perm_val = self._emit_layout(perm, sub_dims, idx_ty)
            perm_vals.append(perm_val)

        return _emit_order_by_op(perm_vals)

    def _emit_group_by(self, group_by, dim_vals, idx_ty):
        """Emit lego.group_by with its object layouts."""
        obj_vals = []
        for obj in group_by.objects:
            obj_dim_values = obj.dims()
            obj_dim_vals = []
            for d in obj_dim_values:
                c = arith_dialect.ConstantOp(
                    idx_ty, IntegerAttr.get(idx_ty, int(d))
                )
                obj_dim_vals.append(c.result)
            obj_val = self._emit_layout(obj, obj_dim_vals, idx_ty)
            obj_vals.append(obj_val)

        return _emit_group_by_op(dim_vals, obj_vals)


# ============================================================================
# Low-level op emission using MLIR Python bindings
# ============================================================================
# These use the generic Operation.create API since the auto-generated Python
# op classes may not be available yet. When the TableGen Python bindings are
# built, these can be replaced with the generated classes.

def _lego_layout_type():
    """Get the !lego.layout type from the current context."""
    from mlir.ir import Type
    return Type.parse("!lego.layout")


def _emit_reg_p_raw(perm, dim_vals, idx_ty):
    """Emit a lego.reg_p op from a raw permutation list."""
    from mlir.ir import Operation, ArrayAttr, IntegerAttr, IntegerType

    layout_ty = _lego_layout_type()
    i64_ty = IntegerType.get_signless(64)
    perm_attrs = [IntegerAttr.get(i64_ty, p) for p in perm]
    perm_array = ArrayAttr.get(perm_attrs)

    op = Operation.create(
        "lego.reg_p",
        results=[layout_ty],
        operands=dim_vals,
        attributes={"perm": perm_array},
    )
    return op.result


def _emit_reg_p(reg_p, dim_vals, idx_ty):
    """Emit a lego.reg_p op from a RegP object."""
    return _emit_reg_p_raw(reg_p._raw_perm, dim_vals, idx_ty)


def _emit_order_by_op(perm_vals):
    """Emit a lego.order_by op."""
    from mlir.ir import Operation
    layout_ty = _lego_layout_type()

    op = Operation.create(
        "lego.order_by",
        results=[layout_ty],
        operands=perm_vals,
    )
    return op.result


def _emit_group_by_op(dim_vals, obj_vals):
    """Emit a lego.group_by op."""
    from mlir.ir import Operation, DenseI32ArrayAttr
    layout_ty = _lego_layout_type()

    # group_by has AttrSizedOperandSegments: [num_group_dims, num_objects]
    segments = DenseI32ArrayAttr.get([len(dim_vals), len(obj_vals)])

    op = Operation.create(
        "lego.group_by",
        results=[layout_ty],
        operands=dim_vals + obj_vals,
        attributes={"operandSegmentSizes": segments},
    )
    return op.result


def _emit_apply(layout_val, indices):
    """Emit a lego.apply op. Returns the flat index Value."""
    from mlir.ir import Operation, IndexType
    idx_ty = IndexType.get()

    op = Operation.create(
        "lego.apply",
        results=[idx_ty],
        operands=[layout_val] + list(indices),
    )
    return op.result


def _emit_apply_inverse(layout_val, flat_index, rank):
    """Emit a lego.apply_inverse op. Returns list of index Values."""
    from mlir.ir import Operation, IndexType
    idx_ty = IndexType.get()

    op = Operation.create(
        "lego.apply_inverse",
        results=[idx_ty] * rank,
        operands=[layout_val, flat_index],
    )
    return list(op.results)


# ============================================================================
# Compiler classes
# ============================================================================

def get_compiler(layout, shape, dtype="f32"):
    """Get a LayoutCompiler for the given layout and shape."""
    if isinstance(dtype, str):
        d = dtype
    else:
        d = _dtype_to_mlir(dtype)
    return LayoutCompiler(layout, shape, d)


class LayoutCompiler:
    """Compiles a LEGO layout into a JIT-executable module.

    Uses MLIR Python bindings to construct IR programmatically, then
    JIT-compiles via the lego-to-llvm pipeline using mlir.execution_engine.
    """

    def __init__(self, layout, shape, dtype="f32"):
        if not isinstance(layout, GroupBy):
            raise TypeError(
                f"Expected GroupBy layout, got {type(layout).__name__}. "
                "Wrap your layout in GroupBy first."
            )
        self._layout = layout
        self._shape = tuple(int(s) for s in shape)
        if isinstance(dtype, str):
            self._dtype = dtype
        else:
            self._dtype = _dtype_to_mlir(dtype)
        self._ctx = None
        self._engine = None
        self._mlir_text = None

    @property
    def mlir_text(self):
        """Return the generated MLIR module text (before lowering)."""
        if self._mlir_text is None:
            builder = IRBuilder(self._layout, self._shape, self._dtype)
            _, module = builder.build_module()
            self._mlir_text = str(module)
        return self._mlir_text

    def compile(self):
        """JIT-compile the layout and return the ExecutionEngine."""
        if self._engine is None:
            builder = IRBuilder(self._layout, self._shape, self._dtype)
            self._ctx, module = builder.build_module()

            # Store pre-lowering text for debugging
            if self._mlir_text is None:
                self._mlir_text = str(module)

            # Run the lego-to-llvm pipeline
            with self._ctx:
                pm = PassManager.parse("builtin.module(lego-to-llvm)")
                pm.run(module.operation)

                # JIT compile the LLVM-dialect module
                self._engine = ExecutionEngine(module, opt_level=2)

        return self._engine

    def transform_numpy(self, arr):
        """Apply the layout transformation to a NumPy array."""
        engine = self.compile()
        total = 1
        for s in self._shape:
            total *= s

        src = np.ascontiguousarray(arr, dtype=arr.dtype).ravel()
        dst = np.empty_like(src)

        src_ptr = ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(src)))
        dst_ptr = ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(dst)))
        n = (ctypes.c_int64 * 1)(total)

        engine.invoke("transform", src_ptr, dst_ptr, n)
        return dst.reshape(arr.shape)

    def inverse_transform_numpy(self, arr):
        """Apply the inverse layout transformation to a NumPy array."""
        engine = self.compile()
        total = 1
        for s in self._shape:
            total *= s

        src = np.ascontiguousarray(arr, dtype=arr.dtype).ravel()
        dst = np.empty_like(src)

        src_ptr = ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(src)))
        dst_ptr = ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(dst)))
        n = (ctypes.c_int64 * 1)(total)

        engine.invoke("inverse_transform", src_ptr, dst_ptr, n)
        return dst.reshape(arr.shape)
