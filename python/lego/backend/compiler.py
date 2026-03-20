"""
LEGO Layout Compiler

- IRBuilder: constructs LEGO dialect IR from layout objects
- LayoutCompiler: JIT-compiles via lego-to-llvm pipeline
"""

import ctypes
import sys
import numpy as np

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, MemRefType, FunctionType, IntegerAttr, StringAttr,
    IntegerType, F32Type, F64Type, UnitAttr,
)
from mlir.dialects import func as func_dialect
from mlir.dialects import scf as scf_dialect
from mlir.dialects import memref as memref_dialect
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from lego.backend.dialects.lego_dialect import register as register_lego
from lego.backend._ops import (
    _LEGO_DEBUG,
    _index_const, _emit_row, _emit_order_by, _emit_group_by,
    _emit_apply, _emit_apply_inverse,
)


# ============================================================================
# Dtype mapping
# ============================================================================

def _dtype_to_mlir(dtype):
    """Map numpy/torch dtype to MLIR element type string."""
    dtype_map = {
        np.float32: "f32", np.float64: "f64",
        np.int32: "i32", np.int64: "i64", np.int16: "i16", np.int8: "i8",
    }
    try:
        import torch
        dtype_map.update({
            torch.float32: "f32", torch.float64: "f64",
            torch.float16: "f16", torch.bfloat16: "bf16",
            torch.int32: "i32", torch.int64: "i64",
            torch.int16: "i16", torch.int8: "i8",
        })
    except ImportError:
        pass
    if hasattr(dtype, 'numpy_dtype'):
        dtype = dtype.numpy_dtype
    if hasattr(dtype, 'type'):
        dtype = dtype.type
    return dtype_map.get(dtype, "f32")


def _get_mlir_element_type(ctx, dtype_str):
    type_map = {
        "f32": lambda: F32Type.get(ctx),
        "f64": lambda: F64Type.get(ctx),
        "i32": lambda: IntegerType.get_signless(32, ctx),
        "i64": lambda: IntegerType.get_signless(64, ctx),
        "i16": lambda: IntegerType.get_signless(16, ctx),
        "i8": lambda: IntegerType.get_signless(8, ctx),
    }
    return type_map.get(dtype_str, type_map["f32"])()


# ============================================================================
# IR Builder — emits LEGO dialect IR from core layout objects
# ============================================================================

def _get_layout_dims(layout):
    """Get dims as tuple of ints from a layout object."""
    dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
    return tuple(int(d) for d in dims)


class IRBuilder:
    """Builds a complete MLIR module with @transform / @inverse_transform."""

    def __init__(self, layout, shape, dtype_str="f32"):
        self._layout = layout
        self._shape = shape
        self._dtype_str = dtype_str
        self._total = 1
        for s in shape:
            self._total *= int(s)

    def build_module(self):
        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            idx_ty = IndexType.get()
            elem_ty = _get_mlir_element_type(ctx, self._dtype_str)
            memref_ty = MemRefType.get([self._total], elem_ty)
            self._build_function(module, "transform", idx_ty, memref_ty, forward=True)
            self._build_function(module, "inverse_transform", idx_ty, memref_ty, forward=False)
        return ctx, module

    def _build_function(self, module, name, idx_ty, memref_ty, forward):
        func_ty = FunctionType.get([memref_ty, memref_ty, idx_ty], [])
        with InsertionPoint(module.body):
            f = func_dialect.FuncOp(name, func_ty)
            f.sym_visibility = StringAttr.get("public")
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        entry = f.add_entry_block()
        src, dst, n = entry.arguments

        with InsertionPoint(entry):
            from lego.backend.symbolic import emit_layout_from_python

            layout_dims = _get_layout_dims(self._layout)
            dim_vals = [_index_const(s) for s in layout_dims]
            rank = len(layout_dims)

            identity = _emit_group_by(dim_vals, [_emit_order_by([_emit_row(dim_vals)])])
            layout_val = emit_layout_from_python(self._layout, {})

            loop = scf_dialect.ForOp(_index_const(0), n, _index_const(1))
            with InsertionPoint(loop.body):
                iv = loop.induction_variable
                if forward:
                    val = memref_dialect.LoadOp(src, [iv])
                    indices = _emit_apply_inverse(layout_val, iv, rank)
                    memref_dialect.StoreOp(val.result, dst, [_emit_apply(identity, indices)])
                else:
                    val = memref_dialect.LoadOp(src, [iv])
                    indices = _emit_apply_inverse(identity, iv, rank)
                    memref_dialect.StoreOp(val.result, dst, [_emit_apply(layout_val, indices)])
                scf_dialect.YieldOp([])
            func_dialect.ReturnOp([])


# ============================================================================
# JIT Compiler
# ============================================================================

def get_compiler(layout, shape, dtype="f32"):
    d = _dtype_to_mlir(dtype) if not isinstance(dtype, str) else dtype
    return LayoutCompiler(layout, shape, d)


class LayoutCompiler:
    """Compiles a layout into a JIT-executable module."""

    def __init__(self, layout, shape, dtype="f32"):
        self._layout = layout
        self._shape = tuple(int(s) for s in shape)
        self._dtype = _dtype_to_mlir(dtype) if not isinstance(dtype, str) else dtype
        self._ctx = None
        self._engine = None
        self._mlir_text = None
        self._perm_table = None

    @property
    def mlir_text(self):
        if self._mlir_text is None:
            _, module = IRBuilder(self._layout, self._shape, self._dtype).build_module()
            self._mlir_text = str(module)
        return self._mlir_text

    def compile(self):
        if self._engine is None:
            self._ctx, module = IRBuilder(self._layout, self._shape, self._dtype).build_module()
            if self._mlir_text is None:
                self._mlir_text = str(module)
            with self._ctx:
                if _LEGO_DEBUG:
                    print("=== MLIR input (LEGO dialect) ===", file=sys.stderr)
                    print(module, file=sys.stderr)
                    print(file=sys.stderr)
                    module_copy = Module.parse(str(module))
                    pm_pre = PassManager.parse("builtin.module(canonicalize,cse)")
                    pm_pre.run(module_copy.operation)
                    print("=== MLIR after canonicalize + CSE ===", file=sys.stderr)
                    print(module_copy, file=sys.stderr)
                    print(file=sys.stderr)
                pm = PassManager.parse("builtin.module(lego-to-llvm)")
                pm.run(module.operation)
                if _LEGO_DEBUG:
                    print("=== MLIR after lego-to-llvm ===", file=sys.stderr)
                    print(module, file=sys.stderr)
                    print(file=sys.stderr)
                self._engine = ExecutionEngine(module, opt_level=2)
        return self._engine

    def _invoke(self, func_name, arr):
        engine = self.compile()
        total = 1
        for s in self._shape:
            total *= s
        src = np.ascontiguousarray(arr, dtype=arr.dtype).ravel()
        dst = np.empty_like(src)
        src_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(src)))
        dst_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(dst)))
        engine.invoke(func_name, src_ptr, dst_ptr, (ctypes.c_int64 * 1)(total))
        return dst.reshape(arr.shape)

    def transform_numpy(self, arr):
        return self._invoke("transform", arr)

    def inverse_transform_numpy(self, arr):
        return self._invoke("inverse_transform", arr)

    def get_permutation_table(self):
        """Compute forward and inverse permutation as int64 arrays."""
        if self._perm_table is not None:
            return self._perm_table

        total = 1
        for s in self._shape:
            total *= s

        i64_compiler = LayoutCompiler(self._layout, self._shape, "i64")
        identity = np.arange(total, dtype=np.int64)
        fwd = i64_compiler.transform_numpy(identity).ravel()
        inv = i64_compiler.inverse_transform_numpy(identity).ravel()

        self._perm_table = (fwd, inv)
        return self._perm_table
