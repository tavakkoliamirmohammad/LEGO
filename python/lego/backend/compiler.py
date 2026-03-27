"""
LEGO Layout Compiler

- IRBuilder: constructs LEGO dialect IR from layout objects
- LayoutCompiler: JIT-compiles via lego-to-llvm pipeline

Supports both concrete and symbolic layout dimensions.  When a layout
contains SymPy symbols, those symbols become function parameters in the
generated MLIR and are bound to concrete values at invocation time.
"""

import ctypes
import hashlib
import sys
import threading
import numpy as np
import sympy as sp

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, MemRefType, FunctionType, IntegerAttr, StringAttr,
    IntegerType, F32Type, F64Type, F16Type, BF16Type, UnitAttr,
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
# Global caches (thread-safe)
# ============================================================================

_CACHE_LOCK = threading.Lock()
_COMPILER_CACHE: dict = {}   # SHA-256 of MLIR text -> (ctx, ExecutionEngine)
_PERM_TABLE_CACHE: dict = {} # SHA-256 of i64 MLIR text -> (fwd, inv)


def _mlir_cache_key(mlir_text: str) -> str:
    """Compute SHA-256 hash of MLIR text for cache keying."""
    return hashlib.sha256(mlir_text.encode('utf-8')).hexdigest()


# ============================================================================
# Dtype mapping
# ============================================================================

def _dtype_to_mlir(dtype):
    """Map numpy/torch dtype to MLIR element type string."""
    dtype_map = {
        np.float32: "f32", np.float64: "f64",
        np.int32: "i32", np.int64: "i64", np.int16: "i16", np.int8: "i8",
        np.bool_: "i1", np.uint8: "ui8",
    }
    try:
        import torch
        dtype_map.update({
            torch.float32: "f32", torch.float64: "f64",
            torch.float16: "f16", torch.bfloat16: "bf16",
            torch.int32: "i32", torch.int64: "i64",
            torch.int16: "i16", torch.int8: "i8",
            torch.bool: "i1", torch.uint8: "ui8",
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
        "f16": lambda: F16Type.get(ctx),
        "bf16": lambda: BF16Type.get(ctx),
        "i32": lambda: IntegerType.get_signless(32, ctx),
        "i64": lambda: IntegerType.get_signless(64, ctx),
        "i16": lambda: IntegerType.get_signless(16, ctx),
        "i8": lambda: IntegerType.get_signless(8, ctx),
        "i1": lambda: IntegerType.get_signless(1, ctx),
        "ui8": lambda: IntegerType.get_unsigned(8, ctx),
    }
    return type_map.get(dtype_str, type_map["f32"])()


# ============================================================================
# IR Builder — emits LEGO dialect IR from core layout objects
# ============================================================================

def _get_layout_dims(layout):
    """Get dims from a layout object (may be int or SymPy expressions)."""
    dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
    return tuple(dims)


def _collect_symbolic_dims(layout):
    """Collect all free SymPy symbols from a layout's dimensions."""
    from lego.backend.symbolic import _collect_free_symbols
    return sorted(_collect_free_symbols(layout), key=lambda s: s.name)


def _has_symbolic_dims(layout):
    """Check if a layout contains any SymPy symbols."""
    return len(_collect_symbolic_dims(layout)) > 0


class IRBuilder:
    """Builds a complete MLIR module with @transform / @inverse_transform.

    When the layout has symbolic dimensions, they become additional index
    parameters appended after (src, dst, n).  The caller must provide
    concrete values for these symbols at invocation time.
    """

    def __init__(self, layout, shape, dtype_str="f32"):
        self._layout = layout
        self._shape = shape
        self._dtype_str = dtype_str
        self._symbolic = _has_symbolic_dims(layout)
        self._sym_list = _collect_symbolic_dims(layout) if self._symbolic else []
        self._total = 1
        for s in shape:
            self._total *= int(s)

    @property
    def sym_list(self):
        """Ordered list of SymPy symbols that become function parameters."""
        return self._sym_list

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
        if not forward:
            from lego.frontends.python_mlir import _check_layout_invertible
            _check_layout_invertible(self._layout)
        # Base args: src, dst, n.  Symbolic dims appended as extra index args.
        param_types = [memref_ty, memref_ty, idx_ty] + [idx_ty] * len(self._sym_list)
        func_ty = FunctionType.get(param_types, [])
        with InsertionPoint(module.body):
            f = func_dialect.FuncOp(name, func_ty)
            f.sym_visibility = StringAttr.get("public")
            f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        entry = f.add_entry_block()
        src, dst, n = entry.arguments[0], entry.arguments[1], entry.arguments[2]

        # Build sym_to_val mapping for symbolic dims
        sym_to_val = {}
        for i, sym in enumerate(self._sym_list):
            sym_to_val[sym] = entry.arguments[3 + i]

        with InsertionPoint(entry):
            from lego.backend.symbolic import emit_layout_from_python, _resolve_dim

            layout_dims = _get_layout_dims(self._layout)
            dim_vals = [_resolve_dim(d, sym_to_val) for d in layout_dims]
            rank = len(layout_dims)

            identity = _emit_group_by(dim_vals, [_emit_order_by([_emit_row(dim_vals)])])
            layout_val = emit_layout_from_python(self._layout, sym_to_val)

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

    def build_composed_module(self, layout_b, shape):
        """Build a single MLIR module that composes two layouts.

        The composed transform is: apply(B, apply_inverse(identity, apply(A, indices)))
        where A = self._layout, B = layout_b.  MLIR simplification passes can
        cancel redundant apply/apply_inverse pairs.
        """
        ctx = Context()
        register_lego(ctx)
        with ctx, Location.unknown():
            module = Module.create()
            idx_ty = IndexType.get()
            total = 1
            for s in shape:
                total *= int(s)
            elem_ty = _get_mlir_element_type(ctx, self._dtype_str)
            memref_ty = MemRefType.get([total], elem_ty)

            # Collect symbolic dims from both layouts
            sym_list_a = _collect_symbolic_dims(self._layout) if _has_symbolic_dims(self._layout) else []
            sym_list_b = _collect_symbolic_dims(layout_b) if _has_symbolic_dims(layout_b) else []
            all_syms = list(dict.fromkeys(sym_list_a + sym_list_b))

            param_types = [memref_ty, memref_ty, idx_ty] + [idx_ty] * len(all_syms)
            func_ty = FunctionType.get(param_types, [])
            with InsertionPoint(module.body):
                f = func_dialect.FuncOp("composed_transform", func_ty)
                f.sym_visibility = StringAttr.get("public")
                f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

            entry = f.add_entry_block()
            src, dst, n = entry.arguments[0], entry.arguments[1], entry.arguments[2]

            sym_to_val = {}
            for i, sym in enumerate(all_syms):
                sym_to_val[sym] = entry.arguments[3 + i]

            with InsertionPoint(entry):
                from lego.backend.symbolic import emit_layout_from_python, _resolve_dim

                layout_dims = _get_layout_dims(self._layout)
                dim_vals = [_resolve_dim(d, sym_to_val) for d in layout_dims]
                rank = len(layout_dims)

                identity = _emit_group_by(dim_vals, [_emit_order_by([_emit_row(dim_vals)])])
                layout_a_val = emit_layout_from_python(self._layout, sym_to_val)
                layout_b_val = emit_layout_from_python(layout_b, sym_to_val)

                loop = scf_dialect.ForOp(_index_const(0), n, _index_const(1))
                with InsertionPoint(loop.body):
                    iv = loop.induction_variable
                    val = memref_dialect.LoadOp(src, [iv])
                    # Compose: apply A inverse, then apply B
                    indices = _emit_apply_inverse(layout_a_val, iv, rank)
                    logical_idx = _emit_apply(identity, indices)
                    composed_indices = _emit_apply_inverse(layout_b_val, logical_idx, rank)
                    memref_dialect.StoreOp(val.result, dst, [_emit_apply(identity, composed_indices)])
                    scf_dialect.YieldOp([])
                func_dialect.ReturnOp([])
        return ctx, module


# ============================================================================
# JIT Compiler
# ============================================================================

def get_compiler(layout, shape, dtype="f32", dim_bindings=None):
    d = _dtype_to_mlir(dtype) if not isinstance(dtype, str) else dtype
    return LayoutCompiler(layout, shape, d, dim_bindings=dim_bindings)


class LayoutCompiler:
    """Compiles a layout into a JIT-executable module.

    Supports layouts with symbolic dimensions.  Concrete dimension values
    are resolved from the provided ``shape`` via ``dim_bindings``, or
    inferred automatically when the shape matches the layout's group dims.
    """

    def __init__(self, layout, shape, dtype="f32", dim_bindings=None):
        """
        Parameters
        ----------
        layout : GroupBy or TileByLayout
        shape : tuple of int
            Concrete tensor shape.
        dtype : str or numpy/torch dtype
        dim_bindings : dict, optional
            Maps SymPy symbols to concrete int values.  If not provided,
            symbols are resolved by matching the layout's ``_dims`` tuple
            against ``shape`` positionally.
        """
        self._layout = layout
        self._shape = tuple(int(s) for s in shape)
        self._dtype = _dtype_to_mlir(dtype) if not isinstance(dtype, str) else dtype
        self._ctx = None
        self._engine = None
        self._mlir_text = None
        self._perm_table = None
        self._ir_builder = None
        self._dim_bindings = dim_bindings or {}

    def _get_ir_builder(self):
        if self._ir_builder is None:
            self._ir_builder = IRBuilder(self._layout, self._shape, self._dtype)
        return self._ir_builder

    def _resolve_sym_values(self):
        """Resolve concrete values for all symbolic parameters.

        Uses multi-pass resolution: in each pass, substitutes known
        bindings into unsolved equations before solving.  This handles
        compound expressions like ``R*T = 120`` where R is already known.
        """
        builder = self._get_ir_builder()
        if not builder.sym_list:
            return []

        # Use explicit bindings if provided
        bindings = dict(self._dim_bindings)

        # Auto-resolve: match layout._dims against shape positionally
        if not bindings:
            layout_dims = _get_layout_dims(self._layout)
            if len(layout_dims) == len(self._shape):
                # Build equation list: (dim_expr, concrete_value)
                equations = []
                for dim_expr, concrete_val in zip(layout_dims, self._shape):
                    if isinstance(dim_expr, sp.Symbol):
                        bindings[dim_expr] = concrete_val
                    elif isinstance(dim_expr, sp.Expr):
                        equations.append((dim_expr, concrete_val))

                # Multi-pass: substitute known bindings and solve iteratively
                max_passes = len(equations) + 1
                for _ in range(max_passes):
                    progress = False
                    remaining = []
                    for dim_expr, concrete_val in equations:
                        # Substitute known bindings into the expression
                        substituted = dim_expr.subs(bindings)
                        unknowns = substituted.free_symbols - set(bindings.keys())
                        if not unknowns:
                            # Fully resolved — verify consistency
                            continue
                        if len(unknowns) == 1:
                            sym = unknowns.pop()
                            try:
                                solutions = sp.solve(substituted - concrete_val, sym)
                                if len(solutions) == 1:
                                    val = solutions[0]
                                    if val.is_Integer and int(val) > 0:
                                        bindings[sym] = int(val)
                                        progress = True
                                        continue
                            except (NotImplementedError, ValueError):
                                pass
                        remaining.append((dim_expr, concrete_val))
                    equations = remaining
                    if not progress:
                        break

        values = []
        for sym in builder.sym_list:
            if sym not in bindings:
                known = {str(k): v for k, v in bindings.items()}
                raise ValueError(
                    f"Cannot resolve symbolic dimension '{sym}'. "
                    f"Known bindings: {known}. "
                    f"Provide dim_bindings={{'{sym}': <value>}} to LayoutCompiler, "
                    f"or ensure the layout's dims match the tensor shape positionally."
                )
            values.append(int(bindings[sym]))
        return values

    @property
    def mlir_text(self):
        if self._mlir_text is None:
            _, module = self._get_ir_builder().build_module()
            self._mlir_text = str(module)
        return self._mlir_text

    def compile(self):
        if self._engine is None:
            # Check global cache first
            cache_key = _mlir_cache_key(self.mlir_text)
            with _CACHE_LOCK:
                cached = _COMPILER_CACHE.get(cache_key)
            if cached is not None:
                self._ctx, self._engine = cached
                return self._engine

            builder = self._get_ir_builder()
            self._ctx, module = builder.build_module()
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

            with _CACHE_LOCK:
                _COMPILER_CACHE[cache_key] = (self._ctx, self._engine)
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

        # Build invocation args: src, dst, n, [sym_val_0, sym_val_1, ...]
        sym_values = self._resolve_sym_values()
        n_arr = (ctypes.c_int64 * 1)(total)
        sym_args = [(ctypes.c_int64 * 1)(v) for v in sym_values]
        engine.invoke(func_name, src_ptr, dst_ptr, n_arr, *sym_args)
        return dst.reshape(arr.shape)

    def transform_numpy(self, arr):
        return self._invoke("transform", arr)

    def inverse_transform_numpy(self, arr):
        return self._invoke("inverse_transform", arr)

    def get_permutation_table(self):
        """Compute forward and inverse permutation as int64 arrays.

        Returns (fwd, inv) where both are 1-D int64 arrays of length numel.

        Semantics (gather-based):
          fwd: logical->physical gather table.  output[i] = input[fwd[i]]
          inv: physical->logical gather table.  output[i] = input[inv[i]]

        Invariant: inv[fwd[i]] == i  and  fwd[inv[i]] == i  for all i.
        """
        if self._perm_table is not None:
            return self._perm_table

        # Check global perm table cache
        i64_compiler = LayoutCompiler(self._layout, self._shape, "i64")
        perm_cache_key = _mlir_cache_key(i64_compiler.mlir_text)
        with _CACHE_LOCK:
            cached = _PERM_TABLE_CACHE.get(perm_cache_key)
        if cached is not None:
            self._perm_table = cached
            return self._perm_table

        total = 1
        for s in self._shape:
            total *= s

        identity = np.arange(total, dtype=np.int64)
        fwd = i64_compiler.transform_numpy(identity).ravel()
        inv = i64_compiler.inverse_transform_numpy(identity).ravel()

        self._perm_table = (fwd, inv)
        with _CACHE_LOCK:
            _PERM_TABLE_CACHE[perm_cache_key] = self._perm_table
        return self._perm_table
