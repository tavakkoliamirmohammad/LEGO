"""
LEGO Layout Compiler

Builds LEGO MLIR IR programmatically using the MLIR Python bindings, then
JIT-compiles via the lego-to-llvm pipeline for efficient tensor transforms.

Uses mlir.execution_engine.ExecutionEngine for JIT (no custom C++ wrapper).
"""

import ctypes
import numpy as np

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, MemRefType, FunctionType, IntegerAttr, StringAttr,
    IntegerType, F32Type, F64Type, UnitAttr,
)
from mlir.dialects import func as func_dialect
from mlir.dialects import arith as arith_dialect
from mlir.dialects import scf as scf_dialect
from mlir.dialects import memref as memref_dialect
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from lego.dialects.lego_dialect import register as register_lego


# ============================================================================
# Lightweight layout descriptors (no SymPy dependency)
# ============================================================================

class RegPDesc:
    """Regular permutation: dims + permutation vector."""

    def __init__(self, dims, perm):
        self.dims = tuple(int(d) for d in dims)
        self.perm = [int(p) for p in perm]


class RowDesc:
    """Mirrors lego.row [dims] : !lego.layout"""

    def __init__(self, dims):
        self.dims = tuple(int(d) for d in dims)


class ColDesc:
    """Mirrors lego.col [dims] : !lego.layout"""

    def __init__(self, dims):
        self.dims = tuple(int(d) for d in dims)


class OrderByDesc:
    """Sequence of permutation blocks."""

    def __init__(self, perms):
        self.perms = list(perms)

    @property
    def dims(self):
        result = []
        for p in self.perms:
            result.extend(p.dims)
        return tuple(result)

    def tile_by(self, *tile_groups):
        """Chain into a TileByDesc."""
        return TileByDesc(self, list(tile_groups))


class GroupByDesc:
    """Grouped layout: outer dims + inner layout objects."""

    def __init__(self, dims, objects):
        self.dims = tuple(int(d) for d in dims)
        self.objects = list(objects)


class TileByDesc:
    """Mirrors lego.tile_by %input tile_dims [[g0], [g1], ...] : !lego.layout"""

    def __init__(self, input_layout, tile_groups):
        self.input = input_layout
        self.tile_groups = [tuple(int(x) for x in g) for g in tile_groups]

    @property
    def d(self):
        return len(self.tile_groups[0])

    @property
    def q(self):
        return len(self.tile_groups)

    @property
    def dims(self):
        """Flattened tile dims — matches getLayoutInputDims(TileByOp)."""
        return tuple(x for g in self.tile_groups for x in g)

    @property
    def tile_shape(self):
        """[d, d, ..., d] (q times) — the tile_shape I64ArrayAttr."""
        return [self.d] * self.q


class GenPDesc:
    """Mirrors lego.gen_p [dims] apply { ... } inv { ... } : !lego.layout

    apply_builder: callable(block_args: list[Value]) -> Value  (single flat index)
    inv_builder:   callable(block_arg: Value) -> list[Value]   (N-D indices)
    """

    def __init__(self, dims, apply_builder, inv_builder):
        self.dims = tuple(int(d) for d in dims)
        self.apply_builder = apply_builder
        self.inv_builder = inv_builder


_LAYOUT_DESC_TYPES = (
    RegPDesc, RowDesc, ColDesc, OrderByDesc, GroupByDesc, TileByDesc, GenPDesc,
)


def _layout_from_sympy(block):
    """Convert a SymPy-based layout block to a lightweight descriptor."""
    from .lego import RegP, OrderBy, GroupBy

    if isinstance(block, RegP):
        return RegPDesc(block.dims(), block._raw_perm)
    elif isinstance(block, OrderBy):
        return OrderByDesc([_layout_from_sympy(p) for p in block.perms])
    elif isinstance(block, GroupBy):
        return GroupByDesc(
            block.dims(),
            [_layout_from_sympy(o) for o in block.objects],
        )
    else:
        raise TypeError(f"Cannot convert {type(block).__name__} to descriptor")


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
    functions that use LEGO dialect ops (reg_p, row, col, order_by, group_by,
    tile_by, gen_p, apply, apply_inverse) and standard dialects.
    """

    def __init__(self, layout_desc, shape, dtype_str="f32"):
        if not isinstance(layout_desc, _LAYOUT_DESC_TYPES):
            layout_desc = _layout_from_sympy(layout_desc)
        self._layout = layout_desc
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

    def _make_identity_layout(self, layout_dims, dim_vals, idx_ty):
        """Auto-generate row-major identity layout matching the descriptor's dims."""
        identity_perm = list(range(len(layout_dims)))
        identity_reg_p = _emit_reg_p(identity_perm, dim_vals)
        identity_layout = _emit_order_by([identity_reg_p])
        return _emit_group_by(dim_vals, [identity_layout])

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
            # Use the layout's own dims for linearization/delinearization
            layout_dims = self._layout.dims

            dim_vals = []
            for s in layout_dims:
                c = arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(s)))
                dim_vals.append(c.result)

            # Build the identity (row-major) layout for delinearization
            identity_group = self._make_identity_layout(layout_dims, dim_vals, idx_ty)

            # Build the custom layout
            layout_val = self._emit_layout(self._layout, dim_vals, idx_ty)

            # Loop: scf.for %i = 0 to %n step 1
            c0 = arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 0))
            c1 = arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))

            loop = scf_dialect.ForOp(c0.result, n, c1.result)
            with InsertionPoint(loop.body):
                iv = loop.induction_variable

                rank = len(layout_dims)
                if forward:
                    # Forward: row-major → custom layout
                    # Delinearize with identity, relinearize with custom
                    val = memref_dialect.LoadOp(src, [iv])
                    indices = _emit_apply_inverse(identity_group, iv, rank)
                    new_idx = _emit_apply(layout_val, indices)
                    memref_dialect.StoreOp(val.result, dst, [new_idx])
                else:
                    # Inverse: custom layout → row-major
                    # Delinearize with custom, relinearize with identity
                    val = memref_dialect.LoadOp(src, [iv])
                    indices = _emit_apply_inverse(layout_val, iv, rank)
                    new_idx = _emit_apply(identity_group, indices)
                    memref_dialect.StoreOp(val.result, dst, [new_idx])

                scf_dialect.YieldOp([])

            func_dialect.ReturnOp([])

    def _emit_layout(self, block, dim_vals, idx_ty):
        """Emit LEGO dialect ops for a layout descriptor. Returns the layout SSA value."""
        if isinstance(block, RegPDesc):
            return _emit_reg_p(block.perm, dim_vals)
        elif isinstance(block, RowDesc):
            return _emit_row(dim_vals)
        elif isinstance(block, ColDesc):
            return _emit_col(dim_vals)
        elif isinstance(block, OrderByDesc):
            return self._emit_order_by(block, dim_vals, idx_ty)
        elif isinstance(block, GroupByDesc):
            return self._emit_group_by(block, dim_vals, idx_ty)
        elif isinstance(block, TileByDesc):
            return self._emit_tile_by(block, idx_ty)
        elif isinstance(block, GenPDesc):
            return _emit_gen_p(dim_vals, len(block.dims),
                               block.apply_builder, block.inv_builder, idx_ty)
        else:
            raise TypeError(f"Unsupported layout block type: {type(block).__name__}")

    def _emit_order_by(self, order_by, dim_vals, idx_ty):
        """Emit lego.order_by with its sub-permutation blocks."""
        offset = 0
        perm_vals = []
        for perm in order_by.perms:
            count = len(perm.dims)
            sub_dims = dim_vals[offset:offset + count]
            offset += count
            perm_val = self._emit_layout(perm, sub_dims, idx_ty)
            perm_vals.append(perm_val)

        return _emit_order_by(perm_vals)

    def _emit_group_by(self, group_by, dim_vals, idx_ty):
        """Emit lego.group_by with its object layouts."""
        obj_vals = []
        for obj in group_by.objects:
            obj_dim_vals = []
            for d in obj.dims:
                c = arith_dialect.ConstantOp(
                    idx_ty, IntegerAttr.get(idx_ty, int(d))
                )
                obj_dim_vals.append(c.result)
            obj_val = self._emit_layout(obj, obj_dim_vals, idx_ty)
            obj_vals.append(obj_val)

        return _emit_group_by(dim_vals, obj_vals)

    def _emit_tile_by(self, tile_by, idx_ty):
        """Emit lego.tile_by with its input layout and tile dimensions."""
        # Emit input layout (e.g. OrderByDesc) with its own dims
        input_dim_vals = [
            arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(d))).result
            for d in tile_by.input.dims
        ]
        input_val = self._emit_layout(tile_by.input, input_dim_vals, idx_ty)

        # Emit tile dim constants
        tile_dim_vals = [
            arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(d))).result
            for d in tile_by.dims
        ]
        return _emit_tile_by(input_val, tile_dim_vals, tile_by.tile_shape)


# ============================================================================
# Op emission helpers using auto-generated LEGO dialect Python bindings
# ============================================================================

def _lego_layout_type():
    """Get the !lego.layout type from the current context."""
    from mlir.ir import Type
    return Type.parse("!lego.layout")


def _emit_reg_p(perm, dim_vals):
    """Emit a lego.reg_p op. Returns the layout SSA value."""
    from lego.dialects.lego_dialect import RegPOp
    layout_ty = _lego_layout_type()
    return RegPOp(result=layout_ty, perm=perm, dims=dim_vals).result


def _emit_row(dim_vals):
    """Emit a lego.row op. Returns the layout SSA value."""
    from lego.dialects.lego_dialect import RowOp
    layout_ty = _lego_layout_type()
    return RowOp(result=layout_ty, dims=dim_vals).result


def _emit_col(dim_vals):
    """Emit a lego.col op. Returns the layout SSA value."""
    from lego.dialects.lego_dialect import ColOp
    layout_ty = _lego_layout_type()
    return ColOp(result=layout_ty, dims=dim_vals).result


def _emit_order_by(perm_vals):
    """Emit a lego.order_by op. Returns the layout SSA value."""
    from lego.dialects.lego_dialect import OrderByOp
    layout_ty = _lego_layout_type()
    return OrderByOp(result=layout_ty, perms=perm_vals).result


def _emit_group_by(dim_vals, obj_vals):
    """Emit a lego.group_by op. Returns the layout SSA value."""
    from lego.dialects.lego_dialect import GroupByOp
    layout_ty = _lego_layout_type()
    return GroupByOp(result=layout_ty, group_dims=dim_vals, objects=obj_vals).result


def _emit_tile_by(input_val, tile_dim_vals, tile_shape):
    """Emit a lego.tile_by op. Returns the layout SSA value."""
    from lego.dialects.lego_dialect import TileByOp
    layout_ty = _lego_layout_type()
    return TileByOp(result=layout_ty, input=input_val,
                    tile_dims=tile_dim_vals, tile_shape=tile_shape).result


def _emit_gen_p(dim_vals, rank, apply_builder, inv_builder, idx_ty):
    """Emit a lego.gen_p op with apply and inverse regions."""
    from lego.dialects.lego_dialect import GenPOp, YieldOp
    layout_ty = _lego_layout_type()
    gen_p_op = GenPOp(result=layout_ty, dims=dim_vals)

    # Apply region: block args = N indices → yield single flat index
    apply_block = gen_p_op.body.blocks.append(*([idx_ty] * rank))
    with InsertionPoint(apply_block):
        flat_result = apply_builder(list(apply_block.arguments))
        YieldOp(values=[flat_result])

    # Inverse region: block arg = single flat index → yield N indices
    inv_block = gen_p_op.inv_body.blocks.append(idx_ty)
    with InsertionPoint(inv_block):
        index_results = inv_builder(inv_block.arguments[0])
        YieldOp(values=list(index_results))

    return gen_p_op.result


def _emit_apply(layout_val, indices):
    """Emit a lego.apply op. Returns the flat index Value."""
    from lego.dialects.lego_dialect import ApplyOp
    idx_ty = IndexType.get()
    return ApplyOp(flat_index=idx_ty, layout=layout_val, indices=list(indices)).result


def _emit_apply_inverse(layout_val, flat_index, rank):
    """Emit a lego.apply_inverse op. Returns list of index Values."""
    from lego.dialects.lego_dialect import ApplyInverseOp
    idx_ty = IndexType.get()
    op = ApplyInverseOp(indices=[idx_ty] * rank, layout=layout_val,
                        flat_index=flat_index)
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
        if isinstance(layout, _LAYOUT_DESC_TYPES):
            self._layout = layout
        else:
            # Convert SymPy-based GroupBy to descriptor
            self._layout = _layout_from_sympy(layout)
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
