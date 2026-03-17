"""
LEGO Layout Compiler & MLIR Codegen

- IRBuilder: constructs LEGO dialect IR from layout objects
- LayoutCompiler: JIT-compiles via lego-to-llvm pipeline
- MLIRPrinter / MLIRTensor: GPU kernel codegen helpers
- mlir_apply, mlir_load, mlir_store, mlir_loop: pure-MLIR layout ops
"""

import ctypes
import functools
import numpy as np
from enum import Enum

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, MemRefType, FunctionType, IntegerAttr, StringAttr,
    IntegerType, F32Type, F64Type, UnitAttr, Type,
)
from mlir import ir
from mlir.dialects import func as func_dialect
from mlir.dialects import arith as arith_dialect
from mlir.dialects import scf as scf_dialect
from mlir.dialects import memref as memref_dialect
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from lego.backend.dialects.lego_dialect import register as register_lego
from lego.core import (
    LayoutBlock, Row, Col, RegP, OrderBy, GroupBy, TileByLayout, GenP, product,
)
from lego.backend._ops import (
    _index_const, _emit_reg_p, _emit_row, _emit_col, _emit_order_by,
    _emit_group_by, _emit_tile_by, _emit_gen_p,
    _emit_apply, _emit_apply_inverse,
    _emit_cast_view, _emit_load, _emit_store,
)

try:
    from mlir.dialects import gpu
except ImportError:
    gpu = None
try:
    import mlir.extras.types as T
except ImportError:
    T = None


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


def _emit_layout(layout, dim_vals):
    """Emit LEGO dialect ops for a core layout object. Returns layout SSA value."""
    if isinstance(layout, Row):
        return _emit_row(dim_vals)
    if isinstance(layout, Col):
        return _emit_col(dim_vals)
    if isinstance(layout, RegP):
        return _emit_reg_p(layout._perm_vector, dim_vals)
    if isinstance(layout, OrderBy):
        offset, perm_vals = 0, []
        for perm in layout.perms:
            count = len(perm.dims() if callable(getattr(perm, 'dims', None)) else perm._dims)
            perm_vals.append(_emit_layout(perm, dim_vals[offset:offset + count]))
            offset += count
        return _emit_order_by(perm_vals)
    if isinstance(layout, TileByLayout):
        all_perm_vals = []
        for orderby in layout._input_chain:
            for p in orderby.perms:
                all_perm_vals.append(_emit_layout(p, [_index_const(int(d)) for d in p.dims()]))
        input_val = _emit_order_by(all_perm_vals)
        tile_dim_vals = [_index_const(int(d)) for g in layout._tile_groups for d in g]
        return _emit_tile_by(input_val, tile_dim_vals, layout.tile_shape)
    if isinstance(layout, GroupBy):
        obj_vals = []
        for obj in layout.objects:
            obj_dims = [_index_const(int(d)) for d in obj.dims()]
            obj_vals.append(_emit_layout(obj, obj_dims))
        return _emit_group_by(dim_vals, obj_vals)
    if isinstance(layout, GenP):
        return _emit_gen_p(dim_vals, len(layout._dims),
                           layout.f_apply, layout.f_inv)
    raise TypeError(f"Unsupported: {type(layout).__name__}")


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
            layout_dims = _get_layout_dims(self._layout)
            dim_vals = [_index_const(s) for s in layout_dims]
            rank = len(layout_dims)

            identity_perm = list(range(rank))
            identity = _emit_group_by(dim_vals, [_emit_order_by([_emit_reg_p(identity_perm, dim_vals)])])
            layout_val = _emit_layout(self._layout, dim_vals)

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
                pm = PassManager.parse("builtin.module(lego-to-llvm)")
                pm.run(module.operation)
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


# ============================================================================
# GPU codegen helpers (pure MLIR, no SymPy)
# ============================================================================

class MemorySpace(Enum):
    HOST_MEMORY = 0
    GLOBAL_MEMORY = 1
    SHARED_MEMORY = 3
    PRIVATE_MEMORY = 5


class MLIRPrinter:
    """MLIR code generation helper for GPU kernels."""

    def __init__(self, ctx=Context()):
        self.ctx = ctx
        self.ctx.allow_unregistered_dialects = True
        try:
            from lego.backend.dialects.lego_dialect import register as _reg
            _reg(self.ctx)
        except Exception:
            pass

    def generate_mlir(self, schedule=None):
        def decorator_body(body):
            @functools.wraps(body)
            def wrapper():
                with self.ctx, Location.unknown():
                    module = Module.create()
                    with InsertionPoint(module.body):
                        @func_dialect.FuncOp.from_py_func()
                        def main():
                            return body()
                    if schedule:
                        print(Module.parse(schedule))
                    print(module)
            return wrapper()
        return decorator_body

    @staticmethod
    def insert_barrier():
        gpu.barrier()

    @staticmethod
    def get_token_type():
        return ir.Type.parse("!gpu.async.token")

    @staticmethod
    def generate_gpu_kernel(ins, outs, gridSize, blockSize,
                            workgroup_memory=[], private_memory=[]):
        def decorator_body(body):
            @functools.wraps(body)
            def wrapper():
                token = gpu.wait([])
                for t in set(ins + outs):
                    token = t.gpu_allocate(token)
                    t.host_allocate()
                for t in ins:
                    t.fill_host()
                    token = t.copy_to_device(token)
                gpu.wait([token])

                launch_op = gpu.LaunchOp(
                    list(map(arith_dialect.ConstantOp.create_index, gridSize)),
                    list(map(arith_dialect.ConstantOp.create_index, blockSize)),
                    async_dependencies=[])
                launch_op.attributes["workgroup_attributions"] = IntegerAttr.get(
                    T.i64(), len(workgroup_memory))

                block = launch_op.body.blocks[0]
                for w in workgroup_memory:
                    block.add_argument(w.get_memref_type_address_space(3), Location.unknown())
                for p in private_memory:
                    block.add_argument(p.get_memref_type_address_space(5), Location.unknown())

                with InsertionPoint(block):
                    for t in set(ins + outs):
                        t.set_memory_space(MemorySpace.GLOBAL_MEMORY)
                        memref_dialect.assume_alignment(t.gpu_alloc_ref, 128)
                    for idx in range(len(workgroup_memory)):
                        workgroup_memory[idx].shared_memory_ref = block.arguments[12 + idx]
                        workgroup_memory[idx].set_memory_space(MemorySpace.SHARED_MEMORY)
                    for p in private_memory:
                        p.set_memory_space(MemorySpace.PRIVATE_MEMORY)
                    body(block.arguments)
                    gpu.terminator()

                for t in set(ins + outs):
                    token = t.dealloc_gpu(token)
            return wrapper()
        return decorator_body


class MLIRTensor:
    """Tensor backed by a memref with a LEGO layout."""

    def __init__(self, layout, dtype="", is_dim_shape=False):
        self.layout = layout
        self.alloc_ref = None
        self.gpu_alloc_ref = None
        self.shared_memory_ref = None
        self.private_memory_ref = None
        self.data_type = None
        self.is_dim_shape = is_dim_shape
        self.dimension = layout.d
        self.memory_space = MemorySpace.HOST_MEMORY
        if dtype == "f32":
            self.data_type = T.f32()
        if dtype == "f16":
            self.data_type = T.f16()

    def get_memref_type(self):
        return self.get_memref_type_address_space(self.memory_space)

    def get_flattend_shape(self):
        return product(self.layout.dims())

    def get_memref_type_address_space(self, address_space):
        return T.memref(self.get_flattend_shape(), self.data_type, memory_space=address_space)

    def host_allocate(self):
        self.alloc_ref = memref_dialect.alloc(self.get_memref_type_address_space(0), [], [])
        return self

    def dealloc_gpu(self, *tokens):
        if tokens is None:
            gpu.dealloc(self.gpu_alloc_ref)
            return None
        return gpu.dealloc(Type.parse("!gpu.async.token"), list(tokens), self.gpu_alloc_ref)

    def set_memory_space(self, memory_space):
        self.memory_space = memory_space
        return self

    def gpu_allocate(self, *tokens):
        if tokens is None:
            self.gpu_alloc_ref = gpu.alloc(self.get_memref_type(), [], [], [], [])
            return None
        token_ty = Type.parse("!gpu.async.token")
        tmp = gpu.alloc(self.get_memref_type_address_space(0), token_ty, list(tokens), [], [])
        self.gpu_alloc_ref = tmp[0]
        return tmp[1]

    def fill_host(self):
        for_op = scf_dialect.ForOp(
            _index_const(0), _index_const(int(self.get_flattend_shape())), _index_const(1))
        with InsertionPoint(for_op.body):
            iv = for_op.induction_variable
            f_i = arith_dialect.sitofp(self.data_type, arith_dialect.index_cast(T.i32(), iv))
            self.store_physical_1d([iv], f_i)
            scf_dialect.YieldOp([])
        return self

    def store_physical_1d(self, coords, value):
        return memref_dialect.store(value, self.get_memory_ref_address_space(), coords)

    def copy_to_device(self, token):
        token_ty = Type.parse("!gpu.async.token")
        if token is None:
            gpu.memcpy(None, [], self.gpu_alloc_ref, self.alloc_ref)
            return None
        return gpu.memcpy(token_ty, [token], self.gpu_alloc_ref, self.alloc_ref)

    def get_memory_ref_address_space(self):
        if self.memory_space == MemorySpace.SHARED_MEMORY:
            return self.shared_memory_ref
        if self.memory_space == MemorySpace.PRIVATE_MEMORY:
            return self.private_memory_ref
        if self.memory_space == MemorySpace.GLOBAL_MEMORY:
            return self.gpu_alloc_ref
        return self.alloc_ref


printer = MLIRPrinter()


# ============================================================================
# Pure-MLIR layout helpers (no SymPy)
# ============================================================================

def mlir_layout(layout):
    """Emit a Python layout object as MLIR LEGO dialect ops."""
    from .symbolic import emit_layout_from_python
    return emit_layout_from_python(layout, {})


def mlir_apply(layout, indices):
    """Forward-apply a layout to MLIR index values."""
    return _emit_apply(mlir_layout(layout), indices)


def mlir_apply_inverse(layout, flat_idx):
    """Inverse-apply a layout to an MLIR flat index."""
    dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
    return _emit_apply_inverse(mlir_layout(layout), flat_idx, len(dims))


def mlir_cast_view(tensor):
    """Create a lego.view from an MLIRTensor's memref and layout."""
    return _emit_cast_view(mlir_layout(tensor.layout),
                           tensor.get_memory_ref_address_space(),
                           tensor.data_type)


def mlir_load(tensor, indices):
    """Load from an MLIRTensor using lego.cast_view + lego.load."""
    return _emit_load(mlir_cast_view(tensor), tensor.data_type, indices)


def mlir_store(value, tensor, indices):
    """Store to an MLIRTensor using lego.cast_view + lego.store."""
    _emit_store(value, mlir_cast_view(tensor), indices)


def mlir_loop(layout, body_fn):
    """Generate an scf.for loop with LEGO apply_inverse (no SymPy)."""
    dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
    total = 1
    for d in dims:
        total *= int(d)
    for_op = scf_dialect.ForOp(_index_const(0), _index_const(total), _index_const(1))
    with InsertionPoint(for_op.body):
        idx = for_op.induction_variable
        body_fn(mlir_apply_inverse(layout, idx), idx)
        scf_dialect.YieldOp([])
