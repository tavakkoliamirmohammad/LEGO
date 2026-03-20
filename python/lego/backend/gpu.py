"""
LEGO GPU Codegen Helpers

GPU kernel building utilities extracted from compiler.py:
- MemorySpace: GPU memory space enum
- MLIRPrinter: MLIR code generation for GPU kernels
- MLIRTensor: memref-backed tensor with LEGO layout
- mlir_apply, mlir_load, mlir_store, mlir_loop: pure-MLIR layout ops
"""

import functools
import sys
from enum import Enum

from mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IntegerAttr, Type,
)
from mlir import ir
from mlir.dialects import func as func_dialect
from mlir.dialects import arith as arith_dialect
from mlir.dialects import scf as scf_dialect
from mlir.dialects import memref as memref_dialect
from lego.core import product
from lego.backend._ops import (
    _LEGO_DEBUG,
    _index_const,
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
                    if _LEGO_DEBUG:
                        print("=== MLIR GPU module ===", file=sys.stderr)
                        print(module, file=sys.stderr)
                        if schedule:
                            print("=== MLIR schedule ===", file=sys.stderr)
                            print(Module.parse(schedule), file=sys.stderr)
                        print(file=sys.stderr)
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
