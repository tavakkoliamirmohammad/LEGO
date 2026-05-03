"""
Pure-Python GPU kernel DSL with AST transformation.

Transforms native Python syntax (arithmetic, for, if, while, indexing)
into MLIR GPU IR via the existing KernelBuilder/KernelContext infrastructure.

Usage::

    from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

    @gpu_kernel(grid=(N // 256,), block=(256,))
    def vecadd(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
        gid = block_id.x * block_dim.x + thread_id.x
        C[gid] = A[gid] + B[gid]

    # vecadd is a KernelBuilder — compile to any target:
    result = vecadd.compile(target='cuda')
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Tuple

from lego.mlir.ir import InsertionPoint, F32Type
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import scf as scf_dialect

from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer, _index_const, _TensorCoreHandle
from lego.backend._dsl_base import (
    _BaseCompiler,
    CT_INT, CT_FLOAT, INDEX, F32, I1, I32, CT_OBJ,
    _is_ct, _to_runtime, _promote,
)

# Public alias for TensorCore — used inside @gpu_kernel bodies
TensorCore = _TensorCoreHandle


# ============================================================================
# Type annotations for kernel parameters
# ============================================================================

class Buffer:
    """Global GPU buffer.

    Two forms::

        Buffer[N]          # Row(N) layout (default)
        Buffer[M, K]       # Row(M, K) layout (default)
        Buffer(layout, M, K)  # custom LEGO layout
    """
    def __new__(cls, layout, *dims, dtype=DType.f32):
        return _BufferType(dims=tuple(dims), shared=False, dtype=dtype, layout=layout)

    def __class_getitem__(cls, dims):
        if not isinstance(dims, tuple):
            dims = (dims,)
        return _BufferType(dims=dims, shared=False)


class Shared:
    """Shared (workgroup) memory buffer.

    Two forms::

        Shared[TILE]              # Row(TILE) layout
        Shared(layout, TILE, TILE)  # custom layout
    """
    def __new__(cls, layout, *dims, dtype=DType.f32):
        return _BufferType(dims=tuple(dims), shared=True, dtype=dtype, layout=layout)

    def __class_getitem__(cls, dims):
        if not isinstance(dims, tuple):
            dims = (dims,)
        return _BufferType(dims=dims, shared=True)


@dataclass
class _BufferType:
    dims: tuple
    shared: bool = False
    dtype: DType = DType.f32
    layout: object = None       # None → default Row(*dims)


# ============================================================================
# GPU-only value tags
# ============================================================================

MMA_FRAG = "mma_frag"   # MLIR !gpu.mma_matrix Value

# Shared tags CT_INT, CT_FLOAT, INDEX, F32, I1, I32, CT_OBJ imported above.
# _is_ct, _to_runtime, _promote imported above.


# ============================================================================
# Decorator
# ============================================================================

def gpu_kernel(grid, block):
    """Decorator that transforms a Python function into a :class:`KernelBuilder`."""
    def decorator(fn):
        return _build(fn, grid, block)
    return decorator


def _build(fn, grid, block):
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    func_def = tree.body[0]

    # Resolve outer-scope names (TILE, K, N, …)
    outer = fn.__globals__.copy()
    if fn.__code__.co_freevars and fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            outer[name] = cell.cell_contents

    # Parse buffer annotations
    buf_params = []
    for arg in func_def.args.args:
        ann = eval(                                      # noqa: S307
            compile(ast.Expression(arg.annotation), "<ann>", "eval"), outer)
        assert isinstance(ann, _BufferType), (
            f"Expected Buffer[…] or Shared[…] for '{arg.arg}'")
        dims = tuple(int(d) for d in ann.dims)
        buf_params.append((arg.arg, _BufferType(dims, ann.shared, ann.dtype, ann.layout)))

    buffers = [
        LayoutBuffer(bt.layout or Row(*bt.dims), shape=bt.dims, dtype=bt.dtype,
                     shared=bt.shared)
        for _, bt in buf_params
    ]

    def kernel_body(ctx):
        _Compiler(ctx, func_def, buf_params, outer).run()

    return KernelBuilder(
        buffers=buffers,
        kernel_body=kernel_body,
        name=fn.__name__,
        grid=grid,
        block=block,
    )


# ============================================================================
# AST → MLIR compiler
# ============================================================================

class _Compiler(_BaseCompiler):
    """Walk a Python function AST and emit MLIR via KernelContext.

    Inherits shared AST-dispatch, binop, if/while, compare, and utility
    methods from _BaseCompiler.  Only GPU-specific overrides are here:
    - __init__: no scalar_params (GPU buffers only).
    - _for: range() only (no tile_range sentinel).
    - _name: handles TensorCore compile-time objects.
    - _attribute: resolves block_id.x, thread_id.y, block_dim.z.
    - _load/_store: GPU buffer semantics.
    - _call/_method_call: GPU intrinsics (barrier, shuffle, reduce, MMA, …).
    """

    _GPU_DIMS = {"block_id", "thread_id", "block_dim"}

    def __init__(self, ctx, func_def, buf_params, outer):
        self.ctx = ctx
        self.func_def = func_def
        self.outer = outer
        self.env = {}                     # name → (value, tag)
        self.buf_map = {}                 # name → buffer index
        for i, (name, _) in enumerate(buf_params):
            self.buf_map[name] = i

    # -- for (range() only; no tile_range) ----------------------------

    def _for(self, node):
        var = node.target.id
        call = node.iter
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        assert call.func.id == "range"
        args = [self._expr(a) for a in call.args]
        if len(args) == 1:
            lb, ub, step = (_index_const(0), self._idx(args[0]), _index_const(1))
        elif len(args) == 2:
            lb, ub, step = (self._idx(args[0]), self._idx(args[1]), _index_const(1))
        else:
            lb, ub, step = (self._idx(args[0]), self._idx(args[1]), self._idx(args[2]))

        env_before = set(self.env)
        modified = self._modified_names(node.body) & env_before
        modified.discard(var)
        ia_names = sorted(modified)

        ia_vals, ia_tags = [], []
        for n in ia_names:
            v, t = self.env[n]
            if _is_ct(t):
                v, t = _to_runtime(self.ctx, v, t)
            ia_vals.append(v)
            ia_tags.append(t)

        loop = scf_dialect.ForOp(lb, ub, step, ia_vals or None)
        with InsertionPoint(loop.body):
            self.env[var] = (loop.induction_variable, INDEX)
            for i, n in enumerate(ia_names):
                self.env[n] = (loop.inner_iter_args[i], ia_tags[i])
            for s in node.body:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in ia_names])

        for i, n in enumerate(ia_names):
            self.env[n] = (loop.results[i], ia_tags[i])
        for n in set(self.env) - env_before:
            del self.env[n]

    # -- name resolution -----------------------------------------------

    def _name(self, name):
        if name in self.env:
            return self.env[name]
        if name in self.outer:
            v = self.outer[name]
            if isinstance(v, (int, bool)):
                return (int(v), CT_INT)
            if isinstance(v, float):
                return (v, CT_FLOAT)
            if isinstance(v, _TensorCoreHandle):
                return (v, CT_OBJ)
            raise TypeError(f"Unsupported type for '{name}': {type(v)}")
        raise NameError(f"Undefined: {name}")

    # -- attribute access (GPU dims) -----------------------------------

    def _attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self._GPU_DIMS:
            accessor = {"block_id": self.ctx.block_id,
                        "thread_id": self.ctx.thread_id,
                        "block_dim": self.ctx.block_dim}[node.value.id]
            return (getattr(accessor, node.attr), INDEX)
        raise NotImplementedError(f"Attribute {ast.dump(node)}")

    # -- buffer load / store -------------------------------------------

    def _load(self, node):
        idx = self.buf_map[node.value.id]
        return (self.ctx.load(idx, self._indices(node)), F32)

    def _store(self, node, val, tag):
        if _is_ct(tag):
            val, tag = _to_runtime(self.ctx, val, tag, F32)
        idx = self.buf_map[node.value.id]
        self.ctx.store(val, idx, self._indices(node))

    # -- function calls ------------------------------------------------

    def _call(self, node):
        if isinstance(node.func, ast.Attribute):
            return self._method_call(node)
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError(f"Call {ast.dump(node.func)}")
        name = node.func.id
        if name == "barrier":
            self.ctx.barrier()
            return (None, CT_INT)
        if name == "apply_inverse":
            layout = self._eval_ct(node.args[0])
            idx_val = self._idx(self._expr(node.args[1]))
            from lego.backend.compiler import _get_layout_dims
            rank = len(_get_layout_dims(layout))
            result = self.ctx.apply_inverse(layout, idx_val)
            return (list(result), "tuple")
        if name == "apply":
            layout = self._eval_ct(node.args[0])
            indices = [self._idx(self._expr(a)) for a in node.args[1:]]
            return (self.ctx.apply(layout, *indices), INDEX)
        if name == "set_layout":
            buf_name = node.args[0].id
            buf_idx = self.buf_map[buf_name]
            layout = self._eval_ct(node.args[1])
            self.ctx.set_layout(buf_idx, layout)
            return (None, CT_INT)
        if name == "lane_id":
            return (self.ctx.lane_id(), INDEX)
        if name == "warp_size":
            return (self.ctx.subgroup_size(), INDEX)
        if name in ("shuffle_down", "shuffle_up", "shuffle_xor", "shuffle_idx"):
            val, vtag = self._expr(node.args[0])
            arg1_v, arg1_t = self._expr(node.args[1])
            if _is_ct(arg1_t):
                arg1_v = int(arg1_v)
            fn = getattr(self.ctx, name)
            return (fn(val, arg1_v), F32)
        if name == "subgroup_reduce_add":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.subgroup_reduce_add(val), F32)
        if name == "subgroup_reduce_mul":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.subgroup_reduce_mul(val), F32)
        if name == "subgroup_reduce_max":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.subgroup_reduce_max(val), F32)
        if name == "subgroup_reduce_min":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.subgroup_reduce_min(val), F32)
        if name == "all_reduce_add":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.all_reduce_add(val), F32)
        if name == "all_reduce_mul":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.all_reduce_mul(val), F32)
        if name == "all_reduce_max":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.all_reduce_max(val), F32)
        if name == "all_reduce_min":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.all_reduce_min(val), F32)
        if name == "broadcast":
            val, vtag = self._expr(node.args[0])
            lane = 0
            if len(node.args) > 1:
                lane_v, lane_t = self._expr(node.args[1])
                lane = int(lane_v) if _is_ct(lane_t) else lane_v
            return (self.ctx.subgroup_broadcast(val, lane), F32)
        if name == "warp_prefix_sum":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.warp_prefix_sum_inclusive(val), F32)
        if name == "warp_prefix_sum_exclusive":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.warp_prefix_sum_exclusive(val), F32)
        if name == "mma_reshape":
            buf_name = node.args[0].id
            buf_idx = self.buf_map[buf_name]
            rows = self._eval_ct(node.args[1])
            cols = self._eval_ct(node.args[2])
            return (self.ctx._reshape_buf_2d(buf_idx, rows, cols), "memref_2d")
        if name == "mma_load_a":
            buf_2d, _ = self._expr(node.args[0])
            row = self._idx(self._expr(node.args[1]))
            col = self._idx(self._expr(node.args[2]))
            lead_dim = self._eval_ct(node.args[3])
            tile_m = self._eval_ct(node.args[4])
            tile_k = self._eval_ct(node.args[5])
            return (self.ctx.mma_load_a(buf_2d, row, col, lead_dim, tile_m, tile_k), MMA_FRAG)
        if name == "mma_load_b":
            buf_2d, _ = self._expr(node.args[0])
            row = self._idx(self._expr(node.args[1]))
            col = self._idx(self._expr(node.args[2]))
            lead_dim = self._eval_ct(node.args[3])
            tile_k = self._eval_ct(node.args[4])
            tile_n = self._eval_ct(node.args[5])
            return (self.ctx.mma_load_b(buf_2d, row, col, lead_dim, tile_k, tile_n), MMA_FRAG)
        if name == "mma_zero_c":
            tile_m = self._eval_ct(node.args[0])
            tile_n = self._eval_ct(node.args[1])
            return (self.ctx.mma_zero_c(tile_m, tile_n), MMA_FRAG)
        if name == "mma_compute":
            a, _ = self._expr(node.args[0])
            b, _ = self._expr(node.args[1])
            c, _ = self._expr(node.args[2])
            return (self.ctx.mma_compute(a, b, c), MMA_FRAG)
        if name == "mma_store":
            frag, _ = self._expr(node.args[0])
            buf_2d, _ = self._expr(node.args[1])
            row = self._idx(self._expr(node.args[2]))
            col = self._idx(self._expr(node.args[3]))
            lead_dim = self._eval_ct(node.args[4])
            self.ctx.mma_store(frag, buf_2d, row, col, lead_dim)
            return (None, CT_INT)
        if name == "exp":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.exp(val), F32)
        if name == "sqrt":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.sqrt(val), F32)
        if name == "rsqrt":
            val, vtag = self._expr(node.args[0])
            return (self.ctx.rsqrt(val), F32)
        raise NotImplementedError(f"Unknown function: {name}")

    def _method_call(self, node):
        """Handle obj.method(args) calls — e.g., tc.load_a(buf, row, col, ld)."""
        obj_name = node.func.value.id
        method = node.func.attr
        obj_val, obj_tag = self.env.get(obj_name, (None, None))
        if obj_val is None and obj_name in self.outer:
            obj_val = self.outer[obj_name]
            obj_tag = CT_OBJ if isinstance(obj_val, _TensorCoreHandle) else None

        if obj_tag == CT_OBJ and hasattr(obj_val, method):
            fn = getattr(obj_val, method)
            if method == "zero":
                return (fn(), MMA_FRAG)
            if method in ("load_a", "load_b"):
                buf_2d, _ = self._expr(node.args[0])
                row = self._idx(self._expr(node.args[1]))
                col = self._idx(self._expr(node.args[2]))
                lead_dim = self._eval_ct(node.args[3])
                return (fn(buf_2d, row, col, lead_dim), MMA_FRAG)
            if method == "mma":
                a, _ = self._expr(node.args[0])
                b, _ = self._expr(node.args[1])
                c, _ = self._expr(node.args[2])
                return (fn(a, b, c), MMA_FRAG)
            if method == "store":
                frag, _ = self._expr(node.args[0])
                buf_2d, _ = self._expr(node.args[1])
                row = self._idx(self._expr(node.args[2]))
                col = self._idx(self._expr(node.args[3]))
                lead_dim = self._eval_ct(node.args[4])
                fn(frag, buf_2d, row, col, lead_dim)
                return (None, CT_INT)
        raise NotImplementedError(f"Method call: {obj_name}.{method}")
