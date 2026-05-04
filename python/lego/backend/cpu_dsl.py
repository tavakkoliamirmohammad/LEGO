"""
Pure-Python CPU kernel DSL with AST transformation.

Transforms native Python syntax (arithmetic, for, if, while, indexing)
into MLIR CPU IR (scf + arith + memref + Lego ops) via the
CPUKernelBuilder / CPUKernelContext infrastructure.

Mirrors ``gpu_dsl.py`` exactly in architecture; drops GPU-specific
primitives (block_id, thread_id, Shared, tensor_core, warp shuffles).

Usage::

    from lego.backend.cpu_dsl import cpu_kernel, Buffer

    N = 1024

    @cpu_kernel
    def saxpy(a: float, X: Buffer[N], Y: Buffer[N]):
        for i in range(N):
            Y[i] = a * X[i] + Y[i]

    # Compile to x86 AVX-512 and run:
    jit_fn = saxpy.compile()
    jit_fn(2.5, X_np, Y_np)   # modifies Y_np in-place

Key differences from gpu_dsl:
  - ``@cpu_kernel`` — no decorator arguments. Iteration is described by
    explicit ``for i in range(N):`` in the body.
  - No ``Shared[...]``, no ``block_id``/``thread_id``/``block_dim``.
  - Scalar parameters are allowed (annotated with Python ``float`` or
    ``int``); they map to f32/index function arguments.
  - ``Buffer[N]`` → flat memref of size N (same as gpu_dsl).
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Optional, Tuple

from lego.mlir.ir import InsertionPoint, IntegerType, IndexType
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import scf as scf_dialect

from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.cpu_builder import CPUKernelBuilder, LayoutBuffer
from lego.backend._ops import _index_const
from lego.backend._dsl_base import (
    _BaseCompiler,
    CT_INT, CT_FLOAT, INDEX, F32, I1, I32, CT_OBJ,
    _is_ct, _to_runtime, _promote,
)


# ============================================================================
# Type annotations for kernel parameters
# ============================================================================

class Buffer:
    """CPU buffer parameter type.

    Two forms::

        Buffer[N]          # Row(N) layout (1-D)
        Buffer[M, K]       # Row(M, K) layout (2-D)
        Buffer(layout, N)  # custom LEGO layout

    Unlike gpu_dsl.Buffer there is no ``Shared`` variant on CPU.
    """

    def __new__(cls, layout, *dims, dtype=DType.f32):
        return _BufferType(dims=tuple(dims), dtype=dtype, layout=layout)

    def __class_getitem__(cls, dims):
        if not isinstance(dims, tuple):
            dims = (dims,)
        return _BufferType(dims=dims)


@dataclass
class _BufferType:
    dims: tuple
    dtype: DType = DType.f32
    layout: object = None       # None → default Row(*dims)


# ============================================================================
# Decorator
# ============================================================================

def cpu_kernel(fn=None):
    """Decorator: transform a Python function into a :class:`CPUKernelBuilder`.

    The kernel's iteration space is described entirely by the body's
    ``for i in range(...):`` loops. SIMD width is picked automatically by
    the vectoriser from the target hardware. No decorator arguments are
    required — both ``@cpu_kernel`` and ``@cpu_kernel()`` are accepted.

    Example::

        N = 1 << 20

        @cpu_kernel
        def saxpy(a: float, X: Buffer[N], Y: Buffer[N]):
            for i in range(N):
                Y[i] = a * X[i] + Y[i]

        jit_fn = saxpy.compile()       # selects x86 / AVX-512 automatically
        jit_fn(2.5, X_np, Y_np)       # modifies Y_np in-place
    """
    if fn is None:
        return lambda f: _build(f)
    return _build(fn)


def _build(fn):
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    func_def = tree.body[0]

    # Resolve outer-scope names (N, TILE, K, …)
    outer = fn.__globals__.copy()
    if fn.__code__.co_freevars and fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            outer[name] = cell.cell_contents

    # Parse parameter annotations: float/int → scalar, Buffer[…] → memref
    scalar_params_meta = []  # list of (name, dtype_str)
    buf_params = []           # list of (name, _BufferType)

    for arg in func_def.args.args:
        ann_node = arg.annotation
        # Evaluate annotation in outer scope
        ann = eval(                                       # noqa: S307
            compile(ast.Expression(ann_node), "<ann>", "eval"), outer)
        if isinstance(ann, _BufferType):
            dims = tuple(int(d) for d in ann.dims)
            buf_params.append((arg.arg, _BufferType(dims, ann.dtype, ann.layout)))
        elif ann is float or ann == "float":
            scalar_params_meta.append((arg.arg, "f32"))
        elif ann is int or ann == "int":
            scalar_params_meta.append((arg.arg, "index"))
        else:
            # Unknown annotation: try treating as scalar f32
            scalar_params_meta.append((arg.arg, "f32"))

    buffers = [
        LayoutBuffer(bt.layout or Row(*bt.dims), shape=bt.dims, dtype=bt.dtype)
        for _, bt in buf_params
    ]

    scalar_dtypes = [dtype_str for _, dtype_str in scalar_params_meta]

    def kernel_body(ctx):
        # The body's ``for i in range(...):`` describes the iteration
        # space directly; the compiler walks the AST and emits scf.for ops
        # one-to-one. lego-vectorize then strip-mines the inner loop.
        _Compiler(
            ctx=ctx,
            func_def=func_def,
            buf_params=buf_params,
            scalar_params=scalar_params_meta,
            outer=outer,
        ).run()

    return CPUKernelBuilder(
        buffers=buffers,
        kernel_body=kernel_body,
        name=fn.__name__,
        scalar_params=scalar_dtypes,
    )


# ============================================================================
# AST → MLIR compiler
# ============================================================================

class _Compiler(_BaseCompiler):
    """Walk a Python function AST and emit MLIR via CPUKernelContext.

    Inherits shared AST-dispatch, binop, if/while, compare, and utility
    methods from _BaseCompiler.  Only CPU-specific overrides are here:
    - __init__: scalar_params (no grid/tile — kernels use plain range()).
    - _for: emits scf.for from ``for i in range(...):`` directly; the
      vectoriser picks SIMD width from the target.
    - _name: scalar function arguments; no GPU dimension names.
    - _attribute: rejects GPU-only dimension accessors.
    - _load/_store: strict error checking on buffer name.
    - _call/_method_call: CPU ops only; clear error for GPU ops.
    """

    def __init__(self, ctx, func_def, buf_params, scalar_params, outer):
        self.ctx = ctx
        self.func_def = func_def
        self.outer = outer
        self.env = {}         # name → (value, tag)
        self.buf_map = {}     # name → buffer index (among buf_params only)
        self.buf_dtype = {}   # name → dtype string ("f32"/"f64"/"i32"/"i64"/...)

        # Populate env with scalar function arguments (MLIR Values already
        # bound by the function's entry block, exposed via ctx._scalar_vals).
        scalar_vals = getattr(ctx, '_scalar_vals', [])
        for i, (name, dtype_str) in enumerate(scalar_params):
            if i < len(scalar_vals):
                tag = F32 if dtype_str == "f32" else INDEX
                self.env[name] = (scalar_vals[i], tag)

        # Map buffer parameter names → indices and remember each buffer's
        # element dtype.  The ``_load`` site uses this to tag the loaded
        # value correctly so subsequent uses (e.g., as a subscript index)
        # see the right element type.
        for i, (name, bt) in enumerate(buf_params):
            self.buf_map[name] = i
            # bt.dtype is a DType enum that inherits str — use the string value.
            dt = bt.dtype
            self.buf_dtype[name] = dt.value if hasattr(dt, "value") else str(dt)

    # -- for -----------------------------------------------------------

    def _for(self, node):
        var = node.target.id
        call = node.iter

        # The body must use ``for i in range(...):``.
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name), \
            f"For iter must be range(), got {ast.dump(call)}"
        assert call.func.id == "range"
        args = [self._expr(a) for a in call.args]
        if len(args) == 1:
            lb, ub, step = (_index_const(0), self._idx(args[0]), _index_const(1))
        elif len(args) == 2:
            lb, ub, step = (self._idx(args[0]), self._idx(args[1]), _index_const(1))
        else:
            lb, ub, step = (self._idx(args[0]), self._idx(args[1]), self._idx(args[2]))

        # Detect iter-args: outer vars modified in the body
        env_before = set(self.env)
        modified = self._modified_names(node.body) & env_before
        modified.discard(var)
        ia_names = sorted(modified)

        # Convert init vals to runtime
        ia_vals, ia_tags = [], []
        for n in ia_names:
            v, t = self.env[n]
            if _is_ct(t):
                v, t = _to_runtime(self.ctx, v, t)
            ia_vals.append(v)
            ia_tags.append(t)

        loop = scf_dialect.ForOp(lb, ub, step, ia_vals or None)
        with InsertionPoint(loop.body):
            # In the flat-loop model, the induction variable directly is the
            # global index — no tile_id offset arithmetic needed.
            self.env[var] = (loop.induction_variable, INDEX)
            for i, n in enumerate(ia_names):
                self.env[n] = (loop.inner_iter_args[i], ia_tags[i])
            for s in node.body:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in ia_names])

        for i, n in enumerate(ia_names):
            self.env[n] = (loop.results[i], ia_tags[i])
        # Remove loop-local vars (don't dominate outside the loop)
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
            # Compile-time objects (layouts, etc.)
            raise TypeError(f"Unsupported type for '{name}': {type(v)}")
        raise NameError(f"Undefined name: '{name}'")

    # -- attribute access ----------------------------------------------

    def _attribute(self, node):
        # No GPU dimension accessors on CPU; raise clearly.
        if isinstance(node.value, ast.Name):
            if node.value.id in ("block_id", "thread_id", "block_dim"):
                raise RuntimeError(
                    f"'{node.value.id}.{node.attr}' is a GPU-only construct; "
                    f"not available in @cpu_kernel."
                )
        raise NotImplementedError(f"Attribute {ast.dump(node)}")

    # -- buffer load / store -------------------------------------------

    def _load(self, node):
        if not isinstance(node.value, ast.Name):
            raise NotImplementedError(
                f"Subscript load on non-Name: {ast.dump(node.value)}")
        name = node.value.id
        if name not in self.buf_map:
            raise NameError(
                f"'{name}' is not a declared Buffer parameter in this kernel")
        idx = self.buf_map[name]
        # Tag the loaded value with the buffer's actual element dtype so
        # downstream uses (subscript indices, arithmetic) see the right type.
        return (self.ctx.load(idx, self._indices(node)),
                self.buf_dtype.get(name, F32))

    def _store(self, node, val, tag):
        if _is_ct(tag):
            val, tag = _to_runtime(self.ctx, val, tag, F32)
        if not isinstance(node.value, ast.Name):
            raise NotImplementedError(
                f"Subscript store on non-Name: {ast.dump(node.value)}")
        name = node.value.id
        if name not in self.buf_map:
            raise NameError(
                f"'{name}' is not a declared Buffer parameter in this kernel")
        idx = self.buf_map[name]
        self.ctx.store(val, idx, self._indices(node))

    # -- function calls ------------------------------------------------

    def _call(self, node):
        if isinstance(node.func, ast.Attribute):
            return self._method_call(node)
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError(f"Call {ast.dump(node.func)}")
        name = node.func.id

        # Layout utilities (shared with gpu_dsl)
        if name == "apply_inverse":
            from lego.backend.compiler import _get_layout_dims
            from lego.backend.symbolic import emit_layout_from_python
            from lego.backend._ops import _emit_apply_inverse
            layout = self._eval_ct(node.args[0])
            idx_val = self._idx(self._expr(node.args[1]))
            rank = len(_get_layout_dims(layout))
            layout_val = emit_layout_from_python(layout, {})
            result = _emit_apply_inverse(layout_val, idx_val, rank)
            return (list(result), "tuple")
        if name == "apply":
            from lego.backend.symbolic import emit_layout_from_python
            from lego.backend._ops import _emit_apply
            layout = self._eval_ct(node.args[0])
            indices = [self._idx(self._expr(a)) for a in node.args[1:]]
            layout_val = emit_layout_from_python(layout, {})
            return (_emit_apply(layout_val, indices), INDEX)
        if name == "set_layout":
            buf_name = node.args[0].id
            buf_idx = self.buf_map[buf_name]
            layout = self._eval_ct(node.args[1])
            self.ctx.set_layout(buf_idx, layout)
            return (None, CT_INT)

        # Math operations
        if name == "exp":
            val, _ = self._expr(node.args[0])
            return (self.ctx.exp(val), F32)
        if name == "sqrt":
            val, _ = self._expr(node.args[0])
            return (self.ctx.sqrt(val), F32)
        if name == "rsqrt":
            val, _ = self._expr(node.args[0])
            return (self.ctx.rsqrt(val), F32)
        if name in ("max", "min"):
            a, _ = self._expr(node.args[0])
            b, _ = self._expr(node.args[1])
            op = self.ctx.maximumf if name == "max" else self.ctx.minimumf
            return (op(a, b), F32)

        # GPU-only constructs — raise with a helpful message
        _gpu_only = {
            "barrier", "lane_id", "warp_size", "shuffle_down", "shuffle_up",
            "shuffle_xor", "shuffle_idx", "subgroup_reduce_add",
            "subgroup_reduce_mul", "subgroup_reduce_max", "subgroup_reduce_min",
            "all_reduce_add", "all_reduce_mul", "all_reduce_max", "all_reduce_min",
            "broadcast", "warp_prefix_sum", "warp_prefix_sum_exclusive",
            "mma_reshape", "mma_load_a", "mma_load_b", "mma_zero_c",
            "mma_compute", "mma_store",
        }
        if name in _gpu_only:
            raise RuntimeError(
                f"'{name}()' is a GPU-only operation; not available in @cpu_kernel."
            )

        raise NotImplementedError(f"Unknown function: '{name}'")

    def _method_call(self, node):
        """Handle obj.method(args) — not supported for CPU kernels in v1."""
        obj_name = node.func.value.id if isinstance(node.func.value, ast.Name) else "?"
        method = node.func.attr
        raise NotImplementedError(
            f"Method call '{obj_name}.{method}()' not supported in @cpu_kernel v1."
        )
