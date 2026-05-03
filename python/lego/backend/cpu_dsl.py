"""
Pure-Python CPU kernel DSL with AST transformation.

Transforms native Python syntax (arithmetic, for, if, while, indexing)
into MLIR CPU IR (scf + arith + memref + Lego ops) via the
CPUKernelBuilder / CPUKernelContext infrastructure.

Mirrors ``gpu_dsl.py`` exactly in architecture; drops GPU-specific
primitives (block_id, thread_id, Shared, tensor_core, warp shuffles) and
adds CPU-specific ``tile`` parameter.

Usage::

    from lego.backend.cpu_dsl import cpu_kernel, Buffer

    N = 1024

    @cpu_kernel(grid=(N,), tile=(8,))
    def saxpy(a: float, X: Buffer[N], Y: Buffer[N]):
        for i in tile_range:
            Y[i] = a * X[i] + Y[i]

    # Compile to x86 AVX-512 and run:
    jit_fn = saxpy.compile()
    jit_fn(2.5, X_np, Y_np)   # modifies Y_np in-place

Key differences from gpu_dsl:
  - ``@cpu_kernel(grid=(N,), tile=(T,))`` — no ``block`` parameter.
  - ``tile_range`` is a magic sentinel used inside kernel bodies for the
    within-tile loop variable; the decorator rewrites it to ``range(tile)``.
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

from lego.mlir.ir import InsertionPoint, F32Type, IntegerType
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import scf as scf_dialect

from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.cpu_builder import CPUKernelBuilder, LayoutBuffer
from lego.backend._ops import _index_const


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
# Value tags (identical to gpu_dsl — shared semantics)
# ============================================================================

CT_INT   = "ct_int"    # compile-time Python int
CT_FLOAT = "ct_float"  # compile-time Python float
INDEX    = "index"     # MLIR index Value
F32      = "f32"       # MLIR f32 Value
I1       = "i1"        # MLIR i1 Value
CT_OBJ   = "ct_obj"   # compile-time Python object (layouts, etc.)


def _is_ct(tag):
    return tag in (CT_INT, CT_FLOAT)


def _to_runtime(ctx, val, tag, target_tag=None):
    """Promote a compile-time Python value to an MLIR constant."""
    if tag == CT_FLOAT or target_tag == F32:
        return ctx.const_f32(float(val)), F32
    return ctx.const_index(int(val)), INDEX


def _promote(ctx, lv, lt, rv, rt):
    """Ensure both operands are runtime, matching types."""
    if _is_ct(lt) and _is_ct(rt):
        return lv, lt, rv, rt
    if _is_ct(lt):
        lv, lt = _to_runtime(ctx, lv, lt, rt)
    if _is_ct(rt):
        rv, rt = _to_runtime(ctx, rv, rt, lt)
    return lv, lt, rv, rt


# ============================================================================
# Decorator
# ============================================================================

# Sentinel object used as the iter in ``for i in tile_range:``
class _TileRangeSentinel:
    """Magic sentinel that ``for i in tile_range:`` is rewritten to range(tile)."""
tile_range = _TileRangeSentinel()


def cpu_kernel(grid: Tuple, tile: Optional[Tuple] = None):
    """Decorator: transform a Python function into a :class:`CPUKernelBuilder`.

    Args:
        grid: Outer iteration shape, e.g. ``(N,)`` or ``(M, N)``.
              The kernel body runs once per grid point.
        tile: Optional per-task tile shape, e.g. ``(8,)`` or ``(4, 4)``.
              If given, the kernel body is wrapped in an outer scf.for over
              the grid tiles; the user writes ``for i in tile_range:`` to
              iterate within the tile.  If omitted, grid is treated as the
              per-element range.
    """
    def decorator(fn):
        return _build(fn, grid, tile)
    return decorator


def _build(fn, grid, tile):
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
        if tile is not None:
            # Outer grid loop: iterates over tiles (tile_id = 0 .. num_tiles-1).
            num_tiles = grid[0] // tile[0]
            outer_loop = scf_dialect.ForOp(
                _index_const(0), _index_const(num_tiles), _index_const(1))
            with InsertionPoint(outer_loop.body):
                tile_id = outer_loop.induction_variable
                _Compiler(
                    ctx=ctx,
                    func_def=func_def,
                    buf_params=buf_params,
                    scalar_params=scalar_params_meta,
                    outer=outer,
                    grid=grid,
                    tile=tile,
                    tile_id=tile_id,
                ).run()
                scf_dialect.YieldOp([])
        else:
            _Compiler(
                ctx=ctx,
                func_def=func_def,
                buf_params=buf_params,
                scalar_params=scalar_params_meta,
                outer=outer,
                grid=grid,
                tile=tile,
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

class _Compiler:
    """Walk a Python function AST and emit MLIR via CPUKernelContext.

    Mirrors gpu_dsl._Compiler — same dispatch table, same value-tag system,
    same compile-time / runtime promotion rules.  GPU-specific constructs
    (block_id, thread_id, Shared, mma_sync, warp shuffles) are absent.

    CPU additions:
    - Scalar function arguments appear in ``env`` as f32 / index Values.
    - ``tile_range`` sentinel in ``for i in tile_range:`` is rewritten to
      ``range(tile[0])`` (using the tile parameter from ``@cpu_kernel``).
    """

    def __init__(self, ctx, func_def, buf_params, scalar_params, outer,
                 grid=None, tile=None, tile_id=None):
        self.ctx = ctx
        self.func_def = func_def
        self.outer = outer
        self.grid = grid
        self.tile = tile
        self._tile_id = tile_id   # outer-loop IV (MLIR Value); None if no tiling
        self.env = {}         # name → (value, tag)
        self.buf_map = {}     # name → buffer index (among buf_params only)

        # Populate env with scalar function arguments (MLIR Values already
        # bound by the function's entry block, exposed via ctx._scalar_vals).
        scalar_vals = getattr(ctx, '_scalar_vals', [])
        for i, (name, dtype_str) in enumerate(scalar_params):
            if i < len(scalar_vals):
                tag = F32 if dtype_str == "f32" else INDEX
                self.env[name] = (scalar_vals[i], tag)

        # Map buffer parameter names → indices
        for i, (name, _) in enumerate(buf_params):
            self.buf_map[name] = i

    def run(self):
        for stmt in self.func_def.body:
            self._stmt(stmt)

    # ------------------------------------------------------------------ stmts

    def _stmt(self, node):
        if isinstance(node, ast.Assign):
            self._assign(node)
        elif isinstance(node, ast.AugAssign):
            self._aug_assign(node)
        elif isinstance(node, ast.For):
            self._for(node)
        elif isinstance(node, ast.If):
            self._if(node)
        elif isinstance(node, ast.While):
            self._while(node)
        elif isinstance(node, ast.Expr):
            self._expr(node.value)
        else:
            raise NotImplementedError(f"Statement {type(node).__name__}")

    def _assign(self, node):
        assert len(node.targets) == 1
        tgt = node.targets[0]
        val, tag = self._expr(node.value)
        if isinstance(tgt, ast.Name):
            self.env[tgt.id] = (val, tag)
        elif isinstance(tgt, ast.Tuple):
            assert isinstance(val, (list, tuple)), "Tuple unpack requires tuple-valued RHS"
            for i, elt in enumerate(tgt.elts):
                self.env[elt.id] = (val[i], INDEX)
        elif isinstance(tgt, ast.Subscript):
            self._store(tgt, val, tag)
        else:
            raise NotImplementedError(f"Assign to {type(tgt).__name__}")

    def _aug_assign(self, node):
        rval, rtag = self._expr(node.value)
        if isinstance(node.target, ast.Name):
            lv, lt = self.env[node.target.id]
            lv, lt, rval, rtag = _promote(self.ctx, lv, lt, rval, rtag)
            self.env[node.target.id] = self._binop_rt(lv, lt, rval, rtag, type(node.op))
        elif isinstance(node.target, ast.Subscript):
            lv, lt = self._load(node.target)
            lv, lt, rval, rtag = _promote(self.ctx, lv, lt, rval, rtag)
            res_val, res_tag = self._binop_rt(lv, lt, rval, rtag, type(node.op))
            self._store(node.target, res_val, res_tag)
        else:
            raise NotImplementedError

    # -- for -----------------------------------------------------------

    def _for(self, node):
        var = node.target.id
        call = node.iter

        # Detect ``for i in tile_range:`` — rewrite to range(tile[0])
        _is_tile_range = isinstance(call, ast.Name) and call.id == "tile_range"
        if _is_tile_range:
            # Substitute with range(tile)
            if self.tile is None:
                raise ValueError(
                    "``tile_range`` used but @cpu_kernel was not given a ``tile`` arg"
                )
            tile_size = self.tile[0]
            ub = _index_const(tile_size)
            lb = _index_const(0)
            step = _index_const(1)
        else:
            # Regular range(…) call
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name), \
                f"For iter must be range() or tile_range, got {ast.dump(call)}"
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
            if _is_tile_range and self._tile_id is not None:
                # Global index = tile_id * tile_size + local_i
                # so that buffer accesses use the correct global offset.
                tile_size_val = _index_const(self.tile[0])
                base = arith_dialect.MulIOp(self._tile_id, tile_size_val).result
                global_iv = arith_dialect.AddIOp(
                    base, loop.induction_variable).result
                self.env[var] = (global_iv, INDEX)
            else:
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

    # -- if ------------------------------------------------------------

    def _if(self, node):
        cv, ct = self._expr(node.test)
        if _is_ct(ct):                           # compile-time branch
            stmts = node.body if cv else node.orelse
            for s in stmts:
                self._stmt(s)
            return

        env_before = set(self.env)
        modified = (self._modified_names(node.body)
                    | self._modified_names(node.orelse)) & env_before
        yield_names = sorted(modified)

        if not yield_names:
            has_else = bool(node.orelse)
            if_op = scf_dialect.IfOp(cv, has_else=has_else)
            with InsertionPoint(if_op.then_block):
                for s in node.body:
                    self._stmt(s)
                scf_dialect.YieldOp([])
            if has_else:
                with InsertionPoint(if_op.else_block):
                    for s in node.orelse:
                        self._stmt(s)
                    scf_dialect.YieldOp([])
            return

        init_vals, init_tags = [], []
        for n in yield_names:
            v, t = self.env[n]
            if _is_ct(t):
                v, t = _to_runtime(self.ctx, v, t)
                self.env[n] = (v, t)
            init_vals.append(v)
            init_tags.append(t)

        result_types = [v.type for v in init_vals]
        if_op = scf_dialect.IfOp(cv, result_types, has_else=True)

        with InsertionPoint(if_op.then_block):
            for n, v, t in zip(yield_names, init_vals, init_tags):
                self.env[n] = (v, t)
            for s in node.body:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in yield_names])

        with InsertionPoint(if_op.else_block):
            for n, v, t in zip(yield_names, init_vals, init_tags):
                self.env[n] = (v, t)
            for s in node.orelse:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in yield_names])

        for i, n in enumerate(yield_names):
            self.env[n] = (if_op.results[i], init_tags[i])

    # -- while (compile-time unroll) -----------------------------------

    def _while(self, node):
        while True:
            cv, ct = self._expr(node.test)
            assert _is_ct(ct), "Runtime while not supported — use for + range()"
            if not cv:
                break
            for s in node.body:
                self._stmt(s)

    # ---------------------------------------------------------------- exprs

    def _expr(self, node):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, float):
                return (v, CT_FLOAT)
            return (v, CT_INT)
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.Subscript):
            return self._load(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise NotImplementedError(f"Expr {type(node).__name__}")

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

    # -- binary ops ----------------------------------------------------

    def _binop(self, node):
        lv, lt = self._expr(node.left)
        rv, rt = self._expr(node.right)
        if _is_ct(lt) and _is_ct(rt):
            return self._binop_ct(lv, lt, rv, rt, type(node.op))
        lv, lt, rv, rt = _promote(self.ctx, lv, lt, rv, rt)
        return self._binop_rt(lv, lt, rv, rt, type(node.op))

    @staticmethod
    def _binop_ct(lv, lt, rv, rt, op):
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
               ast.Mod: lambda a, b: a % b, ast.Div: lambda a, b: a / b,
               ast.Pow: lambda a, b: a ** b}
        r = ops[op](lv, rv)
        tag = CT_FLOAT if isinstance(r, float) or lt == CT_FLOAT or rt == CT_FLOAT else CT_INT
        return (r if tag == CT_FLOAT else int(r), tag)

    def _binop_rt(self, lv, lt, rv, rt, op):
        if lt == F32 or rt == F32:
            # Promote index operand to f32 if needed
            if lt == INDEX:
                i32 = IntegerType.get_signless(32)
                lv = arith_dialect.IndexCastOp(i32, lv).result
                lv = arith_dialect.SIToFPOp(F32Type.get(), lv).result
                lt = F32
            if rt == INDEX:
                i32 = IntegerType.get_signless(32)
                rv = arith_dialect.IndexCastOp(i32, rv).result
                rv = arith_dialect.SIToFPOp(F32Type.get(), rv).result
                rt = F32
            m = {ast.Add: self.ctx.addf, ast.Sub: self.ctx.subf,
                 ast.Mult: self.ctx.mulf,
                 ast.Div: lambda a, b: arith_dialect.DivFOp(a, b).result}
            return (m[op](lv, rv), F32)
        m = {ast.Add: self.ctx.addi,
             ast.Sub: lambda a, b: arith_dialect.SubIOp(a, b).result,
             ast.Mult: self.ctx.muli,
             ast.FloorDiv: lambda a, b: arith_dialect.FloorDivSIOp(a, b).result,
             ast.Mod: lambda a, b: arith_dialect.RemUIOp(a, b).result}
        return (m[op](lv, rv), INDEX)

    def _unary(self, node):
        v, t = self._expr(node.operand)
        if isinstance(node.op, ast.USub):
            if _is_ct(t):
                return (-v, t)
            if t == F32:
                return (self.ctx.subf(self.ctx.const_f32(0.0), v), F32)
            return (arith_dialect.SubIOp(_index_const(0), v).result, INDEX)
        raise NotImplementedError(f"UnaryOp {type(node.op).__name__}")

    # -- comparisons ---------------------------------------------------

    def _compare(self, node):
        assert len(node.ops) == 1
        lv, lt = self._expr(node.left)
        rv, rt = self._expr(node.comparators[0])
        op = type(node.ops[0])
        if _is_ct(lt) and _is_ct(rt):
            m = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                 ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
                 ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}
            return (m[op](lv, rv), CT_INT)
        lv, lt, rv, rt = _promote(self.ctx, lv, lt, rv, rt)
        if lt == F32:
            m = {ast.Lt: arith_dialect.CmpFPredicate.OLT,
                 ast.LtE: arith_dialect.CmpFPredicate.OLE,
                 ast.Gt: arith_dialect.CmpFPredicate.OGT,
                 ast.GtE: arith_dialect.CmpFPredicate.OGE,
                 ast.Eq: arith_dialect.CmpFPredicate.OEQ,
                 ast.NotEq: arith_dialect.CmpFPredicate.ONE}
            return (arith_dialect.CmpFOp(m[op], lv, rv).result, I1)
        m = {ast.Lt: arith_dialect.CmpIPredicate.ult,
             ast.LtE: arith_dialect.CmpIPredicate.ule,
             ast.Gt: arith_dialect.CmpIPredicate.ugt,
             ast.GtE: arith_dialect.CmpIPredicate.uge,
             ast.Eq: arith_dialect.CmpIPredicate.eq,
             ast.NotEq: arith_dialect.CmpIPredicate.ne}
        return (arith_dialect.CmpIOp(m[op], lv, rv).result, I1)

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
        return (self.ctx.load(idx, self._indices(node)), F32)

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

    def _indices(self, node):
        if isinstance(node.slice, ast.Tuple):
            elts = node.slice.elts
        else:
            elts = [node.slice]
        out = []
        for e in elts:
            v, t = self._expr(e)
            if _is_ct(t):
                v, t = _to_runtime(self.ctx, v, t, INDEX)
            out.append(v)
        return out

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

    def _eval_ct(self, node):
        """Evaluate an AST node as a compile-time Python expression."""
        ns = dict(self.outer)
        for name, (val, tag) in self.env.items():
            if _is_ct(tag):
                ns[name] = val
        return eval(compile(ast.Expression(node), "<ct>", "eval"), ns)  # noqa: S307

    # ---------------------------------------------------------------- util

    def _idx(self, pair):
        """Ensure a (val, tag) pair is an MLIR index Value."""
        v, t = pair
        if _is_ct(t):
            return _index_const(int(v))
        return v

    def _modified_names(self, stmts):
        """Collect names assigned (including augmented) in a statement list."""
        out = set()
        for s in stmts:
            out |= self._collect_assigns(s)
        return out

    def _collect_assigns(self, node):
        s = set()
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    s.add(t.id)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                s.add(node.target.id)
        elif isinstance(node, (ast.For, ast.While)):
            for st in node.body:
                s |= self._collect_assigns(st)
        elif isinstance(node, ast.If):
            for st in node.body + node.orelse:
                s |= self._collect_assigns(st)
        return s
