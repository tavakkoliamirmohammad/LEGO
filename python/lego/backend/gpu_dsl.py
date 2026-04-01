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
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer, _index_const


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
# Value tags for the compiler's type system
# ============================================================================

CT_INT = "ct_int"        # compile-time Python int
CT_FLOAT = "ct_float"    # compile-time Python float
INDEX = "index"          # MLIR index Value
F32 = "f32"              # MLIR f32 Value
I1 = "i1"               # MLIR i1 Value


def _is_ct(tag):
    return tag in (CT_INT, CT_FLOAT)


def _to_runtime(ctx, val, tag, target_tag=None):
    """Promote a compile-time value to an MLIR constant."""
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

class _Compiler:
    """Walk a Python function AST and emit MLIR via KernelContext."""

    _GPU_DIMS = {"block_id", "thread_id", "block_dim"}

    def __init__(self, ctx, func_def, buf_params, outer):
        self.ctx = ctx
        self.func_def = func_def
        self.outer = outer
        self.env = {}                     # name → (value, tag)
        self.buf_map = {}                 # name → buffer index
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
            # Tuple unpacking: a, b = apply_inverse(…)
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
        # Parse range(…)
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

        # Detect iter-args: outer vars modified in the body
        # Only variables that already exist become iter_args; new vars
        # defined inside the body are loop-local and removed afterwards.
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
            self.env[var] = (loop.induction_variable, INDEX)
            for i, n in enumerate(ia_names):
                self.env[n] = (loop.inner_iter_args[i], ia_tags[i])
            for s in node.body:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in ia_names])

        for i, n in enumerate(ia_names):
            self.env[n] = (loop.results[i], ia_tags[i])
        # Remove loop variable and any vars defined only inside the body
        # (their MLIR Values don't dominate outside the loop region)
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

        # Detect which env variables are modified in then/else bodies.
        # These must be yielded from scf.if so their updated values
        # dominate uses after the if.
        env_before = set(self.env)
        modified = (self._modified_names(node.body)
                    | self._modified_names(node.orelse)) & env_before
        yield_names = sorted(modified)

        if not yield_names:
            # No local variables modified — simple fire-and-forget if
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

        # Promote init vals to runtime so scf.if can yield them
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

        # Then block
        with InsertionPoint(if_op.then_block):
            # Reset env to use block args (init vals dominate inside region)
            for n, v, t in zip(yield_names, init_vals, init_tags):
                self.env[n] = (v, t)
            for s in node.body:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in yield_names])

        # Else block — execute else body if present, otherwise pass through
        with InsertionPoint(if_op.else_block):
            for n, v, t in zip(yield_names, init_vals, init_tags):
                self.env[n] = (v, t)
            for s in node.orelse:
                self._stmt(s)
            scf_dialect.YieldOp([self.env[n][0] for n in yield_names])

        # After the if: env points to the if-op results
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
            raise TypeError(f"Unsupported type for '{name}': {type(v)}")
        raise NameError(f"Undefined: {name}")

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
            # Promote index operand to f32 if needed (e.g., A[i] * (block_id + 1))
            # index → i32 → f32 (arith.sitofp requires integer, not index)
            if lt == INDEX:
                from lego.mlir.ir import IntegerType
                i32 = IntegerType.get_signless(32)
                lv = arith_dialect.IndexCastOp(i32, lv).result
                lv = arith_dialect.SIToFPOp(F32Type.get(), lv).result
                lt = F32
            if rt == INDEX:
                from lego.mlir.ir import IntegerType
                i32 = IntegerType.get_signless(32)
                rv = arith_dialect.IndexCastOp(i32, rv).result
                rv = arith_dialect.SIToFPOp(F32Type.get(), rv).result
                rt = F32
            m = {ast.Add: self.ctx.addf, ast.Sub: self.ctx.subf,
                 ast.Mult: self.ctx.mulf, ast.Div: lambda a, b: arith_dialect.DivFOp(a, b).result}
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

    # -- attribute access (block_id.x, …) ------------------------------

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
        # --- Warp / subgroup operations ---
        if name == "lane_id":
            return (self.ctx.lane_id(), INDEX)
        if name == "warp_size":
            return (self.ctx.subgroup_size(), INDEX)
        if name in ("shuffle_down", "shuffle_up", "shuffle_xor", "shuffle_idx"):
            val, vtag = self._expr(node.args[0])
            arg1_v, arg1_t = self._expr(node.args[1])
            # offset/mask/lane: pass as Python int for constant, or MLIR value
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
        # --- Block-wide operations ---
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
        raise NotImplementedError(f"Unknown function: {name}")

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
