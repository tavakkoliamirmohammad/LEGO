"""_BaseCompiler — shared AST-to-MLIR compiler core.

Both :class:`cpu_dsl._Compiler` and :class:`gpu_dsl._Compiler` walk the same
Python AST node types and emit the same SCF / arith / memref dialect ops for
the common subset of operations.  This module factors out all shared methods
into :class:`_BaseCompiler` so neither subclass duplicates them.

Override points (subclasses must implement):
    ``_for()``          — CPU adds tile_range detection; GPU uses range() only.
    ``_name()``         — GPU handles block_id/thread_id; CPU handles scalars.
    ``_attribute()``    — GPU resolves e.g. block_id.x; CPU rejects.
    ``_load()``         — different error-checking strictness.
    ``_store()``        — different error-checking strictness.
    ``_call()``         — GPU implements GPU intrinsics; CPU rejects them.
    ``_method_call()``  — GPU implements GPU object methods; CPU rejects.

Constants (re-exported for subclass use):
    CT_INT, CT_FLOAT, INDEX, F32, I1, I32, CT_OBJ
    _is_ct(), _to_runtime(), _promote()
"""

import ast

from lego.mlir.ir import InsertionPoint, F32Type, IntegerType, IndexType
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import scf as scf_dialect

from lego.backend._ops import _index_const

# ---------------------------------------------------------------------------
# Value tags (shared between CPU and GPU DSLs)
# ---------------------------------------------------------------------------

CT_INT   = "ct_int"    # compile-time Python int
CT_FLOAT = "ct_float"  # compile-time Python float
INDEX    = "index"     # MLIR index Value
F32      = "f32"       # MLIR f32 Value
I1       = "i1"        # MLIR i1 Value
I32      = "i32"       # MLIR i32 Value (result of bitwise/shift ops)
CT_OBJ   = "ct_obj"   # compile-time Python object (layouts, TensorCore, …)


def _is_ct(tag):
    return tag in (CT_INT, CT_FLOAT)


def _to_runtime(ctx, val, tag, target_tag=None):
    """Promote a compile-time Python value to an MLIR constant."""
    if tag == CT_FLOAT or target_tag == F32:
        return ctx.const_f32(float(val)), F32
    if target_tag == I32:
        i32 = IntegerType.get_signless(32)
        return arith_dialect.ConstantOp(i32, int(val)).result, I32
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


# ---------------------------------------------------------------------------
# Base compiler class
# ---------------------------------------------------------------------------

class _BaseCompiler:
    """Shared AST-to-MLIR compiler logic for both CPU and GPU DSLs.

    Subclasses must override the abstract methods listed at the top of this
    module (``_for``, ``_name``, ``_attribute``, ``_load``, ``_store``,
    ``_call``, ``_method_call``) and call ``super().__init__()`` after setting
    up any backend-specific state.
    """

    # Class-level sentinel so subclasses can define _BITWISE_OPS once.
    _BITWISE_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)

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

    # -- binary ops ----------------------------------------------------

    def _binop(self, node):
        lv, lt = self._expr(node.left)
        rv, rt = self._expr(node.right)
        op = type(node.op)
        if _is_ct(lt) and _is_ct(rt):
            return self._binop_ct(lv, lt, rv, rt, op)
        # For bitwise/shift ops, promote CT_INT directly to I32 (not INDEX).
        # This avoids generating index_const + index_cast chains which confuse
        # the vectorizer.
        if op in self._BITWISE_OPS:
            i32 = IntegerType.get_signless(32)
            if _is_ct(lt):
                lv = arith_dialect.ConstantOp(i32, int(lv)).result
                lt = I32
            elif lt == INDEX:
                lv = arith_dialect.IndexCastOp(i32, lv).result
                lt = I32
            if _is_ct(rt):
                rv = arith_dialect.ConstantOp(i32, int(rv)).result
                rt = I32
            elif rt == INDEX:
                rv = arith_dialect.IndexCastOp(i32, rv).result
                rt = I32
            return self._binop_rt(lv, lt, rv, rt, op)
        lv, lt, rv, rt = _promote(self.ctx, lv, lt, rv, rt)
        return self._binop_rt(lv, lt, rv, rt, op)

    @staticmethod
    def _binop_ct(lv, lt, rv, rt, op):
        ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
               ast.Mult: lambda a, b: a * b, ast.FloorDiv: lambda a, b: a // b,
               ast.Mod: lambda a, b: a % b, ast.Div: lambda a, b: a / b,
               ast.Pow: lambda a, b: a ** b,
               ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b,
               ast.BitXor: lambda a, b: a ^ b,
               ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b}
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
        # Bitwise / shift ops: cast index → i32, operate, return I32 tag.
        _bitwise_ops = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
        if op in _bitwise_ops:
            i32 = IntegerType.get_signless(32)
            if lt == INDEX:
                lv = arith_dialect.IndexCastOp(i32, lv).result
            elif lt == I32:
                pass  # already i32
            if rt == INDEX:
                rv = arith_dialect.IndexCastOp(i32, rv).result
            elif rt == I32:
                pass  # already i32
            bm = {
                ast.BitAnd: lambda a, b: arith_dialect.AndIOp(a, b).result,
                ast.BitOr:  lambda a, b: arith_dialect.OrIOp(a, b).result,
                ast.BitXor: lambda a, b: arith_dialect.XOrIOp(a, b).result,
                ast.LShift: lambda a, b: arith_dialect.ShLIOp(a, b).result,
                ast.RShift: lambda a, b: arith_dialect.ShRUIOp(a, b).result,
            }
            return (bm[op](lv, rv), I32)
        m = {ast.Add: self.ctx.addi,
             ast.Sub: lambda a, b: arith_dialect.SubIOp(a, b).result,
             ast.Mult: self.ctx.muli,
             ast.FloorDiv: lambda a, b: arith_dialect.FloorDivSIOp(a, b).result,
             ast.Mod: lambda a, b: arith_dialect.RemUIOp(a, b).result}
        # If either operand is I32 from a prior bitwise op, cast to index first.
        if lt == I32:
            lv = arith_dialect.IndexCastOp(IndexType.get(), lv).result
            lt = INDEX
        if rt == I32:
            rv = arith_dialect.IndexCastOp(IndexType.get(), rv).result
            rt = INDEX
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

    # -- subscript index list ------------------------------------------

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
            elif t == I32:
                v = arith_dialect.IndexCastOp(IndexType.get(), v).result
                t = INDEX
            out.append(v)
        return out

    # -- compile-time expression evaluator -----------------------------

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
        if t == I32:
            return arith_dialect.IndexCastOp(IndexType.get(), v).result
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
