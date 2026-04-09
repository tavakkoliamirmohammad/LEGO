"""
SymPy lowering pipeline: SymPy layout expressions → MLIR LEGO dialect → arith → SymPy.
"""
from lego.core import *
import sys
import threading
import sympy as sp

from lego.mlir.dialects import arith, scf
from lego.mlir import ir
from lego.mlir.ir import Context, Location, Module, InsertionPoint
from lego.mlir.passmanager import PassManager as _PassManager
from lego.mlir.dialects import func as _func_dialect
from lego.backend.dialects.lego_dialect import (
    register as _register_lego,
    ApplyOp, ApplyInverseOp, GenPOp, YieldOp,
)
from lego.backend._ops import (
    _LEGO_DEBUG,
    _index_const, _lego_layout_type,
    _emit_reg_p, _emit_row, _emit_col, _emit_order_by,
    _emit_group_by, _emit_tile_by,
    _emit_apply, _emit_apply_inverse, _emit_assume_bounds,
)


# Thread-local cache for MLIR Context with LEGO dialect pre-registered.
_thread_local = threading.local()

def _get_cached_context():
    """Return a thread-local MLIR Context with the LEGO dialect registered."""
    ctx = getattr(_thread_local, 'mlir_ctx', None)
    if ctx is None:
        ctx = Context()
        _register_lego(ctx)
        _thread_local.mlir_ctx = ctx
    return ctx


def _resolve_dim(dim, sym_to_val):
    """Resolve a dimension (int or SymPy expr) to an MLIR Value."""
    if isinstance(dim, int):
        return _index_const(dim)
    if isinstance(dim, sp.Integer):
        return _index_const(int(dim))
    if isinstance(dim, sp.Symbol):
        if dim in sym_to_val:
            return sym_to_val[dim]
        raise KeyError(f"Symbol {dim} not in sym_to_val mapping")
    if isinstance(dim, sp.Expr):
        return _lower_sympy_to_index(dim, sym_to_val)
    raise TypeError(f"Cannot resolve dim of type {type(dim)}: {dim}")


def _lower_sympy_cond_to_i1(cond, sym_to_val):
    """Lower a SymPy relational to an MLIR i1 value."""
    if cond is sp.true or cond == True:
        return arith.ConstantOp(ir.IntegerType.get_signless(1),
                                ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 1)).result
    if cond is sp.false or cond == False:
        return arith.ConstantOp(ir.IntegerType.get_signless(1),
                                ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)).result

    _PRED_MAP = {
        sp.StrictGreaterThan: arith.CmpIPredicate.sgt,
        sp.GreaterThan: arith.CmpIPredicate.sge,
        sp.StrictLessThan: arith.CmpIPredicate.slt,
        sp.LessThan: arith.CmpIPredicate.sle,
        sp.Equality: arith.CmpIPredicate.eq,
        sp.Unequality: arith.CmpIPredicate.ne,
    }
    pred = _PRED_MAP.get(type(cond))
    if pred is None:
        raise NotImplementedError(f"Unsupported condition: {cond}")
    lhs = _lower_sympy_to_index(cond.lhs, sym_to_val)
    rhs = _lower_sympy_to_index(cond.rhs, sym_to_val)
    return arith.cmpi(pred, lhs, rhs)


def _lower_sympy_to_index(expr, sym_to_val):
    """Lower a SymPy expression to MLIR index-typed arith ops."""
    if isinstance(expr, sp.Integer) or isinstance(expr, int):
        return _index_const(int(expr))
    if isinstance(expr, sp.Symbol):
        if expr in sym_to_val:
            return sym_to_val[expr]
        raise KeyError(f"Symbol {expr} not found")

    if isinstance(expr, sp.Add):
        # Separate positive and negative terms for cleaner IR.
        # SymPy represents a - b as Add(a, Mul(-1, b)).
        pos_terms = []
        neg_terms = []
        for a in expr.args:
            coeff = a.as_coeff_Mul()[0] if isinstance(a, sp.Mul) else None
            if isinstance(a, sp.Integer) and int(a) < 0:
                neg_terms.append(_index_const(-int(a)))
            elif coeff is not None and coeff.is_Integer and int(coeff) < 0:
                neg_terms.append(_lower_sympy_to_index(-a, sym_to_val))
            else:
                pos_terms.append(_lower_sympy_to_index(a, sym_to_val))

        if not pos_terms:
            # All negative: 0 - sum(neg)
            acc = _index_const(0)
            for v in neg_terms:
                acc = arith.subi(acc, v)
            return acc
        acc = pos_terms[0]
        for v in pos_terms[1:]:
            acc = arith.addi(acc, v)
        for v in neg_terms:
            acc = arith.subi(acc, v)
        return acc

    if isinstance(expr, sp.Mul):
        num, den = expr.as_numer_denom()
        if den != sp.S.One:
            return arith.divsi(_lower_sympy_to_index(num, sym_to_val),
                               _lower_sympy_to_index(den, sym_to_val))
        coeff, rest = expr.as_coeff_Mul()
        if rest == sp.S.One:
            return _index_const(int(coeff))
        vals = [_lower_sympy_to_index(a, sym_to_val) for a in expr.args]
        acc = vals[0]
        for v in vals[1:]:
            acc = arith.muli(acc, v)
        return acc

    if isinstance(expr, sp.floor):
        inner = expr.args[0]
        num, den = inner.as_numer_denom()
        return arith.divsi(_lower_sympy_to_index(num, sym_to_val),
                           _lower_sympy_to_index(den, sym_to_val))

    if isinstance(expr, sp.Mod):
        return arith.remsi(_lower_sympy_to_index(expr.args[0], sym_to_val),
                           _lower_sympy_to_index(expr.args[1], sym_to_val))

    if isinstance(expr, sp.Abs):
        inner = _lower_sympy_to_index(expr.args[0], sym_to_val)
        zero = _index_const(0)
        neg = arith.subi(zero, inner)
        is_nonneg = arith.cmpi(arith.CmpIPredicate.sge, inner, zero)
        return arith.select(is_nonneg, inner, neg)

    if isinstance(expr, sp.ceiling):
        # ceiling(a/b) = (a + b - 1) / b  (for positive integers)
        inner = expr.args[0]
        num, den = inner.as_numer_denom()
        if den != sp.S.One:
            n = _lower_sympy_to_index(num, sym_to_val)
            d = _lower_sympy_to_index(den, sym_to_val)
            one = _index_const(1)
            return arith.divui(arith.addi(arith.subi(n, one), d), d)
        # If no denominator, ceiling of integer is itself
        return _lower_sympy_to_index(inner, sym_to_val)

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == sp.S.NegativeOne:
            raise NotImplementedError(f"Negative power in index expr: {expr}")
        if exp.is_Integer and int(exp) >= 0:
            b = _lower_sympy_to_index(base, sym_to_val)
            result = _index_const(1)
            for _ in range(int(exp)):
                result = arith.muli(result, b)
            return result

    if expr.func == sp.Max:
        a = _lower_sympy_to_index(expr.args[0], sym_to_val)
        b = _lower_sympy_to_index(expr.args[1], sym_to_val)
        return arith.select(arith.cmpi(arith.CmpIPredicate.sge, a, b), a, b)

    if expr.func == sp.Min:
        a = _lower_sympy_to_index(expr.args[0], sym_to_val)
        b = _lower_sympy_to_index(expr.args[1], sym_to_val)
        return arith.select(arith.cmpi(arith.CmpIPredicate.sle, a, b), a, b)

    if isinstance(expr, sp.core.relational.Relational):
        return _lower_sympy_cond_to_i1(expr, sym_to_val)

    if isinstance(expr, sp.Piecewise):
        from lego.mlir.dialects import scf as _scf
        idx_ty = ir.IndexType.get()

        def _recurse(i):
            val_expr, cond_expr = expr.args[i]
            then_val = _lower_sympy_to_index(val_expr, sym_to_val)
            if i == len(expr.args) - 1:
                return then_val
            cond_val = _lower_sympy_cond_to_i1(cond_expr, sym_to_val)
            ifop = _scf.IfOp(cond_val, [idx_ty], has_else=True)
            with InsertionPoint(ifop.then_block):
                _scf.YieldOp([then_val])
            with InsertionPoint(ifop.else_block):
                _scf.YieldOp([_recurse(i + 1)])
            return ifop.results[0]

        return _recurse(0)

    raise NotImplementedError(f"Cannot lower SymPy expr to index: {expr}")


def emit_layout_from_python(layout, sym_to_val):
    """Convert a Python layout object to MLIR LEGO dialect ops."""
    from lego.mlir.ir import IndexType

    if isinstance(layout, Row):
        return _emit_row([_resolve_dim(d, sym_to_val) for d in layout._dims])

    if isinstance(layout, Col):
        return _emit_col([_resolve_dim(d, sym_to_val) for d in layout._dims])

    if isinstance(layout, RegP):
        return _emit_reg_p(layout._perm_vector,
                           [_resolve_dim(d, sym_to_val) for d in layout._dims])

    if isinstance(layout, OrderBy):
        perm_vals = [emit_layout_from_python(p, sym_to_val) for p in layout.perms]
        return _emit_order_by(perm_vals)

    if isinstance(layout, TileByLayout):
        all_perm_vals = []
        for orderby in layout._input_chain:
            for p in orderby.perms:
                all_perm_vals.append(emit_layout_from_python(p, sym_to_val))
        input_val = _emit_order_by(all_perm_vals)
        tile_dim_vals = [_resolve_dim(d, sym_to_val)
                         for g in layout._tile_groups for d in g]
        return _emit_tile_by(input_val, tile_dim_vals, layout.tile_shape)

    if isinstance(layout, GroupBy):
        dim_vals = [_resolve_dim(d, sym_to_val) for d in layout._dims]
        obj_vals = [emit_layout_from_python(obj, sym_to_val) for obj in layout.objects]
        return _emit_group_by(dim_vals, obj_vals)

    if isinstance(layout, GenP):
        dim_vals = [_resolve_dim(d, sym_to_val) for d in layout._dims]
        idx_ty = IndexType.get()
        lt = _lego_layout_type()
        gen_p_op = GenPOp(result=lt, dims=dim_vals)

        rank = len(layout._dims)
        apply_block = gen_p_op.body.blocks.append(*([idx_ty] * rank))
        with InsertionPoint(apply_block):
            temp_syms = [sp.Symbol(f"_genp_arg_{k}", integer=True) for k in range(rank)]
            local_map = dict(sym_to_val)
            for s, arg in zip(temp_syms, apply_block.arguments):
                local_map[s] = arg
            YieldOp(values=[_lower_sympy_to_index(layout.f_apply(tuple(temp_syms)), local_map)])

        if layout.f_inv is not None:
            inv_block = gen_p_op.inv_body.blocks.append(idx_ty)
            with InsertionPoint(inv_block):
                temp_flat = sp.Symbol("_genp_flat", integer=True)
                local_map = dict(sym_to_val)
                local_map[temp_flat] = inv_block.arguments[0]
                inv_results = [_lower_sympy_to_index(r, local_map)
                               for r in layout.f_inv(temp_flat)]
                YieldOp(values=inv_results)

        return gen_p_op.result

    raise TypeError(f"Unsupported layout type: {type(layout).__name__}")


def arith_to_sympy(value, val_to_sym, memo=None):
    """Convert an MLIR Value (arith ops after lowering) back to SymPy."""
    if memo is None:
        memo = {}
    if value in memo:
        return memo[value]
    if value in val_to_sym:
        result = val_to_sym[value]
        memo[value] = result
        return result
    if isinstance(value.owner, ir.Block):
        raise KeyError(f"Block argument not found in val_to_sym mapping")

    op = value.owner
    op_name = op.name

    if op_name == "arith.constant":
        result = sp.Integer(int(ir.IntegerAttr(op.attributes["value"]).value))
        memo[value] = result
        return result

    _BINARY_OPS = {
        "arith.addi": lambda a, b: a + b,
        "arith.muli": lambda a, b: a * b,
        "arith.subi": lambda a, b: a - b,
        "arith.divui": lambda a, b: sp.floor(a / b),
        "arith.divsi": lambda a, b: sp.floor(a / b),
        "arith.remui": lambda a, b: sp.Mod(a, b),
        "arith.remsi": lambda a, b: sp.Mod(a, b),
        "arith.maxui": lambda a, b: sp.Max(a, b),
        "arith.maxsi": lambda a, b: sp.Max(a, b),
        "arith.minui": lambda a, b: sp.Min(a, b),
        "arith.minsi": lambda a, b: sp.Min(a, b),
        "arith.shli": lambda a, b: a * sp.Pow(2, b),
        "arith.shrui": lambda a, b: sp.floor(a / sp.Pow(2, b)),
        "arith.shrsi": lambda a, b: sp.floor(a / sp.Pow(2, b)),
        "arith.andi": lambda a, b: sp.Mod(a, b + 1) if (isinstance(b, sp.Integer) and ((int(b) + 1) & int(b)) == 0) else sp.Function('bitand')(a, b),
        "arith.ori": lambda a, b: sp.Function('bitor')(a, b),
    }
    if op_name in _BINARY_OPS:
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        result = _BINARY_OPS[op_name](a, b)
        memo[value] = result
        return result

    if op_name == "arith.cmpi":
        pred = int(ir.IntegerAttr(op.attributes["predicate"]).value)
        a = arith_to_sympy(op.operands[0], val_to_sym, memo)
        b = arith_to_sympy(op.operands[1], val_to_sym, memo)
        pred_map = {
            0: sp.Eq, 1: sp.Ne,
            2: sp.StrictLessThan, 3: sp.LessThan,
            4: sp.StrictGreaterThan, 5: sp.GreaterThan,
            6: sp.StrictLessThan, 7: sp.LessThan,
            8: sp.StrictGreaterThan, 9: sp.GreaterThan,
        }
        rel_cls = pred_map.get(pred)
        if rel_cls:
            result = rel_cls(a, b, evaluate=False)
            memo[value] = result
            return result
        raise NotImplementedError(f"Unknown cmpi predicate: {pred}")

    if op_name == "arith.select":
        true_val = arith_to_sympy(op.operands[1], val_to_sym, memo)
        false_val = arith_to_sympy(op.operands[2], val_to_sym, memo)
        cond_op = op.operands[0].owner
        if cond_op.name == "arith.cmpi":
            pred = int(ir.IntegerAttr(cond_op.attributes["predicate"]).value)
            cmp_lhs = arith_to_sympy(cond_op.operands[0], val_to_sym, memo)
            cmp_rhs = arith_to_sympy(cond_op.operands[1], val_to_sym, memo)

            # Recognize Abs pattern: select(x >= 0, x, -x) → Abs(x)
            if (pred in (5, 9)  # sge or uge
                    and cmp_rhs == sp.Integer(0)
                    and true_val == cmp_lhs
                    and false_val == -cmp_lhs):
                result = sp.Abs(cmp_lhs)
                memo[value] = result
                return result

            select_patterns = {
                (5, True): sp.Max, (5, False): sp.Min,
                (3, True): sp.Min, (3, False): sp.Max,
            }
            if true_val == cmp_lhs and false_val == cmp_rhs:
                key = (pred, True)
            elif true_val == cmp_rhs and false_val == cmp_lhs:
                key = (pred, False)
            else:
                key = None
            if key and key in select_patterns:
                result = select_patterns[key](cmp_lhs, cmp_rhs)
                memo[value] = result
                return result
        cond = arith_to_sympy(op.operands[0], val_to_sym, memo)
        result = sp.Piecewise((true_val, cond), (false_val, True))
        memo[value] = result
        return result

    if op_name == "scf.if":
        cond = arith_to_sympy(op.operands[0], val_to_sym, memo)
        then_yield = list(op.regions[0].blocks[0])[-1]
        else_yield = list(op.regions[1].blocks[0])[-1]
        then_val = arith_to_sympy(then_yield.operands[0], val_to_sym, memo)
        else_val = arith_to_sympy(else_yield.operands[0], val_to_sym, memo)
        result = sp.Piecewise((then_val, cond), (else_val, True))
        memo[value] = result
        return result

    raise NotImplementedError(f"Cannot convert op '{op_name}' to SymPy")


def _collect_free_symbols(layout):
    """Collect all SymPy symbols used in a layout's dimensions and GenP bodies (recursive)."""
    syms = set()
    if hasattr(layout, '_dims'):
        for d in layout._dims:
            if isinstance(d, sp.Expr):
                syms |= d.free_symbols
            elif isinstance(d, sp.Symbol):
                syms.add(d)
    if isinstance(layout, GenP):
        # Evaluate f_apply with dummy index symbols to discover extra free symbols
        # (e.g., symbolic parameters used in the forward function body).
        try:
            rank = len(layout._dims)
            dummy_idx = sp.symbols([f"_dummy_idx_{k}" for k in range(rank)], integer=True)
            fwd_expr = layout.f_apply(tuple(dummy_idx))
            if isinstance(fwd_expr, sp.Expr):
                syms |= fwd_expr.free_symbols - set(dummy_idx)
            if layout.f_inv is not None:
                dummy_flat = sp.Symbol("_dummy_flat", integer=True)
                inv_result = layout.f_inv(dummy_flat)
                if inv_result is not None:
                    for expr in inv_result:
                        if isinstance(expr, sp.Expr):
                            syms |= expr.free_symbols - {dummy_flat}
        except Exception:
            pass  # Best-effort; fall back to dim-only collection
    if isinstance(layout, TileByLayout):
        for orderby in layout._input_chain:
            syms |= _collect_free_symbols(orderby)
        for g in layout._tile_groups:
            for d in g:
                if isinstance(d, sp.Expr):
                    syms |= d.free_symbols
                elif isinstance(d, sp.Symbol):
                    syms.add(d)
    elif isinstance(layout, OrderBy):
        for p in layout.perms:
            syms |= _collect_free_symbols(p)
    elif isinstance(layout, GroupBy):
        for obj in layout.objects:
            syms |= _collect_free_symbols(obj)
    return syms


def simplify_via_mlir(layout, mode, args, constraints=None):
    """Compute layout.apply or layout.inv via MLIR roundtrip."""
    from lego.mlir.ir import IndexType, FunctionType, StringAttr

    if constraints is None:
        constraints = {}

    all_syms = set()
    all_syms |= _collect_free_symbols(layout)

    if mode == 'apply':
        for a in args:
            if isinstance(a, sp.Expr):
                all_syms |= a.free_symbols
            elif isinstance(a, sp.Symbol):
                all_syms.add(a)
    else:
        if isinstance(args, sp.Expr):
            all_syms |= args.free_symbols
        elif isinstance(args, sp.Symbol):
            all_syms.add(args)

    for sym in constraints:
        if isinstance(sym, sp.Symbol):
            all_syms.add(sym)
        lb, ub = constraints[sym]
        if isinstance(lb, sp.Expr):
            all_syms |= lb.free_symbols
        if isinstance(ub, sp.Expr):
            all_syms |= ub.free_symbols

    sym_list = sorted(all_syms, key=lambda s: s.name)

    ctx = _get_cached_context()

    try:
        return _simplify_via_mlir_impl(ctx, layout, mode, args, constraints, sym_list)
    except Exception:
        # Invalidate cached context on failure to avoid stale state
        _thread_local.mlir_ctx = None
        raise


def _simplify_via_mlir_impl(ctx, layout, mode, args, constraints, sym_list):
    """Inner implementation of simplify_via_mlir (separated for error handling)."""
    from lego.mlir.ir import IndexType, FunctionType, StringAttr

    with ctx, Location.unknown():
        module = Module.create()
        idx_ty = IndexType.get()

        n_args = len(sym_list)
        rank = len(layout._dims)
        if mode == 'apply':
            func_ty = FunctionType.get([idx_ty] * n_args, [idx_ty])
        else:
            func_ty = FunctionType.get([idx_ty] * n_args, [idx_ty] * rank)

        with InsertionPoint(module.body):
            f = _func_dialect.FuncOp("roundtrip", func_ty)
            f.sym_visibility = StringAttr.get("public")

        entry = f.add_entry_block()

        sym_to_val = {}
        val_to_sym = {}
        for i, sym in enumerate(sym_list):
            sym_to_val[sym] = entry.arguments[i]
            val_to_sym[entry.arguments[i]] = sym

        with InsertionPoint(entry):
            for sym, (lb, ub) in constraints.items():
                if not isinstance(sym, sp.Symbol) or sym not in sym_to_val:
                    continue
                lb_val = _resolve_dim(lb, sym_to_val) if lb is not None else None
                ub_val = _resolve_dim(ub, sym_to_val) if ub is not None else None
                _emit_assume_bounds(sym_to_val[sym], lb=lb_val, ub=ub_val)

            layout_val = emit_layout_from_python(layout, sym_to_val)

            if mode == 'apply':
                arg_vals = [_resolve_dim(a, sym_to_val) for a in args]
                result = _emit_apply(layout_val, arg_vals)
                _func_dialect.ReturnOp([result])
            else:
                flat_val = _resolve_dim(args, sym_to_val)
                results = _emit_apply_inverse(layout_val, flat_val, rank)
                _func_dialect.ReturnOp(results)

        if _LEGO_DEBUG:
            print("=== MLIR input (LEGO dialect) ===", file=sys.stderr)
            print(module, file=sys.stderr)
            print(file=sys.stderr)
            module_copy = Module.parse(str(module))
            pm_pre = _PassManager.parse("builtin.module(canonicalize,cse)")
            pm_pre.run(module_copy.operation)
            print("=== MLIR after canonicalize + CSE ===", file=sys.stderr)
            print(module_copy, file=sys.stderr)
            print(file=sys.stderr)

        pm = _PassManager.parse("builtin.module(lego-lower)")
        pm.run(module.operation)

        if _LEGO_DEBUG:
            print("=== MLIR after lego-lower ===", file=sys.stderr)
            print(module, file=sys.stderr)
            print(file=sys.stderr)

        func_op = None
        for op in module.body:
            if op.name == "roundtrip" or op.name == "func.func":
                func_op = op
                break
        if func_op is None:
            for op in module.body:
                func_op = op
                break

        entry_block = func_op.regions[0].blocks[0]
        val_to_sym_post = {entry_block.arguments[i]: sym for i, sym in enumerate(sym_list)}

        return_op = None
        for op in entry_block:
            if op.name == "func.return":
                return_op = op
                break

        results_sympy = [arith_to_sympy(operand, val_to_sym_post)
                         for operand in return_op.operands]

        return results_sympy[0] if mode == 'apply' else results_sympy
