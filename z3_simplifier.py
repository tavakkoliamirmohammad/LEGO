from z3 import Solver, sat, unsat, Int, Bool, Real, Not as Z3Not, IntVal, RealVal, BoolVal
import sympy as sp
from sympy import Integer, Mod

# --- Compile wildcards once for reuse ---
_wd_int = sp.Wild('d', exclude=[0], properties=[lambda x: x.is_integer])
_wq = sp.Wild('q')
_wr_int = sp.Wild('r', properties=[lambda x: x.is_integer])
_wa = sp.Wild('a', exclude=[0, Mod, sp.floor])
_wx_int = sp.Wild('x', properties=[lambda x: x.is_integer])
_wa_int = sp.Wild('a', exclude=[0, Mod, sp.floor],
                  properties=[lambda x: x.is_integer])
_wx = sp.Wild('x')
_wd = sp.Wild('d', properties=[lambda x: x.is_integer], exclude=[0])
_wq2 = sp.Wild('q', properties=[lambda x: x.is_integer])
_wr = sp.Wild('r')

class ProverContext:
    __slots__ = ("solver", "z3_vars", "_sym2z3_cache",
                 "_result_cache", "_var_types")

    def __init__(self, constraints, var_types=None, timeout=100):
        self._var_types = var_types or {}

        # collect symbols found in base constraints
        syms = set()
        for c in constraints:
            syms |= getattr(c, "free_symbols", set())

        self.z3_vars = build_z3_vars(syms, self._var_types)
        self._sym2z3_cache = {}
        self._result_cache = {}

        self.solver = Solver()
        self.solver.set("timeout", timeout)
        base = [self._to_z3(c) for c in constraints]
        if base:
            self.solver.add(*base)

    # --- NEW: create Z3 vars on demand for any new symbols we see
    def _mk_z3_var(self, s):
        name = str(s)
        t = self._var_types.get(name) if name in self._var_types else (
            'Int' if getattr(s, 'is_integer', False) else
            'Bool' if getattr(s, 'is_boolean', False) else
            'Int'   # fallback favors Int (safer for Mod/floor arithmetic)
        )
        if t == 'Int':
            return Int(name)
        if t == 'Bool':
            return Bool(name)
        return Real(name)

    def _ensure_vars(self, expr):
        for s in getattr(expr, "free_symbols", set()):
            name = str(s)
            if name not in self.z3_vars:
                self.z3_vars[name] = self._mk_z3_var(s)

    def _to_z3(self, expr):
        # <-- ensure presence before translate
        self._ensure_vars(expr)
        z = self._sym2z3_cache.get(expr)
        if z is None:
            z = sympy_to_z3(expr, self.z3_vars)
            self._sym2z3_cache[expr] = z
        return z

    def can_prove(self, ineq, debug=False):
        hit = self._result_cache.get(ineq, None)
        if hit is not None:
            return hit

        if ineq is True or ineq == sp.true:
            self._result_cache[ineq] = True
            return True
        if ineq is False or ineq == sp.false:
            self._result_cache[ineq] = False
            return False
        if not getattr(ineq, 'free_symbols', None):
            try:
                v = bool(ineq)
                self._result_cache[ineq] = v
                return v
            except Exception:
                pass

        self.solver.push()
        self.solver.add(Z3Not(self._to_z3(ineq)))  # _to_z3 now auto-adds vars
        res = self.solver.check()
        self.solver.pop()

        if debug:
            print("Z3 result:", res)

        if res == unsat:
            self._result_cache[ineq] = True
            return True
        if res == sat:
            self._result_cache[ineq] = False
            return False

        self._result_cache[ineq] = False
        return False


def build_z3_vars(sympy_syms, var_types=None):
    z3_vars = {}
    for s in sympy_syms:
        name = str(s)
        t = var_types.get(name) if var_types and name in var_types else (
            'Int' if getattr(s, 'is_integer', False) else
            'Bool' if getattr(s, 'is_boolean', False) else
            'Real'
        )
        if t == 'Int':
            z3_vars[name] = Int(name)
        elif t == 'Bool':
            z3_vars[name] = Bool(name)
        else:
            z3_vars[name] = Int(name)
    return z3_vars

def sympy_to_z3(expr, z3_vars):
    from sympy import Add, Mul, Pow, Mod as SymMod, floor as SymFloor
    from sympy.logic.boolalg import BooleanTrue, BooleanFalse
    from sympy import true, false, And as SAnd, Or as SOr, Not as SNot
    from sympy.core.relational import Relational
    from z3 import And as Z3And, Or as Z3Or, Not as Z3Not, If as Z3If

    # --- Boolean constants ---
    if expr is True or expr == true or isinstance(expr, BooleanTrue):
        return BoolVal(True)
    if expr is False or expr == false or isinstance(expr, BooleanFalse):
        return BoolVal(False)

    # --- Function handlers ---
    if expr.func == SymMod:
        a, b = expr.args
        return sympy_to_z3(a, z3_vars) % sympy_to_z3(b, z3_vars)
    if expr.func == sp.Min:
        a, b = expr.args
        za = sympy_to_z3(a, z3_vars)
        zb = sympy_to_z3(b, z3_vars)
        return Z3If(za <= zb, za, zb)
    if expr.func == sp.Max:
        a, b = expr.args
        za = sympy_to_z3(a, z3_vars)
        zb = sympy_to_z3(b, z3_vars)
        return Z3If(za >= zb, za, zb)
    if expr.func is SymFloor:
        numer, denom = expr.args[0].as_numer_denom()
        return sympy_to_z3(numer, z3_vars) / sympy_to_z3(denom, z3_vars)

    # --- Symbol and constants ---
    if expr.is_Symbol:
        return z3_vars[str(expr)]
    if expr.is_Integer:
        return IntVal(int(expr))
    if expr.is_Rational or expr.is_Float:
        return RealVal(str(expr))

    # --- Arithmetic ---
    if isinstance(expr, Add):
        return sum(sympy_to_z3(a, z3_vars) for a in expr.args)
    if isinstance(expr, Mul):
        invs = [arg for arg in expr.args if isinstance(
            arg, Pow) and arg.args[1].is_Integer and int(arg.args[1]) < 0]
        normals = [arg for arg in expr.args if arg not in invs]
        if invs:
            numer = normals[0] if len(normals) == 1 else sp.Mul(*normals)
            z_numer = sympy_to_z3(numer, z3_vars)
            z_denom = sympy_to_z3(invs[0].args[0], z3_vars)
            for d in invs[1:]:
                z_denom = z_denom * sympy_to_z3(d.args[0], z3_vars)
            return z_numer / z_denom
        z_args = [sympy_to_z3(a, z3_vars) for a in expr.args]
        out = z_args[0]
        for a in z_args[1:]:
            out = out * a
        return out
    if isinstance(expr, Pow):
        b, e = expr.args
        if e.is_integer:
            exp_n = int(e)
            zbase = sympy_to_z3(b, z3_vars)
            if exp_n == 0:
                return IntVal(1)
            if exp_n > 0:
                res = zbase
                for _ in range(exp_n-1):
                    res = res * zbase
                return res
            else:  # exp_n < 0
                res = zbase
                for _ in range(-exp_n-1):
                    res = res * zbase
                return IntVal(1) / res
        else:
            raise NotImplementedError(f"unexpected pow {expr}")

    # --- Relations ---
    if isinstance(expr, Relational):
        lhs = sympy_to_z3(expr.lhs, z3_vars)
        rhs = sympy_to_z3(expr.rhs, z3_vars)
        name = type(expr).__name__
        if name == 'Equality':
            return lhs == rhs
        if name == 'GreaterThan':
            return lhs >= rhs
        if name == 'StrictGreaterThan':
            return lhs > rhs
        if name == 'LessThan':
            return lhs <= rhs
        if name == 'StrictLessThan':
            return lhs < rhs

    # --- Boolean ops ---
    if expr.func == SNot:
        return Z3Not(sympy_to_z3(expr.args[0], z3_vars))
    if expr.func == SAnd:
        return ZAnd(*(sympy_to_z3(a, z3_vars) for a in expr.args))
    if expr.func == SOr:
        return ZOr(*(sympy_to_z3(a, z3_vars) for a in expr.args))

    raise NotImplementedError(f"Cannot convert {expr!r} to Z3.")

# --- Normalization: expand numerators in Mod and floor ---


def _normalize(expr):
    # Expand the first argument of Mod
    if isinstance(expr, Mod):
        a, d = expr.args
        return Mod(sp.expand(a), sp.expand(d))

    # Expand + peel multiples inside floor(x/y)
    if expr.func is sp.floor and expr.args:
        x = expr.args[0]
        num, den = sp.together(x).as_numer_denom()
        num = sp.expand(num)
        den = sp.expand(den)

        # --- peel exact multiples of the denominator from the numerator sum
        if isinstance(num, sp.Add):
            k_int = sp.Wild('k_int', properties=[lambda z: z.is_integer])
            multiples, rest_terms = [], []
            for t in num.args:
                m = t.match(k_int*den) or t.match(den*k_int)
                if m and m.get(k_int) is not None:
                    multiples.append(m[k_int])
                else:
                    rest_terms.append(t)
            if multiples:
                outside = sp.Add(
                    *multiples) if len(multiples) > 1 else multiples[0]
                rest = sp.Add(*rest_terms) if rest_terms else sp.Integer(0)
                # outside is integer, so floor(outside + rest/den) == outside + floor(rest/den)
                return outside + sp.floor(rest/den)

        # Default normalization: just keep floor(expanded_num / expanded_den)
        return sp.floor(num / den)
    return expr


def simplify_int_div_mod(expr: sp.Expr, constraints, bounds=None, prover: ProverContext = None):
    if prover is None:
        # safe default, but simplify_ops will pass a shared one for speed
        prover = ProverContext(constraints)

    # Normalize so patterns see sums/products plainly
    expr = _normalize(expr)
    if expr.is_Atom:
        return expr

    # Recurse first
    if expr.args:
        expr = expr.func(
            *(simplify_int_div_mod(arg, constraints, bounds, prover) for arg in expr.args))
        expr = _normalize(expr)
    if expr.is_Atom:
        return expr

    # print("Simplifying:", expr)

    # print("Visiting:", expr)
    # ----------------------------------------
    # (A) Local algebraic rules (single-node)
    # ----------------------------------------

    # A1: Mod(d*q + r, d) -> Mod(r, d)
    m = expr.match(Mod(_wd_int*_wq + _wr_int, _wd_int))
    if m:
        return Mod(m[_wr_int], m[_wd_int])
    # A1b: Commutative twin Mod(r + d*q, d) -> Mod(r, d)
    m = expr.match(Mod(_wr_int + _wd_int*_wq, _wd_int))
    if m:
        return Mod(m[_wr_int], m[_wd_int])

    # A2: floor((d*q + r)/d) -> q (if 0<=r<d), else q + floor(r/d)
    m = expr.match(sp.floor((_wd_int*_wq2 + _wr)/_wd_int))
    if m:
        d_expr, q_expr, r_expr = m[_wd_int], m[_wq2], m[_wr]
        if prover.can_prove(r_expr >= 0) and prover.can_prove(r_expr < d_expr):
            return q_expr
        return q_expr + sp.floor(r_expr/d_expr)

    # A3: floor(Mod(x, d)/d) -> 0 when d > 0
    m = expr.match(sp.floor(Mod(_wx, _wd_int)/_wd_int))
    if m and prover.can_prove(m[_wd_int] > 0):
        return sp.Integer(0)

    # A4: floor(x/a) -> 0 if 0 <= x < a
    m = expr.match(sp.floor(_wx/_wa_int))
    if m:
        x_expr, a_expr = m[_wx], m[_wa_int]
        if prover.can_prove(x_expr >= 0) and prover.can_prove(x_expr < a_expr):
            return sp.Integer(0)

    # A5: Mod(x, a) -> x if 0 <= x < a
    if isinstance(expr, Mod):
        x_expr, a_expr = expr.args
        if prover.can_prove(a_expr > 0) and prover.can_prove(x_expr >= 0) and prover.can_prove(x_expr < a_expr):
            return x_expr

    # A6 (optional): floor(n + y) -> n + floor(y) for integer n
    m = expr.match(sp.floor(_wx_int + _wx))
    if m:
        return m[_wx_int] + sp.floor(m[_wx])
    m = expr.match(sp.floor(_wx + _wx_int))
    if m:
        return m[_wx_int] + sp.floor(m[_wx])

    # ----------------------------------------
    # (B) Pairwise combine inside Add nodes
    #     ... + a*floor(x/a) + Mod(x,a) + ... -> ... + x + ...
    # ----------------------------------------
    if isinstance(expr, sp.Add):
        args = list(expr.args)
        n = len(args)
        for i, t1 in enumerate(args):
            m1 = t1.match(_wa_int*sp.floor(_wx/_wa_int))
            if not m1:
                continue
            a, x = m1[_wa_int], m1[_wx]
            for j, t2 in enumerate(args):
                if j == i:
                    continue
                if t2 == Mod(x, a):
                    new_args = [args[k] for k in range(n) if k not in (i, j)]
                    new_args.append(x)
                    return simplify_int_div_mod(sp.Add(*new_args), constraints, bounds, prover)

    # ----------------------------------------
    # (C) Local fold (single-node)
    #     a*floor(x/a) + Mod(x, a) -> x
    # ----------------------------------------
    m = expr.match(_wa_int*sp.floor(_wx/_wa_int) + Mod(_wx, _wa_int))
    if m:
        a, x = m[_wa_int], m[_wx]
        if prover.can_prove(a != 0):
            return x

    return expr


def simplify_ops(expr, bounds, additional_bounds=None, var_types=None, timeout=100):
    if additional_bounds is None:
        additional_bounds = []

    # Build constraints once
    constraints = [sp.Ge(sym, lb, evaluate=False)
                   for sym, (lb, _) in bounds.items()]
    constraints += [sp.Lt(sym, ub, evaluate=False)
                    for sym, (_, ub) in bounds.items()]
    constraints += additional_bounds

    # Reuse one solver + caches for the whole fixpoint loop
    prover = ProverContext(constraints, var_types=var_types, timeout=timeout)

    cur = expr
    while True:
        nxt = simplify_int_div_mod(cur, constraints, bounds, prover=prover)
        if nxt == cur:
            return cur
        cur = nxt
