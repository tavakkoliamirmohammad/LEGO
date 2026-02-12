from z3 import Solver, sat, unsat, Int, Bool, Real, Not as Z3Not, IntVal, RealVal, BoolVal
import sympy as sp
from sympy import Integer, Mod
from functools import lru_cache

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
_zero = sp.Integer(0)
_one = sp.Integer(1)

class FastProverContext:
    def __init__(self, constraints, timeout=100):
        self.constraints = constraints
        self.bounds = {}  # Symbol -> {'num': [min, max], 'sym': [min, max]}
        self._process_constraints()
        # Keep referencing the Z3 solver but lazy init
        self.z3_prover = None
        self.timeout = timeout
        self._result_cache = {}
        
    def _process_constraints(self):
        # Pass 1: Extract numeric and simple symbolic bounds
        for c in self.constraints:
            if isinstance(c, sp.Ge): # x >= y
                self._update_ge(c.args[0], c.args[1])
            elif isinstance(c, sp.Gt): # x > y -> x >= y + 1 (integers)
                self._update_ge(c.args[0], c.args[1] + 1)
            elif isinstance(c, sp.Le): # x <= y
                self._update_le(c.args[0], c.args[1])
            elif isinstance(c, sp.Lt): # x < y -> x <= y - 1
                self._update_le(c.args[0], c.args[1] - 1)
            elif isinstance(c, sp.Eq):
                 self._update_ge(c.args[0], c.args[1])
                 self._update_le(c.args[0], c.args[1])

    def _update_ge(self, lhs, rhs):
        # lhs >= rhs
        if lhs.is_Symbol:
            self._update_bound_entry(lhs, 'min', rhs)
        if rhs.is_Symbol:
             # y <= x
             self._update_bound_entry(rhs, 'max', lhs)

    def _update_le(self, lhs, rhs):
        # lhs <= rhs
        if lhs.is_Symbol:
             self._update_bound_entry(lhs, 'max', rhs)
        if rhs.is_Symbol:
             # y >= x
             self._update_bound_entry(rhs, 'min', lhs)

    def _update_bound_entry(self, sym, kind, val):
        if sym not in self.bounds:
            self.bounds[sym] = {'min_num': None, 'max_num': None, 'min_sym': [], 'max_sym': []}
        
        b = self.bounds[sym]
        if val.is_number:
            val_n = val
            if kind == 'min':
                if b['min_num'] is None or val_n > b['min_num']:
                    b['min_num'] = val_n
            else:
                if b['max_num'] is None or val_n < b['max_num']:
                    b['max_num'] = val_n
        else:
             # Store symbolic bound
             if kind == 'min':
                 b['min_sym'].append(val)
             else:
                 b['max_sym'].append(val)

    def get_bounds(self, expr):
        # Return estimated [min, max] for expr
        # This is a recursive estimation
        if isinstance(expr, (int, float, sp.Integer)):
            return expr, expr
        if isinstance(expr, sp.Symbol):
            if expr in self.bounds:
                b = self.bounds[expr]
                # Try to find best numeric bound
                lower = b['min_num'] if b['min_num'] is not None else float('-inf')
                upper = b['max_num'] if b['max_num'] is not None else float('inf')
                return lower, upper
            return float('-inf'), float('inf')
            
        if isinstance(expr, sp.Add):
            min_v, max_v = 0, 0
            for arg in expr.args:
                mn, mx = self.get_bounds(arg)
                if min_v != float('-inf'): min_v = min_v + mn if mn != float('-inf') else float('-inf')
                if max_v != float('inf'): max_v = max_v + mx if mx != float('inf') else float('inf')
            return min_v, max_v
            
        if isinstance(expr, sp.Mod):
             # Mod(a, b). If b is constant positive, range is [0, b-1] (integers)
             # or [0, b) reals.
             # We mainly deal with integers here.
             p, q = expr.args
             if q.is_integer and q.is_number and q > 0:
                 return 0, q - 1
             # If q is symbolic, maybe we can't do much unless we know q > 0
             
        if isinstance(expr, sp.floor):
             mn, mx = self.get_bounds(expr.args[0])
             import math
             l = math.floor(mn) if mn != float('-inf') else float('-inf')
             r = math.floor(mx) if mx != float('inf') else float('inf')
             return l, r

        if isinstance(expr, sp.Mul):
            # perform simple interval mul for positive constants
            c = 1
            rest = []
            for arg in expr.args:
                if arg.is_number: c *= arg
                else: rest.append(arg)
            
            if not rest: return c, c
            if len(rest) == 1:
                mn, mx = self.get_bounds(rest[0])
                if c >= 0:
                     return (mn * c if mn != float('-inf') else float('-inf')), \
                            (mx * c if mx != float('inf') else float('inf'))
                else:
                     return (mx * c if mx != float('inf') else float('-inf')), \
                            (mn * c if mn != float('-inf') else float('inf'))
                            
        return float('-inf'), float('inf')

    def can_prove(self, ineq, debug=False):
        # 0. Check cache
        if ineq in self._result_cache:
            return self._result_cache[ineq]

        # 1. Trivial check
        if ineq is True or ineq == sp.true: 
            self._result_cache[ineq] = True
            return True
        if ineq is False or ineq == sp.false: 
            self._result_cache[ineq] = False
            return False
        
        # 2. Try simple difference check (much faster than simplify)
        if isinstance(ineq, (sp.Ge, sp.Gt, sp.Le, sp.Lt)):
            lhs, rhs = ineq.args
            diff = lhs - rhs
            # Auto-simplification of Add often handles (n+1) - (n-1) -> 2
            if diff.is_number:
                 if isinstance(ineq, sp.Ge) and diff >= 0: 
                     self._result_cache[ineq] = True
                     return True
                 if isinstance(ineq, sp.Gt) and diff > 0: 
                     self._result_cache[ineq] = True
                     return True
                 if isinstance(ineq, sp.Le) and diff <= 0: 
                     self._result_cache[ineq] = True
                     return True
                 if isinstance(ineq, sp.Lt) and diff < 0: 
                     self._result_cache[ineq] = True
                     return True

        # 3. Fast Interval Check (Numeric)
        if isinstance(ineq, (sp.Ge, sp.Gt, sp.Le, sp.Lt)):
             lhs, rhs = ineq.args
             l_min, l_max = self.get_bounds(lhs)
             r_min, r_max = self.get_bounds(rhs)
             
             if isinstance(ineq, sp.Ge): # l >= r
                 if l_min != float('-inf') and r_max != float('inf') and l_min >= r_max: 
                     self._result_cache[ineq] = True
                     return True
             elif isinstance(ineq, sp.Gt): # l > r
                 if l_min != float('-inf') and r_max != float('inf') and l_min > r_max: 
                     self._result_cache[ineq] = True
                     return True
             elif isinstance(ineq, sp.Le): # l <= r
                 if l_max != float('inf') and r_min != float('-inf') and l_max <= r_min: 
                     self._result_cache[ineq] = True
                     return True
             elif isinstance(ineq, sp.Lt): # l < r
                 if l_max != float('inf') and r_min != float('-inf') and l_max < r_min: 
                     self._result_cache[ineq] = True
                     return True

        # 4. Symbolic Bound Check (Linear scan of constraints)
        # Check if ineq is directly in constraints or implied by simple transitivity
        if ineq in self.constraints: 
            self._result_cache[ineq] = True
            return True
        
        # 5. Fallback to Z3
        if self.z3_prover is None:
             self.z3_prover = ProverContext(self.constraints, timeout=self.timeout)
        
        res = self.z3_prover.can_prove(ineq, debug)
        self._result_cache[ineq] = res
        return res


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


@lru_cache(maxsize=None)
def _normalize(expr):
    # Expand the first argument of Mod
    if isinstance(expr, Mod):
        a, d = expr.args
        a_exp = sp.expand(a) if not a.is_Atom else a
        d_exp = sp.expand(d) if not d.is_Atom else d
        if a_exp == a and d_exp == d:
            return expr
        return Mod(a_exp, d_exp)

    # Expand + peel multiples inside floor(x/y)
    if expr.func is sp.floor and expr.args:
        x = expr.args[0]
        if x.is_Atom:
            return expr
            
        num, den = sp.together(x).as_numer_denom()
        num = sp.expand(num) if not num.is_Atom else num
        den = sp.expand(den) if not den.is_Atom else den

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


def simplify_int_div_mod(expr: sp.Expr, constraints, bounds=None, prover: ProverContext = None, memo=None):
    if prover is None:
        # safe default, but simplify_ops will pass a shared one for speed
        prover = ProverContext(constraints)
    
    if memo is None:
        memo = {}
    
    if expr in memo:
        return memo[expr]

    # Normalize so patterns see sums/products plainly
    # Note: _normalize is cheap enough, or we can cache it too? 
    # Let's rely on memo caching the result of the whole function.
    normalized_expr = _normalize(expr)
    if normalized_expr.is_Atom:
        memo[expr] = normalized_expr
        return normalized_expr

    # Recurse first
    if normalized_expr.args:
        new_args = tuple(simplify_int_div_mod(arg, constraints, bounds, prover, memo) for arg in normalized_expr.args)
        if new_args != normalized_expr.args:
             normalized_expr = normalized_expr.func(*new_args)
        normalized_expr = _normalize(normalized_expr)
    
    if normalized_expr.is_Atom:
        memo[expr] = normalized_expr
        return normalized_expr

    # print("Simplifying:", expr)

    # print("Visiting:", expr)
    # ----------------------------------------
    # (A) Local algebraic rules (single-node)
    # ----------------------------------------
    
    res = normalized_expr # default result if no rules apply
    
    # A1: Mod(d*q + r, d) -> Mod(r, d)
    m = normalized_expr.match(Mod(_wd_int*_wq + _wr_int, _wd_int))
    if m:
        res = Mod(m[_wr_int], m[_wd_int])
    # A1b: Commutative twin Mod(r + d*q, d) -> Mod(r, d)
    elif normalized_expr.match(Mod(_wr_int + _wd_int*_wq, _wd_int)): # Use elif for mutual exclusion
        m = normalized_expr.match(Mod(_wr_int + _wd_int*_wq, _wd_int))
        res = Mod(m[_wr_int], m[_wd_int])

    # A2: floor((d*q + r)/d) -> q (if 0<=r<d), else q + floor(r/d)
    elif normalized_expr.match(sp.floor((_wd_int*_wq2 + _wr)/_wd_int)):
         m = normalized_expr.match(sp.floor((_wd_int*_wq2 + _wr)/_wd_int))
         d_expr, q_expr, r_expr = m[_wd_int], m[_wq2], m[_wr]
         if prover.can_prove(r_expr >= 0) and prover.can_prove(r_expr < d_expr):
             res = q_expr
         else:
             res = q_expr + sp.floor(r_expr/d_expr)

    # A3: floor(Mod(x, d)/d) -> 0 when d > 0
    elif normalized_expr.match(sp.floor(Mod(_wx, _wd_int)/_wd_int)):
        m = normalized_expr.match(sp.floor(Mod(_wx, _wd_int)/_wd_int))
        if prover.can_prove(m[_wd_int] > 0):
            res = sp.Integer(0)

    # A4: floor(x/a) -> 0 if 0 <= x < a
    elif normalized_expr.match(sp.floor(_wx/_wa_int)):
        m = normalized_expr.match(sp.floor(_wx/_wa_int))
        x_expr, a_expr = m[_wx], m[_wa_int]
        if prover.can_prove(x_expr >= 0) and prover.can_prove(x_expr < a_expr):
            res = sp.Integer(0)

    # A5: Mod(x, a) -> x if 0 <= x < a
    elif isinstance(normalized_expr, Mod):
        x_expr, a_expr = normalized_expr.args
        if prover.can_prove(a_expr > 0) and prover.can_prove(x_expr >= 0) and prover.can_prove(x_expr < a_expr):
            res = x_expr

    # A6 (optional): floor(n + y) -> n + floor(y) for integer n
    elif normalized_expr.match(sp.floor(_wx_int + _wx)):
        m = normalized_expr.match(sp.floor(_wx_int + _wx))
        res = m[_wx_int] + sp.floor(m[_wx])
    elif normalized_expr.match(sp.floor(_wx + _wx_int)):
        m = normalized_expr.match(sp.floor(_wx + _wx_int))
        res = m[_wx_int] + sp.floor(m[_wx])

    # ----------------------------------------
    # (B) Pairwise combine inside Add nodes
    #     ... + a*floor(x/a) + Mod(x,a) + ... -> ... + x + ...
    # ----------------------------------------
    if res == normalized_expr and isinstance(normalized_expr, sp.Add): # Only check if no other rule applied
        args = list(normalized_expr.args)
        n = len(args)
        found_pairwise = False
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
                    # Recursive call on result
                    res = simplify_int_div_mod(sp.Add(*new_args), constraints, bounds, prover, memo)
                    found_pairwise = True
                    break
            if found_pairwise: break

    # ----------------------------------------
    # (C) Local fold (single-node)
    #     a*floor(x/a) + Mod(x, a) -> x
    # ----------------------------------------
    if res == normalized_expr: # check C only if nothing else
        m = normalized_expr.match(_wa_int*sp.floor(_wx/_wa_int) + Mod(_wx, _wa_int))
        if m:
            a, x = m[_wa_int], m[_wx]
            if prover.can_prove(a != 0):
                 res = x

    memo[expr] = res
    return res


def simplify_ops(expr, bounds, additional_bounds=None, var_types=None, timeout=100, mode='fast'):
    if additional_bounds is None:
        additional_bounds = []

    # Build constraints once
    constraints = [sp.Ge(sym, lb, evaluate=False)
                   for sym, (lb, _) in bounds.items()]
    constraints += [sp.Lt(sym, ub, evaluate=False)
                    for sym, (_, ub) in bounds.items()]
    constraints += additional_bounds

    # Reuse one solver + caches for the whole fixpoint loop
    if mode == 'z3':
        prover = ProverContext(constraints, var_types=var_types, timeout=timeout)
    else:
        prover = FastProverContext(constraints, timeout=timeout)

    cur = expr
    memo = {}
    while True:
        nxt = simplify_int_div_mod(cur, constraints, bounds, prover=prover, memo=memo)
        if nxt == cur:
            return cur
        cur = nxt
