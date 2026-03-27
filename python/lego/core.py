
import math
from sympy import Symbol, symbols, Piecewise, Function
from typing import List, Tuple, Callable
import sympy as sp
from functools import reduce


class BroadcastRange(Function):
    is_integer = True

    @classmethod
    def eval(cls, expr, dim, total):
        # No automatic evaluation; keep the expression unevaluated.
        return None

    def __new__(cls, expr, dim, total):
        # Store expr, dim, total as arguments.
        return Function.__new__(cls, expr, dim, total)

    def _sympystr(self, printer):
        # This method is used by printer.doprint.
        base_expr_str = printer.doprint(self.args[0])
        try:
            dim_val = int(self.args[1])
            total_val = int(self.args[2])
        except (TypeError, ValueError):
            dim_val = self.args[1]
            total_val = self.args[2]
        slices = []
        if isinstance(total_val, int):
            for i in range(total_val):
                slices.append(":" if i == dim_val else "None")
        else:
            slices.append(f"None at dim {dim_val} and ':' otherwise")
        index_str = ", ".join(slices)
        return f"(({base_expr_str})[{index_str}])"

    # Fix the string representation methods
    def __str__(self):
        from sympy.printing.str import StrPrinter
        return self._sympystr(StrPrinter())

    def __repr__(self):
        from sympy.printing.repr import ReprPrinter
        return self._sympystr(ReprPrinter())

    # This property returns the free symbols
    @property
    def free_symbols(self):
        # Include self and free symbols from arguments
        result = {self}
        for arg in self.args:
            if hasattr(arg, 'free_symbols'):
                result.update(arg.free_symbols)
        return result

    # Add comparison methods to ensure inequality operations work
    def _eval_is_ge(self, other):
        return True

    def _eval_is_le(self, other):
        return None  # Let SymPy decide based on other properties


# Keep backward-compatible alias
TritonRange = BroadcastRange


class lego_arange(Function):
    """DSL-agnostic arange placeholder. Each DSL printer renders it appropriately."""
    is_integer = True

    @classmethod
    def eval(cls, start, stop):
        return None


def get_arange(start, stop):
    return lego_arange(start, stop)


def product(symbols: List[Symbol]) -> Symbol:
    return reduce(lambda x, y: x * y, symbols)


def divisibility_constraint(lhs, rhs):
    return sp.Eq(lhs % rhs, 0, evaluate=False)


def le_constraint(lhs, rhs):
    return sp.StrictLessThan(lhs, rhs, evaluate=False)


class LayoutBlock:
    def dims(self) -> Tuple[Symbol, ...]:
        raise NotImplementedError


class GenP(LayoutBlock):
    """Generic Permutation. MLIR: lego.gen_p with apply/inv regions.

    If ``f_inv`` is not provided, attempts to derive it automatically
    for simple (linear/affine) forward functions using SymPy's solver.
    """

    def __init__(self, nd: Tuple[Symbol, ...], f_apply: Callable, f_inv: Callable = None):
        self._dims = nd
        self.f_apply = f_apply
        self.f_inv = f_inv if f_inv is not None else self._try_derive_inverse(nd, f_apply)

    @staticmethod
    def _try_derive_inverse(nd, f_apply):
        """Attempt to symbolically derive the inverse for simple functions.

        Strategy:
        1. For rank-1: use sp.solve directly.
        2. For rank-N: detect weighted-sum (mixed-radix) structure
           ``w0*x0 + w1*x1 + ... + x_{N-1}`` and derive the standard
           divmod inverse.
        """
        rank = len(nd)
        try:
            idx_syms = sp.symbols(
                [f"_auto_inv_{k}" for k in range(rank)],
                integer=True, positive=True,
            )
            flat_sym = sp.Symbol("_auto_inv_flat", integer=True, positive=True)
            forward_expr = f_apply(tuple(idx_syms))

            # Only handle scalar (non-Piecewise) expressions
            if isinstance(forward_expr, sp.Piecewise):
                return None

            # --- Rank 1: direct solve ---
            if rank == 1:
                solutions = sp.solve(forward_expr - flat_sym, idx_syms, dict=True)
                if len(solutions) != 1 or idx_syms[0] not in solutions[0]:
                    return None
                inv_expr = solutions[0][idx_syms[0]]
                allowed = {flat_sym} | set().union(
                    *(e.free_symbols for e in nd if isinstance(e, sp.Expr)))
                if not inv_expr.free_symbols <= allowed:
                    return None
                def f_inv(flat_val, _e=inv_expr, _s=flat_sym):
                    return (_e.subs(_s, flat_val),)
                return f_inv

            # --- Rank N: detect weighted-sum (mixed-radix) pattern ---
            # Expand and collect coefficients of each index symbol.
            expanded = sp.expand(forward_expr)
            coeffs = []
            for s in idx_syms:
                c = expanded.coeff(s)
                if c == 0:
                    return None
                coeffs.append(c)
            # Verify: expanded == sum(c_k * x_k) + constant_term
            reconstructed = sum(c * s for c, s in zip(coeffs, idx_syms))
            remainder = sp.simplify(expanded - reconstructed)
            # remainder must be free of all idx_syms (constant w.r.t. indices)
            if any(s in remainder.free_symbols for s in idx_syms):
                return None

            # Build mixed-radix inverse: iterate left-to-right with divmod.
            # stride[k] = coeffs[k].  inv[k] = floor(rem / stride[k]),
            # rem = rem % stride[k], with initial rem = flat - constant.
            inv_exprs = []
            rem = flat_sym - remainder  # subtract constant offset
            for k in range(rank):
                stride = coeffs[k]
                inv_exprs.append(sp.floor(rem / stride))
                rem = sp.Mod(rem, stride)

            allowed = {flat_sym} | set().union(
                *(e.free_symbols for e in nd if isinstance(e, sp.Expr)))
            for expr in inv_exprs:
                if not expr.free_symbols <= allowed:
                    return None

            def f_inv(flat_val, _exprs=inv_exprs, _s=flat_sym):
                return tuple(e.subs(_s, flat_val) for e in _exprs)
            return f_inv

        except (NotImplementedError, ValueError, TypeError):
            return None

    def dims(self):
        return self._dims


class RegP(LayoutBlock):
    """Regular Permutation. MLIR: lego.reg_p with perm vector."""

    def __init__(self, nd: Tuple[Symbol, ...], perm: Tuple[int, ...]):
        self._dims = nd
        self._perm_vector = list(perm)

    def dims(self):
        return self._dims


class Row(LayoutBlock):
    """Row-major layout. MLIR: lego.row [dims]."""

    def __init__(self, *dims: Symbol):
        self._dims = dims

    def dims(self):
        return self._dims


class Col(LayoutBlock):
    """Column-major layout. MLIR: lego.col [dims]."""

    def __init__(self, *dims: Symbol):
        self._dims = dims

    def dims(self):
        return self._dims


class OrderBy(LayoutBlock):
    """Ordered sequence of permutation blocks. MLIR: lego.order_by [perms]."""

    def __init__(self, *perms: LayoutBlock):
        self.perms = perms
        self.chain = [self]

    def OrderBy(self, *perms: LayoutBlock):
        new_o = OrderBy(*perms)
        new_o.chain = self.chain + [new_o]
        self.chain.append(new_o)
        return self

    def GroupBy(self, group_dims: List[Tuple[Symbol, ...]], user_constraints=[]) -> 'GroupBy':
        return GroupBy(group_dims, self.chain, user_constraints)

    def TileBy(self, *group_dims: Tuple[Symbol, ...], user_constraints=[]):
        dims = tuple(d for dim_tuple in group_dims for d in dim_tuple)
        return TileByLayout(
            input_chain=list(self.chain),
            tile_groups=list(group_dims),
            group_dims=[dims],
            user_constraints=user_constraints,
        )

    def dims(self):
        all_dims = []
        for perm in self.perms:
            all_dims.extend(list(perm.dims()))
        return tuple(all_dims)


def _merge_bound(constraints, sym, lb=None, ub=None):
    """Merge a new bound into the constraints dict for sym.

    When both old and new bounds exist, takes the tighter bound:
    max for lower bounds, min for upper bounds.
    """
    if sym not in constraints:
        constraints[sym] = (lb, ub)
    else:
        old_lb, old_ub = constraints[sym]
        if old_lb is None:
            new_lb = lb
        elif lb is None:
            new_lb = old_lb
        else:
            new_lb = sp.Max(old_lb, lb)
        if old_ub is None:
            new_ub = ub
        elif ub is None:
            new_ub = old_ub
        else:
            new_ub = sp.Min(old_ub, ub)
        constraints[sym] = (new_lb, new_ub)


class GroupBy(LayoutBlock):
    """Grouped layout. MLIR: lego.group_by [dims] [objects].

    All evaluation (apply, inv, __getitem__) goes through MLIR roundtrip.
    """

    def __init__(self, group_dims: List[Tuple[Symbol, ...]], objects: List[OrderBy], user_constraints=[]):
        self._dims = tuple(d for dim_tuple in group_dims for d in dim_tuple)
        self.objects = objects
        self.d = len(group_dims[0])  # used by MLIRTensor
        self.user_constraints = user_constraints

    def apply(self, *idx: Symbol) -> Symbol:
        """Apply the GroupBy operation via MLIR roundtrip."""
        from .backend.symbolic import simplify_via_mlir

        if len(idx) != len(self._dims):
            raise ValueError(
                f"Input index dimension mismatch. Expected {len(self._dims)}, got {len(idx)}")

        index_constraints = {}
        for i, item in enumerate(idx):
            if isinstance(item, (sp.Symbol, sp.Expr)):
                index_constraints[item] = (0, self._dims[i])
        constraints = self._build_constraints(index_constraints)
        return simplify_via_mlir(self, 'apply', list(idx), constraints)

    def inv(self, flat_idx: Symbol) -> Tuple[Symbol, ...]:
        """Apply the inverse GroupBy operation via MLIR roundtrip."""
        from .backend.symbolic import simplify_via_mlir

        index_constraints = {flat_idx: (0, product(self._dims))}
        constraints = self._build_constraints(index_constraints)
        return simplify_via_mlir(self, 'inv', flat_idx, constraints)

    def dims(self) -> Tuple[Symbol, ...]:
        return self._dims

    def _get_all_dim_symbols(self):
        syms = set()
        for d in self._dims:
            if isinstance(d, sp.Expr):
                syms |= d.free_symbols
        for obj in self.objects:
            for d in obj.dims():
                if isinstance(d, sp.Expr):
                    syms |= d.free_symbols
        return syms

    def _get_input_constraints(self):
        return list(map(lambda x: sp.Gt(x, 0, evaluate=False), set().union(
            *(e.free_symbols for t in [self.dims()] + [x.dims() for x in self.objects] for e in t if isinstance(e, sp.Expr)))))

    def _build_constraints(self, index_constraints=None):
        """Build a unified constraint dict for MLIR assume_bounds."""
        constraints = dict(index_constraints or {})

        dim_syms = self._get_all_dim_symbols()
        for sym in dim_syms:
            if sym not in constraints:
                constraints[sym] = (1, None)

        for c in self.user_constraints:
            self._parse_relational(c, constraints)

        for c in self._get_input_constraints():
            self._parse_relational(c, constraints)

        return constraints

    @staticmethod
    def _parse_relational(rel, constraints):
        """Parse a SymPy relational into the constraints dict."""
        if not isinstance(rel, sp.core.relational.Relational):
            return

        lhs, rhs = rel.args

        if isinstance(rel, sp.StrictGreaterThan):
            if isinstance(lhs, sp.Symbol):
                lb = rhs + 1 if not isinstance(rhs, (int, sp.Integer)) else int(rhs) + 1
                _merge_bound(constraints, lhs, lb=lb)
        elif isinstance(rel, sp.GreaterThan):
            if isinstance(lhs, sp.Symbol):
                lb = rhs if not isinstance(rhs, (int, sp.Integer)) else int(rhs)
                _merge_bound(constraints, lhs, lb=lb)
        elif isinstance(rel, sp.StrictLessThan):
            if isinstance(lhs, sp.Symbol):
                _merge_bound(constraints, lhs, ub=rhs)
        elif isinstance(rel, sp.LessThan):
            if isinstance(lhs, sp.Symbol):
                ub = rhs + 1 if not isinstance(rhs, (int, sp.Integer)) else int(rhs) + 1
                _merge_bound(constraints, lhs, ub=ub)


    def transform(self, tensor):
        """Apply layout transform via MLIR JIT compilation."""
        from .backend.compiler import get_compiler
        if not hasattr(self, '_compiled'):
            self._compiled = get_compiler(self, tensor.shape)
        return self._compiled.transform_numpy(tensor) if hasattr(tensor, 'ctypes') \
            else self._compiled.transform_numpy(tensor)

    def inverse_transform(self, tensor):
        """Apply inverse layout transform."""
        from .backend.compiler import get_compiler
        if not hasattr(self, '_compiled'):
            self._compiled = get_compiler(self, tensor.shape)
        return self._compiled.inverse_transform_numpy(tensor)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        result = []
        self.constraints = {}
        logical_range = self.dims()
        tr_to_dummy = {}
        dummy_to_tr = {}
        i = 0
        for idx, item in enumerate(key):
            if isinstance(item, slice):
                all_slices = [i for i, it in enumerate(key) if isinstance(it, slice)]
                num_slices = len(all_slices)
                slice_rank = all_slices.index(idx)

                start = 0
                end = logical_range[idx]
                if item.start is not None:
                    start = item.start
                if item.stop is not None:
                    end = item.stop
                expr_new_axis = BroadcastRange(
                    get_arange(start, end), slice_rank, num_slices)
                sym = sp.symbols(f"_tr{i}", integer=True)
                i += 1
                tr_to_dummy[expr_new_axis] = sym
                dummy_to_tr[sym] = expr_new_axis
                self.constraints[sym] = (start, end)
                result.append(sym)
            elif isinstance(item, sp.Expr):
                sym = sp.symbols(f"_tr{i}", integer=True)
                i += 1
                tr_to_dummy[item] = sym
                dummy_to_tr[sym] = item
                self.constraints[sym] = (0, logical_range[idx])
                result.append(sym)
            elif isinstance(item, str):
                s = sp.symbols(item, integer=True)
                result.append(s)
                self.constraints[s] = (0, logical_range[idx])
            else:
                result.append(item)
                self.constraints[item] = (0, logical_range[idx])
        from .backend.symbolic import simplify_via_mlir

        constraints = self._build_constraints(self.constraints)
        simplified = simplify_via_mlir(self, 'apply', list(result),
                                       constraints)
        return simplified.xreplace(dummy_to_tr)


class TileByLayout(GroupBy):
    """Tiled layout. MLIR: lego.tile_by %input [tile_dims] tile_shape.

    Fields for MLIR emission:
      _input_chain  — list of OrderBy (emitted as lego.order_by)
      _tile_groups  — list of dim tuples (emitted as tile_dims)
      tile_shape    — [d, d, ..., d] (q times)
    """

    def __init__(self, input_chain, tile_groups, group_dims, user_constraints=[]):
        self._dims = tuple(d for dim_tuple in group_dims for d in dim_tuple)
        self.d = len(group_dims[0])  # used by MLIRTensor
        self.user_constraints = user_constraints
        self._input_chain = input_chain
        self._tile_groups = tile_groups

    @property
    def tile_shape(self):
        d = len(self._tile_groups[0])
        q = len(self._tile_groups)
        return [d] * q

    def _get_all_dim_symbols(self):
        syms = set()
        for d in self._dims:
            if isinstance(d, sp.Expr):
                syms |= d.free_symbols
        for orderby in self._input_chain:
            for p in orderby.perms:
                for d in p.dims():
                    if isinstance(d, sp.Expr):
                        syms |= d.free_symbols
        return syms

    def _get_input_constraints(self):
        all_dims = list(self.dims())
        for orderby in self._input_chain:
            for p in orderby.perms:
                all_dims.extend(p.dims())
        constraints = list(map(lambda x: sp.Gt(x, 0, evaluate=False),
                        set().union(*(e.free_symbols for e in all_dims if isinstance(e, sp.Expr)))))

        # Auto-infer divisibility from tile dimensions containing floor(A/B).
        # If a tile dim is floor(A/B), infer that B divides A (i.e. A % B == 0).
        for g in self._tile_groups:
            for d in g:
                if isinstance(d, sp.Expr) and not isinstance(d, sp.Symbol):
                    # SymPy represents M//B as floor(M/B)
                    if isinstance(d, sp.floor):
                        inner = d.args[0]
                        num, den = inner.as_numer_denom()
                        if den != sp.S.One:
                            constraints.append(
                                sp.Eq(sp.Mod(num, den), 0, evaluate=False))
                    # Also check for Mul with Pow(-1) patterns: M * B**(-1)
                    elif isinstance(d, sp.Mul):
                        num, den = d.as_numer_denom()
                        if den != sp.S.One and isinstance(den, sp.Expr):
                            constraints.append(
                                sp.Eq(sp.Mod(num, den), 0, evaluate=False))

        return constraints


def antidiag(n, args: Tuple[Symbol, ...]):
    i, j = args
    antidiag = i + j + 1

    flat_ind_expr = Piecewise(
        ((antidiag * (antidiag - 1) // 2) + i, antidiag <= n),
        (
            (n * n - n) + i - ((2 * n - antidiag)
                               * (2 * n - antidiag - 1) // 2),
            True
        )
    )
    return flat_ind_expr


def antidiag_inv(n, x0):
    """
    Inverse of the antidiagonal mapping for an n x n matrix.
    Given a flattened index x0, returns the (i, j) coordinates.
    """
    S1 = n * (n + 1) // 2

    if x0 < S1:
        k = math.floor((math.sqrt(8 * x0 + 1) - 1) / 2) + 1
        i = x0 - (k * (k - 1) // 2)
        j = (k - 1) - i
    else:
        m2 = x0 - S1
        d = math.floor(
            (2 * n - 1 - math.sqrt((2 * n - 1) ** 2 - 8 * m2)) / 2) + 1
        prev = (d - 1) * n - ((d - 1) * d // 2)
        i = d + (m2 - prev)
        j = (n + d - 1) - i

    return (i, j)


def new_antidiag(n1, n2, args: Tuple[Symbol, ...]):
    i, j = args
    return (n1-1-i) * n2 + n2-1-j
