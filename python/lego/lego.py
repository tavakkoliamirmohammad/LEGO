
import math
from sympy import Symbol, symbols, Piecewise, Function
from typing import List, Tuple, Callable
import sympy as sp
from functools import reduce
# simplifier.py (z3) no longer needed — using MLIR roundtrip


class TritonRange(Function):
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


def get_arange(start, stop):
    tl_arange = Function("tl.arange", integer=True)
    return tl_arange(start, stop)


def product(symbols: List[Symbol]) -> Symbol:
    return reduce(lambda x, y: x * y, symbols)


def divisibility_constraint(lhs, rhs):
    return sp.Eq(lhs % rhs, 0, evaluate=False)


def le_constraint(lhs, rhs):
    return sp.StrictLessThan(lhs, rhs, evaluate=False)


def equal_constraint(lhs, rhs):
    return sp.Eq(lhs, rhs, evaluate=False)


def get_sigma_perm(d, q):
    return list(sum([[k + d * h for h in range(q)]
                for k in range(d)], []))


def flatten_index(indices: Tuple[Symbol, ...], dims: Tuple[Symbol, ...]):
    """
    Flattens multi-dimensional indices into a single flat index.
    Equivalent to B_n(i, n)
    Example: flatten_index((i1, i2, i3), (n1, n2, n3)) = i1*n2*n3 + i2*n3 + i3
    """
    if len(indices) != len(dims):
        raise ValueError("Number of indices must match number of dimensions")

    flat_index = 0
    rank = len(dims)
    for k in range(rank):
        # Calculate the product of subsequent dimensions
        prod_n_k_plus_1_to_q = product(dims[k + 1:]) if k + 1 < rank else 1
        flat_index += indices[k] * prod_n_k_plus_1_to_q

    return flat_index


def unflatten_index(flat_index: int, dims: Tuple[Symbol, ...]) -> Tuple[Symbol, ...]:
    """
    Converts a flat index back into multi-dimensional indices.
    Equivalent to B_n^-1(iflat, n)
    Example: unflatten_index(iflat, (n1, n2, n3)) = (i1, i2, i3)
    """
    if not dims:
        return ()  # Empty tuple for empty dimensions

    indices = []
    rank = len(dims)
    current_index = flat_index

    for k in range(rank):
        # Calculate the product of subsequent dimensions
        prod_n_k_plus_1_to_q = product(dims[k + 1:]) if k + 1 < rank else 1

        # Calculate the k-th index
        ik = current_index // prod_n_k_plus_1_to_q
        indices.append(ik)

        # Update the remaining index
        current_index %= prod_n_k_plus_1_to_q

    return tuple(indices)


class LayoutBlock:
    def apply(self, idx: Tuple[Symbol, ...]) -> Symbol:
        raise NotImplementedError

    def inv(self, flat_idx: Symbol) -> Tuple[Symbol, ...]:
        raise NotImplementedError

    def dims(self) -> Tuple[Symbol, ...]:
        raise NotImplementedError


class GenP(LayoutBlock):
    """
    Generic Permutation Block.
    Applies arbitrary functions f_nd and f_inv.
    """

    def __init__(self, nd: Tuple[Symbol, ...], f_apply: Callable[[Tuple[Symbol, ...]], Symbol], f_inv: Callable[[Symbol], Tuple[Symbol, ...]]):
        """
        Initializes GenP.
        Args:
            nd: Tuple of dimensions.
            f_apply: The apply function (takes multi-dim index, returns flat index).
            f_inv: The inverse function (takes flat index, returns multi-dim index).
        """
        self._dims = nd
        self.f_apply = f_apply
        self.f_inv = f_inv

    def apply(self, idx: Tuple[Symbol, ...]) -> Symbol:
        if len(idx) != len(self._dims):
            raise ValueError("Input index dimension mismatch")
        return self.f_apply(idx)

    def inv(self, idx: Symbol) -> Tuple[Symbol, ...]:
        return self.f_inv(idx)

    def dims(self) -> Tuple[int, ...]:
        """Returns the dimensions."""
        return self._dims


def sigma(values: Tuple[Symbol, ...], permutation: Tuple[Symbol, ...]):
    d = len(permutation)
    new_arr = [0] * d
    for i in range(d):
        new_arr[i] = values[permutation[i]]
    return new_arr


def inverse_permutation(p: Tuple[int, ...]) -> Tuple[int, ...]:
    d = len(p)
    p_inv = [0] * d
    for i, pi in enumerate(p):
        p_inv[pi] = i
    return p_inv


class RegP(LayoutBlock):
    """
    Regular Permutation Block.
    Applies a permutation sigma.
    """

    def __init__(self, nd: Tuple[Symbol, ...], perm: Tuple[int, ...]):
        """
        Initializes RegP.
        Args:
            nd: Tuple of dimensions.
            sigma: Tuple of integers representing the permutation. It takes a multi-dimensional
                   index tuple and returns the permuted multi-dimensional index tuple.
        """
        self._dims = nd
        self._perm_vector = list(perm)
        self.perm = lambda idx: sigma(idx, perm)
        self.perm_inv = lambda idx: sigma(idx, inverse_permutation(perm))

    def apply(self, idx: Tuple[Symbol, ...]) -> Symbol:
        if len(idx) != len(self._dims):
            raise ValueError("Input index dimension mismatch")
        return flatten_index(self.perm(idx), self.perm(self._dims))

    def inv(self, flat_idx: int) -> Tuple[int, ...]:
        """Unflattens the index and applies the inverse permutation."""
        unflattened_idx = unflatten_index(flat_idx, self.perm(self._dims))
        original_idx = self.perm_inv(unflattened_idx)
        return original_idx

    def dims(self) -> Tuple[int, ...]:
        """Returns the dimensions."""
        return self._dims


def Row(*group_dims: Symbol):
    d = len(group_dims)
    q = 1
    return RegP(group_dims, get_sigma_perm(d, q))


def Col(*group_dims: Symbol):
    d = len(group_dims)
    q = 1
    return RegP(group_dims, get_sigma_perm(d, q)[::-1])


class OrderBy(LayoutBlock):
    """
    OrderBy Block.
    Applies a sequence of Permutation blocks.
    """

    def __init__(self, *perms: LayoutBlock):
        """
        Initializes OrderBy.
        Args:
            perms: A list of PermutationBlock objects (like GenP or RegP).
        """
        self.perms = perms
        self._cached_dims = self._compute_dims()  # Cache dimensions
        self.q = len(self.perms)
        self.d = len(self.perms[0].dims())
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
        d = len(group_dims[0])
        q = len(group_dims)
        sigma_dq = get_sigma_perm(d, q)
        new_order_by = []
        for o in self.chain:
            o_dims = []
            for p in o.perms:
                o_dims.extend(p.dims())
            sigma_o = get_sigma_perm(o.d, o.q)
            sigma_o_inv = inverse_permutation(sigma_o)
            new_order_by.append(o)
            new_order_by.append(
                OrderBy(RegP(sigma(o_dims, sigma_o), sigma_o_inv)))
        return GroupBy([dims], new_order_by + [OrderBy(RegP(dims, sigma_dq))], user_constraints)

    def apply(self, idx: Tuple[Symbol, ...]) -> Symbol:
        """
        Applies the sequence of permutations.
        Input: A multi-dimensional index `idx` matching the combined dimensions.
        Output: A single flat index representing the final transformed position.
        """
        if len(idx) != len(self._cached_dims):
            raise ValueError(
                f"Input index length {len(idx)} must match combined dimension length {len(self._cached_dims)}")

        output_flat_index = 0
        current_idx_offset = 0  # Tracks the starting position in the input `idx`
        # print(self.perms)
        # print("idx: ", idx)
        for k, perm in enumerate(self.perms):
            perm_dims = perm.dims()
            # print(perm_dims)
            dim_count = len(perm_dims)
            # p = Product(n_h) for this perm
            total_perm_elements = product(perm_dims)

            if current_idx_offset + dim_count > len(idx):
                raise ValueError("Index length mismatch during OrderBy apply")
            icur = idx[current_idx_offset: current_idx_offset + dim_count]
            iflat_cur = perm.apply(icur)
            output_flat_index = output_flat_index * total_perm_elements + iflat_cur
            current_idx_offset += dim_count

        return output_flat_index

    def inv(self, flat_idx: Symbol) -> Tuple[Symbol, ...]:
        """
        Applies the inverse of the sequence of permutations.
        Input: A single flat index.
        Output: The original multi-dimensional index.
        """
        original_indices = []
        remaining_flat_idx = flat_idx

        # Iterate through permutations in reverse order
        for perm in reversed(self.perms):
            perm_dims = perm.dims()
            total_perm_elements = product(perm_dims)  # p = Product(n_h)
            if total_perm_elements == 0:
                # Handle zero-dimension case if necessary
                if remaining_flat_idx != 0:
                    raise ValueError(
                        "Invalid flat index for zero-sized dimension")
                # Add empty tuple or handle as appropriate for zero dims
                # For consistency, let's assume dims are always > 0
                raise ValueError("Permutation dimensions cannot be zero")

            # Extract the flat index portion for this permutation
            iflat_cur = remaining_flat_idx % total_perm_elements
            # Update the remaining flat index for the next iteration (previous perm)
            remaining_flat_idx //= total_perm_elements

            # Apply the inverse permutation
            original_icur = perm.inv(iflat_cur)
            # Prepend the original indices found for this permutation
            original_indices = list(original_icur) + original_indices
        return tuple(original_indices)

    def _compute_dims(self) -> Tuple[Symbol, ...]:
        """Computes the combined dimensions from all permutations."""
        # The image notation `n <- empty sequence; for Perm ... n <- n, Perm.dims()`
        # suggests concatenating dimensions.
        all_dims = []
        for perm in self.perms:
            all_dims.extend(list(perm.dims()))
        return tuple(all_dims)

    def dims(self) -> Tuple[Symbol, ...]:
        """Returns the combined dimensions."""
        return self._cached_dims


def _merge_bound(constraints, sym, lb=None, ub=None):
    """Merge a new bound into the constraints dict for sym."""
    if sym not in constraints:
        constraints[sym] = (lb, ub)
    else:
        old_lb, old_ub = constraints[sym]
        new_lb = lb if old_lb is None else (lb if lb is not None else old_lb)
        new_ub = ub if old_ub is None else (ub if ub is not None else old_ub)
        constraints[sym] = (new_lb, new_ub)


class GroupBy(LayoutBlock):
    """
    GroupBy Block.
    Groups operations based on dimensions and applies a sequence of objects/blocks.
    Note: The objects 'O' in the image are not specified as PermutationBlocks,
          they could be other types of transformations with apply/inv/dims methods.
    """

    def __init__(self, group_dims: List[Tuple[Symbol, ...]], objects: List[OrderBy], user_constraints=[]):
        """
        Initializes GroupBy.
        Args:
            group_dims: A list of dimension tuples, e.g., [(n1,), (n2, n3), ...].
                        Corresponds to ([n^1], ..., [n^q]) in the image.
            objects: A list of objects (O^o) that have apply, inv, and dims methods.
                     The dims method should return the dimensions the object operates on.
        """
        # Combine the list of dimension tuples into a single tuple
        self._dims = tuple(d for dim_tuple in group_dims for d in dim_tuple)
        self.objects = objects  # O^o in the image notation
        self.d = len(group_dims[0])
        self.user_constraints = user_constraints

    def apply(self, *idx: Symbol) -> Symbol:
        """
        Applies the GroupBy operation.
        Input: Multi-dimensional index `idx` matching the combined group dimensions.
        Output: A flat index resulting from applying objects in reverse.
        """
        if len(idx) != len(self._dims):
            raise ValueError(
                f"Input index dimension mismatch. Expected {len(self._dims)}, got {len(idx)}")

        current_flat_index = flatten_index(idx, self._dims)

        for obj in reversed(self.objects):
            obj_dims = obj.dims()  # n'd_1, ..., n'd_qo
            idx_for_obj = unflatten_index(current_flat_index, obj_dims)
            current_flat_index = obj.apply(idx_for_obj)
        return current_flat_index

    def inv(self, flat_idx: Symbol) -> Tuple[Symbol, ...]:
        """
        Applies the inverse GroupBy operation.
        Input: A flat index.
        Output: The original multi-dimensional index.
        """
        from .lego_mlir import simplify_via_mlir

        index_constraints = {flat_idx: (0, product(self._dims))}
        constraints = self._build_constraints(index_constraints)
        return simplify_via_mlir(self, 'inv', flat_idx, constraints)

    def dims(self) -> Tuple[Symbol, ...]:
        """Returns the combined dimensions of the GroupBy block."""
        return self._dims

    def _get_all_dim_symbols(self):
        """Collect all SymPy symbols used in layout dimensions."""
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
        """Build a unified constraint dict for MLIR assume_bounds.

        Merges:
        1. index_constraints: {sym: (lb, ub)} from __getitem__/inv
        2. dim_symbols: all layout dimension symbols get lb=1
        3. user_constraints: SymPy relationals from TileBy/GroupBy
        4. _get_input_constraints: auto-inferred "dim > 0" constraints

        Returns:
            dict {sp.Symbol: (lb, ub)} where lb/ub can be int, sp.Expr, or None
        """
        constraints = dict(index_constraints or {})

        # Ensure all dimension symbols have lb >= 1
        dim_syms = self._get_all_dim_symbols()
        for sym in dim_syms:
            if sym not in constraints:
                constraints[sym] = (1, None)

        # Parse user_constraints (SymPy relationals) into constraint dict
        for c in self.user_constraints:
            self._parse_relational(c, constraints)

        # Parse auto-inferred constraints
        for c in self._get_input_constraints():
            self._parse_relational(c, constraints)

        return constraints

    @staticmethod
    def _parse_relational(rel, constraints):
        """Parse a SymPy relational into the constraints dict.

        Updates constraints in-place. Handles:
        - sp.Gt(x, v)  → x: lb = v+1
        - sp.Ge(x, v)  → x: lb = v
        - sp.Lt(x, v)  → x: ub = v
        - sp.Le(x, v)  → x: ub = v+1
        - sp.Eq(x%d, 0) → (divisibility, stored for reference)
        """
        if not isinstance(rel, sp.core.relational.Relational):
            return

        lhs, rhs = rel.args

        if isinstance(rel, sp.StrictGreaterThan):  # lhs > rhs → lhs >= rhs+1
            if isinstance(lhs, sp.Symbol):
                lb = rhs + 1 if not isinstance(rhs, (int, sp.Integer)) else int(rhs) + 1
                _merge_bound(constraints, lhs, lb=lb)
        elif isinstance(rel, sp.GreaterThan):  # lhs >= rhs
            if isinstance(lhs, sp.Symbol):
                lb = rhs if not isinstance(rhs, (int, sp.Integer)) else int(rhs)
                _merge_bound(constraints, lhs, lb=lb)
        elif isinstance(rel, sp.StrictLessThan):  # lhs < rhs
            if isinstance(lhs, sp.Symbol):
                _merge_bound(constraints, lhs, ub=rhs)
        elif isinstance(rel, sp.LessThan):  # lhs <= rhs
            if isinstance(lhs, sp.Symbol):
                ub = rhs + 1 if not isinstance(rhs, (int, sp.Integer)) else int(rhs) + 1
                _merge_bound(constraints, lhs, ub=ub)


    def transform(self, tensor):
        """Apply layout transform to a PyTorch tensor or NumPy array.

        Uses the MLIR Python bindings + JIT compilation path when available,
        otherwise falls back to interpreted (SymPy) evaluation.
        """
        from .compiler import get_compiler
        if not hasattr(self, '_compiled'):
            self._compiled = get_compiler(self, tensor.shape)
        return self._compiled.transform_numpy(tensor) if hasattr(tensor, 'ctypes') \
            else self._compiled.transform_numpy(tensor)

    def inverse_transform(self, tensor):
        """Apply inverse layout transform."""
        from .compiler import get_compiler
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
                # Count slices to determine broadcasting rank
                all_slices = [i for i, it in enumerate(key) if isinstance(it, slice)]
                num_slices = len(all_slices)
                slice_rank = all_slices.index(idx)
                
                start = 0
                end = logical_range[idx]
                if item.start is not None:
                    start = item.start
                if item.stop is not None:
                    end = item.stop
                expr_new_axis = TritonRange(
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
        from .lego_mlir import simplify_via_mlir

        constraints = self._build_constraints(self.constraints)
        simplified = simplify_via_mlir(self, 'apply', list(result),
                                       constraints)
        return simplified.xreplace(dummy_to_tr)


def antidiag(n, args: Tuple[Symbol, ...]):
    i, j = args
    antidiag = i + j + 1

    # Define the formula using Piecewise
    flat_ind_expr = Piecewise(
        # Case 1: When on or above the main anti-diagonal (antidiag <= n)
        ((antidiag * (antidiag - 1) // 2) + i, antidiag <= n),

        # Case 2: When below the main anti-diagonal (antidiag > n)
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
    # Sum of first n natural numbers
    S1 = n * (n + 1) // 2

    if x0 < S1:
        # Within the first antidiagonal triangle
        k = math.floor((math.sqrt(8 * x0 + 1) - 1) / 2) + 1
        i = x0 - (k * (k - 1) // 2)
        j = (k - 1) - i
    else:
        # Beyond the first triangle, in the complementary region
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