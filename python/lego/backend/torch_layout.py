"""
LEGO Layout → PyTorch Compilation

Compiles MLIR-lowered layout index expressions (SymPy) into vectorized
PyTorch functions. This replaces dense O(numel) permutation tables with
O(1) arithmetic per element.

Pipeline:
  Layout → MLIR LEGO dialect → lego-lower → arith ops → SymPy → PyTorch function
"""

import sympy as sp
from functools import reduce

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def sympy_to_torch(expr, symbols):
    """Compile a SymPy expression to a vectorized PyTorch function.

    Parameters
    ----------
    expr : sp.Expr
        The SymPy expression to compile (e.g., ``8*i + j``).
    symbols : list of sp.Symbol
        Ordered list of input symbols. The returned function takes
        one torch.Tensor argument per symbol.

    Returns
    -------
    callable
        A function ``f(*tensors) -> torch.Tensor`` that evaluates
        the expression elementwise on tensors of indices.
    """
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    code = _compile_node(expr, sym_to_idx)

    def compiled_fn(*args):
        return code(args)

    return compiled_fn


def _compile_node(expr, sym_to_idx):
    """Recursively compile a SymPy expression node to a torch lambda."""

    # Integer constant
    if isinstance(expr, (int, sp.Integer)):
        val = int(expr)
        return lambda args: val

    # Symbol → input tensor
    if isinstance(expr, sp.Symbol):
        if expr in sym_to_idx:
            idx = sym_to_idx[expr]
            return lambda args, _i=idx: args[_i]
        raise KeyError(f"Symbol {expr} not in input list")

    # Addition
    if isinstance(expr, sp.Add):
        children = [_compile_node(a, sym_to_idx) for a in expr.args]
        def add_fn(args, _ch=children):
            result = _ch[0](args)
            for c in _ch[1:]:
                result = result + c(args)
            return result
        return add_fn

    # Multiplication
    if isinstance(expr, sp.Mul):
        # Check for division pattern: Mul(a, Pow(b, -1))
        num, den = expr.as_numer_denom()
        if den != sp.S.One:
            num_fn = _compile_node(num, sym_to_idx)
            den_fn = _compile_node(den, sym_to_idx)
            return lambda args, _n=num_fn, _d=den_fn: torch.div(
                _n(args), _d(args), rounding_mode='trunc'
            )
        children = [_compile_node(a, sym_to_idx) for a in expr.args]
        def mul_fn(args, _ch=children):
            result = _ch[0](args)
            for c in _ch[1:]:
                result = result * c(args)
            return result
        return mul_fn

    # Floor division: floor(a/b)
    if isinstance(expr, sp.floor):
        inner = expr.args[0]
        num, den = inner.as_numer_denom()
        if den != sp.S.One:
            num_fn = _compile_node(num, sym_to_idx)
            den_fn = _compile_node(den, sym_to_idx)
            return lambda args, _n=num_fn, _d=den_fn: torch.div(
                _n(args), _d(args), rounding_mode='trunc'
            )
        inner_fn = _compile_node(inner, sym_to_idx)
        return inner_fn

    # Modulo: Mod(a, b)
    if isinstance(expr, sp.Mod):
        a_fn = _compile_node(expr.args[0], sym_to_idx)
        b_fn = _compile_node(expr.args[1], sym_to_idx)
        return lambda args, _a=a_fn, _b=b_fn: _a(args) % _b(args)

    # Power (for positive integer exponents)
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        if exp == sp.S.NegativeOne:
            raise NotImplementedError(f"Negative power in torch compilation: {expr}")
        if exp.is_Integer and int(exp) >= 0:
            base_fn = _compile_node(base, sym_to_idx)
            exp_val = int(exp)
            def pow_fn(args, _b=base_fn, _e=exp_val):
                result = 1
                val = _b(args)
                for _ in range(_e):
                    result = result * val
                return result
            return pow_fn

    # Abs
    if isinstance(expr, sp.Abs):
        inner_fn = _compile_node(expr.args[0], sym_to_idx)
        return lambda args, _f=inner_fn: torch.abs(_f(args))

    # Max / Min
    if expr.func == sp.Max:
        a_fn = _compile_node(expr.args[0], sym_to_idx)
        b_fn = _compile_node(expr.args[1], sym_to_idx)
        return lambda args, _a=a_fn, _b=b_fn: torch.maximum(_a(args), _b(args))

    if expr.func == sp.Min:
        a_fn = _compile_node(expr.args[0], sym_to_idx)
        b_fn = _compile_node(expr.args[1], sym_to_idx)
        return lambda args, _a=a_fn, _b=b_fn: torch.minimum(_a(args), _b(args))

    # Piecewise → torch.where
    if isinstance(expr, sp.Piecewise):
        pieces = list(expr.args)
        return _compile_piecewise(pieces, sym_to_idx)

    # Relational (used inside Piecewise conditions)
    if isinstance(expr, sp.core.relational.Relational):
        return _compile_relational(expr, sym_to_idx)

    raise NotImplementedError(
        f"Cannot compile SymPy expression to PyTorch: {type(expr).__name__}: {expr}"
    )


def _compile_relational(rel, sym_to_idx):
    """Compile a SymPy relational to a torch boolean tensor function."""
    lhs_fn = _compile_node(rel.lhs, sym_to_idx)
    rhs_fn = _compile_node(rel.rhs, sym_to_idx)

    if isinstance(rel, (sp.StrictLessThan, sp.Lt)):
        return lambda args, _l=lhs_fn, _r=rhs_fn: _l(args) < _r(args)
    if isinstance(rel, (sp.LessThan, sp.Le)):
        return lambda args, _l=lhs_fn, _r=rhs_fn: _l(args) <= _r(args)
    if isinstance(rel, (sp.StrictGreaterThan, sp.Gt)):
        return lambda args, _l=lhs_fn, _r=rhs_fn: _l(args) > _r(args)
    if isinstance(rel, (sp.GreaterThan, sp.Ge)):
        return lambda args, _l=lhs_fn, _r=rhs_fn: _l(args) >= _r(args)
    if isinstance(rel, sp.Equality):
        return lambda args, _l=lhs_fn, _r=rhs_fn: _l(args) == _r(args)

    raise NotImplementedError(f"Unsupported relational: {type(rel).__name__}")


def _compile_piecewise(pieces, sym_to_idx):
    """Compile a SymPy Piecewise to nested torch.where calls."""
    if len(pieces) == 1:
        # Last piece (default case)
        return _compile_node(pieces[0][0], sym_to_idx)

    val_fn = _compile_node(pieces[0][0], sym_to_idx)
    cond_fn = _compile_node(pieces[0][1], sym_to_idx)
    else_fn = _compile_piecewise(pieces[1:], sym_to_idx)

    return lambda args, _v=val_fn, _c=cond_fn, _e=else_fn: torch.where(
        _c(args), _v(args), _e(args)
    )


if _HAS_TORCH:

    def compile_layout_transform(layout_obj, shape):
        """Lower a layout through MLIR and compile to PyTorch index functions.

        Parameters
        ----------
        layout_obj : GroupBy or TileByLayout
            The layout object (LegoLayout._layout).
        shape : tuple of int
            Concrete tensor shape.

        Returns
        -------
        (fwd_fn, inv_fn, layout_dims) : tuple
            fwd_fn(i0, i1, ..., i_{layout_rank-1}) -> flat_physical_index
            inv_fn list: each maps flat_index -> one coordinate
            layout_dims: the layout's dimension tuple (may differ from shape for TileBy)
        """
        dims = layout_obj._dims if hasattr(layout_obj, '_dims') else layout_obj.dims()
        layout_rank = len(dims)
        idx_syms = sp.symbols([f'_idx_{k}' for k in range(layout_rank)], integer=True)
        flat_sym = sp.Symbol('_flat', integer=True)

        # Get forward expression via MLIR roundtrip
        fwd_expr = layout_obj.apply(*idx_syms)

        # Get inverse expressions
        inv_exprs = layout_obj.inv(flat_sym)

        # Compile forward: (i0, i1, ...) -> flat
        fwd_compiled = sympy_to_torch(fwd_expr, list(idx_syms))

        # Compile inverse: flat -> (i0, i1, ...)
        inv_compiled = [sympy_to_torch(e, [flat_sym]) for e in inv_exprs]

        layout_dims = tuple(int(d) for d in dims)
        return fwd_compiled, inv_compiled, layout_dims


    def _unflatten_indices(flat_indices, shape):
        """Convert flat indices to multi-dimensional indices (row-major)."""
        multi = []
        rem = flat_indices
        for i in range(len(shape)):
            stride = 1
            for j in range(i + 1, len(shape)):
                stride *= shape[j]
            multi.append(torch.div(rem, stride, rounding_mode='trunc'))
            rem = rem % stride
        return multi

    def _flatten_indices(multi_dim, shape):
        """Convert multi-dimensional indices to flat index (row-major)."""
        flat = torch.zeros_like(multi_dim[0])
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            flat = flat + multi_dim[i] * stride
            stride *= shape[i]
        return flat

    def apply_layout_torch(tensor, layout_obj, shape, compiled=None):
        """Apply a layout transform to a PyTorch tensor using compiled arithmetic.

        Uses MLIR-lowered index expressions instead of dense permutation tables.
        The index computation is O(1) per element (arithmetic), not O(numel) storage.

        Parameters
        ----------
        tensor : torch.Tensor
        layout_obj : GroupBy or TileByLayout
        shape : tuple of int
        compiled : tuple, optional
            Pre-compiled (fwd_fn, inv_fns, layout_dims) from compile_layout_transform.
        """
        if compiled is None:
            compiled = compile_layout_transform(layout_obj, shape)
        fwd_fn, inv_fns, layout_dims = compiled

        numel = tensor.numel()
        flat = tensor.contiguous().view(-1)
        device = tensor.device

        dst_indices = torch.arange(numel, dtype=torch.long, device=device)
        multi_dim = _unflatten_indices(dst_indices, layout_dims)
        src_flat = fwd_fn(*multi_dim)

        return flat[src_flat].view(shape)


    def apply_inverse_layout_torch(tensor, layout_obj, shape, compiled=None):
        """Apply inverse layout transform using compiled arithmetic.

        Parameters
        ----------
        tensor : torch.Tensor
        layout_obj : GroupBy or TileByLayout
        shape : tuple of int
        compiled : tuple, optional
            Pre-compiled (fwd_fn, inv_fns, layout_dims) from compile_layout_transform.
        """
        if compiled is None:
            compiled = compile_layout_transform(layout_obj, shape)
        fwd_fn, inv_fns, layout_dims = compiled

        numel = tensor.numel()
        flat = tensor.contiguous().view(-1)
        device = tensor.device

        dst_indices = torch.arange(numel, dtype=torch.long, device=device)
        src_multi = [fn(dst_indices) for fn in inv_fns]
        src_flat = _flatten_indices(src_multi, layout_dims)

        return flat[src_flat].view(shape)
