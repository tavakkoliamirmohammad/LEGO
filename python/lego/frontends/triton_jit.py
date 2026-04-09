import ast
import os
import sys
from dataclasses import dataclass
from functools import reduce

import sympy as sp

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import DSLAdapter, write_and_exec_temp_file
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# Block-ptr metadata (Triton-specific)
# ---------------------------------------------------------------------------

@dataclass
class BlockPtrInfo:
    """Structured metadata for generating ``tl.make_block_ptr()`` calls."""
    shape: tuple         # global tensor shape, e.g. (M, K)
    strides: tuple       # memory strides in elements, e.g. (K, 1) for Row
    offsets: tuple        # tile offsets, e.g. (pid_m * BM, k * BK)
    block_shape: tuple    # tile dimensions, e.g. (BM, BK)
    order: tuple          # memory layout order, e.g. (1, 0) for Row
    boundary_dims: tuple  # dimensions needing boundary_check


def extract_block_ptr_metadata(layout, subscript_indices):
    """Extract ``BlockPtrInfo`` from a ``TileByLayout``.

    Lowers the inner layout via MLIR to obtain a flat-index expression,
    then extracts stride coefficients.  Works for any strided (linear)
    layout — Row, Col, RegP, or multi-perm OrderBy — not just Row/Col.

    Requires exactly 2 tile groups (grid + block).  Returns ``None``
    for non-strided (non-linear) layouts or structural mismatches.
    """
    from lego.core import TileByLayout, OrderBy

    if not isinstance(layout, TileByLayout):
        return None

    # Must have exactly 2 tile groups (grid dims + block dims)
    if len(layout._tile_groups) != 2:
        return None

    # Build inner layout from the full chain of OrderBy objects
    all_perms = []
    for orderby in layout._input_chain:
        all_perms.extend(orderby.perms)
    inner = OrderBy(*all_perms)
    shape = tuple(inner.dims())
    ndim = len(shape)

    grid_shape = tuple(layout._tile_groups[0])
    block_shape = tuple(layout._tile_groups[1])

    if len(block_shape) != ndim or len(grid_shape) != ndim:
        return None

    # Parse subscript: expect [tile_idx, ..., :, :, ...]
    if not isinstance(subscript_indices, (list, tuple)):
        subscript_indices = [subscript_indices]

    tile_indices = []
    slice_count = 0
    for item in subscript_indices:
        if isinstance(item, slice):
            slice_count += 1
        else:
            tile_indices.append(item)

    if len(tile_indices) != ndim or slice_count != ndim:
        return None

    # Lower the inner layout via MLIR and extract strides + order
    strides, order = _extract_strides_from_lowered(inner, shape, ndim)
    if strides is None:
        return None

    offsets = tuple(tile_indices[i] * block_shape[i] for i in range(ndim))

    # Boundary dims: where grid * block != shape
    boundary_dims = []
    for i in range(ndim):
        diff = sp.simplify(grid_shape[i] * block_shape[i] - shape[i])
        if diff != 0:
            boundary_dims.append(i)

    return BlockPtrInfo(
        shape=shape, strides=strides, offsets=offsets,
        block_shape=block_shape, order=order,
        boundary_dims=tuple(boundary_dims),
    )


def _extract_strides_from_lowered(inner_layout, shape, ndim):
    """Lower *inner_layout* via MLIR and extract ``(strides, order)``.

    Returns ``(None, None)`` if the lowered expression is not a linear
    function of the indices (i.e. not a strided layout).
    """
    from lego.backend.symbolic import simplify_via_mlir

    idx_syms = sp.symbols(
        [f'_bp_idx_{k}' for k in range(ndim)],
        integer=True, positive=True,
    )

    # Constraints: index vars bounded by shape, dim symbols positive
    constraints = {}
    for k, s in enumerate(idx_syms):
        constraints[s] = (0, shape[k])
    for d in shape:
        if isinstance(d, sp.Symbol) and d not in constraints:
            constraints[d] = (1, None)

    # simplify_via_mlir requires layout._dims
    inner_layout._dims = shape
    try:
        flat_expr = simplify_via_mlir(
            inner_layout, 'apply', list(idx_syms), constraints)
    except Exception:
        return None, None
    finally:
        if hasattr(inner_layout, '_dims'):
            del inner_layout._dims

    # Extract coefficient of each index symbol
    expanded = sp.expand(flat_expr)
    strides = []
    for s in idx_syms:
        c = expanded.coeff(s)
        if c == 0:
            return None, None  # degenerate or non-linear dimension
        strides.append(c)

    # Verify linearity: remainder must be free of index symbols
    reconstructed = sum(c * s for c, s in zip(strides, idx_syms))
    remainder = sp.expand(expanded - reconstructed)
    if any(s in remainder.free_symbols for s in idx_syms):
        return None, None  # non-linear layout

    strides = tuple(strides)
    order = _infer_order_from_strides(strides, ndim)
    if order is None:
        return None, None

    return strides, order


def _infer_order_from_strides(strides, ndim):
    """Infer Triton memory order from symbolic strides.

    Returns dimension indices sorted innermost-first (ascending stride),
    or ``None`` if the ordering cannot be determined.
    """
    if ndim == 1:
        return (0,)

    # Collect free symbols across all strides
    free_syms = set()
    for s in strides:
        if isinstance(s, sp.Expr):
            free_syms |= s.free_symbols

    if not free_syms:
        # All strides are concrete — sort directly
        try:
            return tuple(sorted(range(ndim), key=lambda i: int(strides[i])))
        except (TypeError, ValueError):
            return None

    # Substitute dim symbols with distinct large primes to obtain a concrete
    # ordering that preserves symbolic multiplicative relationships.
    _PRIMES = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
               151, 157, 163, 167, 173, 179, 181, 191, 193, 197]
    sub = {sym: _PRIMES[i] for i, sym in enumerate(
        sorted(free_syms, key=lambda s: s.name))}

    concrete = []
    for s in strides:
        v = s.subs(sub) if isinstance(s, sp.Expr) else s
        try:
            concrete.append(int(v))
        except (TypeError, ValueError):
            return None

    return tuple(sorted(range(ndim), key=lambda i: concrete[i]))


# ---------------------------------------------------------------------------
# Triton-specific code printer
# ---------------------------------------------------------------------------

class TritonCodePrinter(LEGOPythonCodePrinter):
    """Renders ``lego_arange`` as ``tl.arange`` for Triton."""

    def _print_lego_arange(self, expr):
        return f"tl.arange({self._print(expr.args[0])}, {self._print(expr.args[1])})"


# ---------------------------------------------------------------------------
# Triton adapter
# ---------------------------------------------------------------------------

class TritonAdapter(DSLAdapter):
    # Triton builtins that take a pointer as first arg
    _POINTER_FUNCS = frozenset((
        'load', 'store', 'atomic_add', 'atomic_max', 'atomic_min',
        'atomic_and', 'atomic_or', 'atomic_xor', 'atomic_xchg', 'atomic_cas',
    ))

    def __init__(self, use_block_ptr=False):
        self.use_block_ptr = use_block_ptr

    def get_rewriter_options(self):
        return {'use_block_ptr': self.use_block_ptr}

    def unwrap(self, fn):
        original_fn = fn
        wrappers = []
        while hasattr(original_fn, 'fn'):
            wrappers.append(original_fn)
            original_fn = original_fn.fn

        source_fn = original_fn
        while hasattr(source_fn, 'src_fn'):
            source_fn = source_fn.src_fn

        return source_fn, original_fn, wrappers

    def find_runtime_vars(self, func_def):
        pointers = set()
        for n in ast.walk(func_def):
            if isinstance(n, ast.Call):
                func_name = ""
                if isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and n.func.value.id == 'tl':
                        func_name = n.func.attr
                elif isinstance(n.func, ast.Name):
                    func_name = n.func.id

                if func_name in self._POINTER_FUNCS:
                    if n.args and isinstance(n.args[0], ast.Name):
                        pointers.add(n.args[0].id)
        return pointers

    def get_code_printer(self):
        return TritonCodePrinter()

    def try_block_ptr_pattern(self, stmt, eval_env, printer):
        """Detect ``var = ptr + L[subscripts]`` and generate make_block_ptr.

        Returns ``(target_name, code_str, ptr_name, BlockPtrInfo)`` or ``None``.
        """
        from lego.rewriter import (
            _extract_subscript_indices, _format_make_block_ptr,
        )
        from lego.core import TileByLayout

        if not (isinstance(stmt.value, ast.BinOp)
                and isinstance(stmt.value.op, ast.Add)):
            return None

        left, right = stmt.value.left, stmt.value.right

        # Try both orderings: ptr + L[...] and L[...] + ptr
        ptr_node, subscript_node = None, None
        if isinstance(right, ast.Subscript):
            ptr_node, subscript_node = left, right
        elif isinstance(left, ast.Subscript):
            ptr_node, subscript_node = right, left
        else:
            return None

        # The subscript target must be a known TileByLayout
        if not isinstance(subscript_node.value, ast.Name):
            return None
        layout_name = subscript_node.value.id
        layout = eval_env.get(layout_name)
        if not isinstance(layout, TileByLayout):
            return None

        # Extract pointer variable name
        ptr_name = ast.unparse(ptr_node)

        # Extract and evaluate subscript indices
        indices = _extract_subscript_indices(subscript_node, eval_env)
        if indices is None:
            return None

        # Get structured block_ptr metadata (returns None for incompatible layouts)
        info = extract_block_ptr_metadata(layout, indices)
        if info is None:
            return None

        target_name = stmt.targets[0].id
        code = _format_make_block_ptr(ptr_name, info, printer)
        return (target_name, code, ptr_name, info)

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        result = write_and_exec_temp_file(
            new_source, tree, original_fn, return_source=return_source)

        if return_source:
            source_text, _ = result
            return source_text

        namespace, transformed_fn = result

        # Re-apply Triton wrappers in reverse order
        if wrappers:
            import triton
            from triton.runtime.jit import JITFunction
            from triton.runtime.autotuner import Autotuner
            for wrapper in reversed(wrappers):
                if isinstance(wrapper, Autotuner):
                    transformed_fn = triton.autotune(
                        configs=wrapper.configs,
                        key=wrapper.keys,
                    )(transformed_fn)
                elif isinstance(wrapper, JITFunction):
                    transformed_fn = triton.jit(transformed_fn)

        return transformed_fn


# ---------------------------------------------------------------------------
# Public API (unchanged)
# ---------------------------------------------------------------------------

def jit(fn=None, use_block_ptr=False, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in Triton kernels.

    Usage:
        @lego.jit
        @triton.jit
        def kernel(M, N, K, ...):
            ...

        @lego.jit(use_block_ptr=True)
        @triton.jit
        def kernel(M, N, K, ...):
            ...  # generates tl.make_block_ptr / tl.advance
    """
    def decorator(fn):
        return rewrite(fn, TritonAdapter(use_block_ptr=use_block_ptr), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator


def get_kernel_source(fn):
    """
    Utility function to retrieve the raw generated Triton source
    code for a @lego.jit decorated function without compiling it.
    """
    return jit(fn, return_source=True)
