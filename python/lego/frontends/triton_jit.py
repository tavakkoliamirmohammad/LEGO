import ast
import contextlib
import io
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
# Block-ptr code generation helpers (Triton-specific)
# ---------------------------------------------------------------------------

def _format_tuple(items, printer):
    """Format a sequence of SymPy expressions / ints as a Python tuple string."""
    parts = []
    for item in items:
        if isinstance(item, (sp.Expr, sp.Symbol)):
            parts.append(printer.doprint(item))
        else:
            parts.append(str(item))
    if len(parts) == 1:
        return f"({parts[0]},)"
    return f"({', '.join(parts)})"


def _format_make_block_ptr(ptr_name, info, printer):
    """Generate tl.make_block_ptr(...) code string from BlockPtrInfo."""
    return (
        f"tl.make_block_ptr(base={ptr_name}, "
        f"shape={_format_tuple(info.shape, printer)}, "
        f"strides={_format_tuple(info.strides, printer)}, "
        f"offsets={_format_tuple(info.offsets, printer)}, "
        f"block_shape={_format_tuple(info.block_shape, printer)}, "
        f"order={_format_tuple(info.order, printer)})"
    )


def _extract_subscript_indices(subscript_node, eval_env):
    """Extract subscript indices from an AST Subscript node.

    Returns a list of SymPy expressions and ``slice`` objects, or ``None``
    if parsing fails.
    """
    slice_node = subscript_node.slice
    elements = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]

    indices = []
    for elt in elements:
        if isinstance(elt, ast.Slice):
            indices.append(slice(None))
        else:
            try:
                code = ast.unparse(elt)
                with contextlib.redirect_stdout(io.StringIO()):
                    val = eval(code, eval_env)  # noqa: S307
                indices.append(val)
            except Exception:
                return None
    return indices


def _extract_loop_start(for_stmt, eval_env):
    """Return the start value of a ``for x in range(start, ...)`` loop."""
    if not isinstance(for_stmt.iter, ast.Call):
        return sp.Integer(0)
    call = for_stmt.iter
    is_range = isinstance(call.func, ast.Name) and call.func.id == 'range'
    if is_range and len(call.args) >= 2:
        try:
            code = ast.unparse(call.args[0])
            with contextlib.redirect_stdout(io.StringIO()):
                return eval(code, eval_env)  # noqa: S307
        except Exception:
            pass
    return sp.Integer(0)


def _transform_block_ptr_loop(for_stmt, block_ptr_vars, eval_env, printer):
    """Hoist block_ptr creation before loop and insert ``tl.advance`` at end.

    Returns ``(hoisted_stmts, new_loop_body)`` or ``None``.
    """
    loop_var = for_stmt.target.id if isinstance(for_stmt.target, ast.Name) else None
    if not loop_var or not block_ptr_vars:
        return None

    loop_var_sym = eval_env.get(loop_var)
    loop_start = _extract_loop_start(for_stmt, eval_env)

    hoisted = []
    new_body = []
    advance_stmts = []

    for stmt in for_stmt.body:
        var_name = None
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            var_name = stmt.targets[0].id

        if var_name and var_name in block_ptr_vars:
            ptr_name, info = block_ptr_vars[var_name]

            has_loop_dep = any(
                isinstance(o, sp.Expr) and loop_var_sym in o.free_symbols
                for o in info.offsets
            )

            if has_loop_dep:
                initial_offsets = tuple(
                    sp.simplify(o.subs(loop_var_sym, loop_start))
                    if isinstance(o, sp.Expr) else o
                    for o in info.offsets
                )

                deltas = []
                for o in info.offsets:
                    if isinstance(o, sp.Expr) and loop_var_sym in o.free_symbols:
                        deltas.append(sp.simplify(o.diff(loop_var_sym)))
                    else:
                        deltas.append(sp.Integer(0))

                initial_info = BlockPtrInfo(
                    shape=info.shape, strides=info.strides,
                    offsets=initial_offsets, block_shape=info.block_shape,
                    order=info.order, boundary_dims=info.boundary_dims)
                hoisted_code = _format_make_block_ptr(ptr_name, initial_info, printer)
                h_stmt = ast.parse(f"{var_name} = {hoisted_code}").body[0]
                ast.copy_location(h_stmt, stmt)
                hoisted.append(h_stmt)

                delta_str = _format_tuple(deltas, printer)
                a_stmt = ast.parse(
                    f"{var_name} = tl.advance({var_name}, {delta_str})").body[0]
                ast.copy_location(a_stmt, stmt)
                advance_stmts.append(a_stmt)

                block_ptr_vars[var_name] = (ptr_name, initial_info)
                continue

        new_body.append(stmt)

    new_body.extend(advance_stmts)
    return hoisted, new_body


class _BlockPtrLoadRewriter(ast.NodeTransformer):
    """Rewrite ``tl.load``/``tl.store`` for block_ptr variables.

    Removes ``mask=`` and adds ``boundary_check=`` when needed.
    """

    def __init__(self, block_ptr_vars):
        self.block_ptr_vars = block_ptr_vars

    def visit_Call(self, node):
        self.generic_visit(node)

        if not isinstance(node.func, ast.Attribute):
            return node
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == 'tl'):
            return node
        if node.func.attr not in ('load', 'store'):
            return node

        if not node.args or not isinstance(node.args[0], ast.Name):
            return node
        var_name = node.args[0].id
        if var_name not in self.block_ptr_vars:
            return node

        _, info = self.block_ptr_vars[var_name]

        node.keywords = [kw for kw in node.keywords if kw.arg != 'mask']

        if info.boundary_dims:
            boundary_str = ', '.join(str(d) for d in info.boundary_dims)
            boundary_node = ast.parse(f"({boundary_str},)").body[0].value
            node.keywords.append(
                ast.keyword(arg='boundary_check', value=boundary_node))

        return node


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
        self._block_ptr_vars = {}

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

    def transform_assignment(self, stmt, eval_env, printer):
        if not self.use_block_ptr:
            return None
        return self._try_block_ptr_pattern(stmt, eval_env, printer)

    def _try_block_ptr_pattern(self, stmt, eval_env, printer):
        """Detect ``var = ptr + L[subscripts]`` and generate make_block_ptr.

        Returns ``(ast_node, eval_updates)`` or ``None``.
        """
        from lego.core import TileByLayout

        if not (isinstance(stmt.value, ast.BinOp)
                and isinstance(stmt.value.op, ast.Add)):
            return None

        left, right = stmt.value.left, stmt.value.right

        ptr_node, subscript_node = None, None
        if isinstance(right, ast.Subscript):
            ptr_node, subscript_node = left, right
        elif isinstance(left, ast.Subscript):
            ptr_node, subscript_node = right, left
        else:
            return None

        if not isinstance(subscript_node.value, ast.Name):
            return None
        layout_name = subscript_node.value.id
        layout = eval_env.get(layout_name)
        if not isinstance(layout, TileByLayout):
            return None

        ptr_name = ast.unparse(ptr_node)

        indices = _extract_subscript_indices(subscript_node, eval_env)
        if indices is None:
            return None

        info = extract_block_ptr_metadata(layout, indices)
        if info is None:
            return None

        target_name = stmt.targets[0].id
        code = _format_make_block_ptr(ptr_name, info, printer)
        new_node = ast.parse(f"{target_name} = {code}").body[0]
        self._block_ptr_vars[target_name] = (ptr_name, info)

        updates = {target_name: sp.Symbol(target_name, integer=True, positive=True)}
        return new_node, updates

    def transform_for_loop(self, for_stmt, body, eval_env, printer):
        if not self.use_block_ptr or not self._block_ptr_vars:
            return None
        return _transform_block_ptr_loop(for_stmt, self._block_ptr_vars,
                                         eval_env, printer)

    def post_process_body(self, func_def):
        if not self._block_ptr_vars:
            return
        rewriter = _BlockPtrLoadRewriter(self._block_ptr_vars)
        for i, stmt in enumerate(func_def.body):
            func_def.body[i] = rewriter.visit(stmt)

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
