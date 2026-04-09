"""DSL-agnostic LEGO AST rewriting engine.

Extracts layout expressions from user code, evaluates them symbolically via
MLIR, and emits the simplified result back into the AST.  All DSL-specific
behaviour (unwrapping decorators, identifying runtime variables, code
printing, and re-wrapping) is delegated to a ``DSLAdapter``.
"""

import ast
import contextlib
import inspect
import io
import sys
import textwrap


import sympy as sp

from lego.core import LayoutBlock
from lego.backend._ops import _LEGO_DEBUG
from lego.frontends._adapter import DSLAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symbol_with_assumptions(name, **kw):
    """Create a SymPy Symbol with integer/positive defaults."""
    kw.setdefault('integer', True)
    kw.setdefault('positive', True)
    return sp.Symbol(name, **kw)


def _parse_and_normalize(source_fn, original_fn):
    """Parse source into an AST and normalize line numbers.

    Returns (tree, func_def, source).
    """
    try:
        source = textwrap.dedent(inspect.getsource(source_fn))
    except TypeError:
        if hasattr(original_fn, 'src'):
            source = original_fn.src
        else:
            raise

    tree = ast.parse(source)
    func_def = tree.body[0]
    func_def.decorator_list = []

    lineno_offset = func_def.lineno - 1
    for node in ast.walk(tree):
        if hasattr(node, 'lineno'):
            node.lineno -= lineno_offset
        if hasattr(node, 'end_lineno') and node.end_lineno is not None:
            node.end_lineno -= lineno_offset

    return tree, func_def, source


def _build_eval_env(original_fn, func_def):
    """Build the evaluation namespace for kernel parameters."""
    eval_env = {**original_fn.__globals__, 'sp': sp, 'Symbol': _symbol_with_assumptions}
    for arg in func_def.args.args:
        eval_env[arg.arg] = _symbol_with_assumptions(arg.arg)
    return eval_env


# ---------------------------------------------------------------------------
# Block-ptr helpers
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
                    val = eval(code, eval_env)
                indices.append(val)
            except Exception:
                return None
    return indices


def _try_block_ptr_pattern(stmt, eval_env, printer, adapter):
    """Detect ``var = ptr + L[subscripts]`` and generate make_block_ptr.

    Delegates to *adapter.try_block_ptr_pattern()* for DSL-specific
    block-ptr metadata extraction.

    Returns ``(target_name, code_str, ptr_name, BlockPtrInfo)`` or ``None``.
    """
    return adapter.try_block_ptr_pattern(stmt, eval_env, printer)


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
                return eval(code, eval_env)
        except Exception:
            pass
    return sp.Integer(0)


def _transform_block_ptr_loop(for_stmt, block_ptr_vars, eval_env, printer):
    """Hoist block_ptr creation before loop and insert ``tl.advance`` at end.

    Returns ``(hoisted_stmts, new_loop_body)``.
    """
    from lego.frontends.triton_jit import BlockPtrInfo

    loop_var = for_stmt.target.id if isinstance(for_stmt.target, ast.Name) else None
    if not loop_var or not block_ptr_vars:
        return [], for_stmt.body

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
                # Initial offsets with loop_var = start
                initial_offsets = tuple(
                    sp.simplify(o.subs(loop_var_sym, loop_start))
                    if isinstance(o, sp.Expr) else o
                    for o in info.offsets
                )

                # Per-iteration advance delta
                deltas = []
                for o in info.offsets:
                    if isinstance(o, sp.Expr) and loop_var_sym in o.free_symbols:
                        deltas.append(sp.simplify(o.diff(loop_var_sym)))
                    else:
                        deltas.append(sp.Integer(0))

                # Hoist make_block_ptr before the loop
                initial_info = BlockPtrInfo(
                    shape=info.shape, strides=info.strides,
                    offsets=initial_offsets, block_shape=info.block_shape,
                    order=info.order, boundary_dims=info.boundary_dims)
                hoisted_code = _format_make_block_ptr(ptr_name, initial_info, printer)
                h_stmt = ast.parse(f"{var_name} = {hoisted_code}").body[0]
                ast.copy_location(h_stmt, stmt)
                hoisted.append(h_stmt)

                # tl.advance at end of the loop body
                delta_str = _format_tuple(deltas, printer)
                a_stmt = ast.parse(
                    f"{var_name} = tl.advance({var_name}, {delta_str})").body[0]
                ast.copy_location(a_stmt, stmt)
                advance_stmts.append(a_stmt)

                block_ptr_vars[var_name] = (ptr_name, initial_info)
                continue  # remove original assignment from loop body

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

        # First positional arg must be a tracked block_ptr variable
        if not node.args or not isinstance(node.args[0], ast.Name):
            return node
        var_name = node.args[0].id
        if var_name not in self.block_ptr_vars:
            return node

        _, info = self.block_ptr_vars[var_name]

        # Remove mask keyword argument
        node.keywords = [kw for kw in node.keywords if kw.arg != 'mask']

        # Add boundary_check if there are boundary dimensions
        if info.boundary_dims:
            boundary_str = ', '.join(str(d) for d in info.boundary_dims)
            boundary_node = ast.parse(f"({boundary_str},)").body[0].value
            node.keywords.append(
                ast.keyword(arg='boundary_check', value=boundary_node))

        return node


# ---------------------------------------------------------------------------
# AST Transformer
# ---------------------------------------------------------------------------

class LEGOASTTransformer(ast.NodeTransformer):
    def __init__(self, lego_code, eval_env, printer):
        self.lego_code = lego_code
        self.eval_env = eval_env
        self.printer = printer

    def visit_Name(self, node):
        if node.id in self.lego_code:
            code = self.lego_code[node.id]
            return ast.parse(f"({code})").body[0].value
        return node

    def visit_Subscript(self, node):
        try:
            s = ast.unparse(node)
            with contextlib.redirect_stdout(io.StringIO()):
                val = eval(s, self.eval_env)

            if isinstance(val, (sp.Expr, sp.Symbol)):
                simplified = sp.simplify(val)
                code = self.printer.doprint(simplified)
                return ast.parse(f"({code})").body[0].value
        except Exception:
            pass
        return self.generic_visit(node)


# ---------------------------------------------------------------------------
# Statement processor
# ---------------------------------------------------------------------------

def _process_stmts(stmts, lego_code, eval_env, printer, runtime_vars,
                   compile_time_names, opts=None, block_ptr_vars=None,
                   adapter=None):
    if opts is None:
        opts = {}
    if block_ptr_vars is None:
        block_ptr_vars = {}

    use_block_ptr = opts.get('use_block_ptr', False)

    new_body = []
    transformer = LEGOASTTransformer(lego_code, eval_env, printer)

    for stmt in stmts:
        # 1. Skip docstrings/constants
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            new_body.append(stmt)
            continue

        # 2. For loops
        if isinstance(stmt, ast.For):
            if isinstance(stmt.target, ast.Name):
                loop_var = stmt.target.id
                eval_env[loop_var] = _symbol_with_assumptions(loop_var)

            if use_block_ptr:
                loop_bpv = {}
                stmt.body = _process_stmts(
                    stmt.body, lego_code, eval_env, printer,
                    runtime_vars, compile_time_names, opts, loop_bpv,
                    adapter)
                hoisted, stmt.body = _transform_block_ptr_loop(
                    stmt, loop_bpv, eval_env, printer)
                new_body.extend(hoisted)
                block_ptr_vars.update(loop_bpv)
            else:
                stmt.body = _process_stmts(stmt.body, lego_code, eval_env,
                                           printer, runtime_vars,
                                           compile_time_names, opts,
                                           block_ptr_vars, adapter)
            new_body.append(stmt)
            continue

        # 3. While loops
        if isinstance(stmt, ast.While):
            stmt.test = transformer.visit(stmt.test)
            stmt.body = _process_stmts(stmt.body, lego_code, eval_env,
                                       printer, runtime_vars,
                                       compile_time_names, opts,
                                       block_ptr_vars, adapter)
            new_body.append(stmt)
            continue

        # 4. If blocks
        if isinstance(stmt, ast.If):
            stmt.test = transformer.visit(stmt.test)
            stmt.body = _process_stmts(stmt.body, lego_code, eval_env,
                                       printer, runtime_vars,
                                       compile_time_names, opts,
                                       block_ptr_vars, adapter)
            stmt.orelse = _process_stmts(stmt.orelse, lego_code, eval_env,
                                         printer, runtime_vars,
                                         compile_time_names, opts,
                                         block_ptr_vars, adapter)
            new_body.append(stmt)
            continue

        # 5. Assignments
        if isinstance(stmt, ast.Assign):
            value_node = stmt.value
            if (isinstance(value_node, ast.Call)
                    and isinstance(value_node.func, ast.Name)
                    and value_node.func.id == 'Symbol'
                    and not value_node.args):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    var_name = stmt.targets[0].id
                    sym_name = var_name[2:] if var_name.startswith('s_') else var_name
                    value_node.args = [ast.Constant(value=sym_name)]

            # 5a. Block-ptr pattern: var = ptr + L[subscripts]
            #     Must be checked BEFORE runtime-vars skip, because pointer
            #     variables (used in tl.load/tl.store) are runtime vars.
            if use_block_ptr and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                bp = _try_block_ptr_pattern(stmt, eval_env, printer, adapter)
                if bp is not None:
                    target_name, code, ptr_name, info = bp
                    new_stmt = ast.parse(f"{target_name} = {code}").body[0]
                    ast.copy_location(new_stmt, stmt)
                    new_body.append(new_stmt)
                    eval_env[target_name] = _symbol_with_assumptions(target_name)
                    block_ptr_vars[target_name] = (ptr_name, info)
                    continue

            # Skip evaluation for runtime variables
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
                if target_name in runtime_vars:
                    stmt.value = transformer.visit(stmt.value)
                    new_body.append(stmt)
                    eval_env[target_name] = _symbol_with_assumptions(target_name)
                    continue

            try:
                code_str = ast.unparse(stmt.value)
                with contextlib.redirect_stdout(io.StringIO()):
                    val = eval(code_str, eval_env)

                # Single target
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id

                    if isinstance(val, LayoutBlock):
                        eval_env[name] = val
                        compile_time_names.add(name)
                        continue
                    elif isinstance(val, sp.Expr):
                        simplified = sp.simplify(val)
                        code = printer.doprint(simplified)
                        if code != name:
                            new_stmt = ast.parse(f"{name} = {code}").body[0]
                            ast.copy_location(new_stmt, stmt)
                            new_body.append(new_stmt)
                            eval_env[name] = _symbol_with_assumptions(name)
                        else:
                            eval_env[name] = val
                        continue
                    else:
                        eval_env[name] = _symbol_with_assumptions(name)
                        new_body.append(stmt)
                        continue

                # Tuple unpacking
                elif isinstance(stmt.targets[0], ast.Tuple):
                    target = stmt.targets[0]
                    var_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]

                    if isinstance(val, (tuple, list)) and len(val) == len(var_names):
                        ALL_LEGO = all(isinstance(v, (sp.Expr, sp.Symbol)) for v in val)
                        if ALL_LEGO:
                            for var_name, v in zip(var_names, val):
                                if isinstance(v, sp.Expr):
                                    simplified = sp.simplify(v)
                                    code = printer.doprint(simplified)
                                    new_stmt = ast.parse(f"{var_name} = {code}").body[0]
                                    ast.copy_location(new_stmt, stmt)
                                    new_body.append(new_stmt)
                                eval_env[var_name] = _symbol_with_assumptions(var_name)
                            continue
            except Exception:
                pass

            # Fallthrough — runtime assignment
            stmt.value = transformer.visit(stmt.value)

            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    eval_env[target.id] = _symbol_with_assumptions(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            eval_env[elt.id] = _symbol_with_assumptions(elt.id)

            new_body.append(stmt)
            continue

        # 6. Other statements
        new_stmt = transformer.visit(stmt)
        new_body.append(new_stmt)

    return new_body


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def rewrite(fn, adapter: DSLAdapter, **kwargs):
    """Rewrite *fn* using LEGO layout algebra, delegating DSL specifics to *adapter*.

    Parameters
    ----------
    fn : callable
        The user function (possibly wrapped by DSL decorators).
    adapter : DSLAdapter
        Provides DSL-specific unwrapping, runtime-var detection, printing,
        and re-wrapping.
    **kwargs
        Forwarded to the adapter's ``compile_and_wrap``.
    """
    source_fn, original_fn, wrappers = adapter.unwrap(fn)

    tree, func_def, _source = _parse_and_normalize(source_fn, original_fn)
    eval_env = _build_eval_env(original_fn, func_def)
    printer = adapter.get_code_printer()
    runtime_vars = adapter.find_runtime_vars(func_def)

    opts = adapter.get_rewriter_options()
    block_ptr_vars = {}

    lego_code = {}
    compile_time_names = set()

    func_def.body = _process_stmts(
        func_def.body, lego_code, eval_env, printer,
        runtime_vars, compile_time_names, opts, block_ptr_vars,
        adapter,
    )

    # Post-process: rewrite tl.load/tl.store for block_ptr variables
    if block_ptr_vars:
        rewriter = _BlockPtrLoadRewriter(block_ptr_vars)
        for i, stmt in enumerate(func_def.body):
            func_def.body[i] = rewriter.visit(stmt)

    ast.fix_missing_locations(tree)

    new_source = ast.unparse(tree)

    if _LEGO_DEBUG:
        print("=== LEGO Generated Kernel ===", file=sys.stderr)
        print(new_source, file=sys.stderr)
        print("=== End Generated Kernel ===", file=sys.stderr)

    return adapter.compile_and_wrap(
        new_source, tree, original_fn, wrappers,
        return_source=kwargs.get('return_source', False),
    )
