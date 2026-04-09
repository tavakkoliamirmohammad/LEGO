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
                   compile_time_names, adapter):
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

            stmt.body = _process_stmts(stmt.body, lego_code, eval_env,
                                       printer, runtime_vars,
                                       compile_time_names, adapter)
            result = adapter.transform_for_loop(stmt, stmt.body, eval_env,
                                                printer)
            if result is not None:
                hoisted, stmt.body = result
                new_body.extend(hoisted)
            new_body.append(stmt)
            continue

        # 3. While loops
        if isinstance(stmt, ast.While):
            stmt.test = transformer.visit(stmt.test)
            stmt.body = _process_stmts(stmt.body, lego_code, eval_env,
                                       printer, runtime_vars,
                                       compile_time_names, adapter)
            new_body.append(stmt)
            continue

        # 4. If blocks
        if isinstance(stmt, ast.If):
            stmt.test = transformer.visit(stmt.test)
            stmt.body = _process_stmts(stmt.body, lego_code, eval_env,
                                       printer, runtime_vars,
                                       compile_time_names, adapter)
            stmt.orelse = _process_stmts(stmt.orelse, lego_code, eval_env,
                                         printer, runtime_vars,
                                         compile_time_names, adapter)
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

            # 5a. Adapter-specific assignment transform.
            #     Must be checked BEFORE runtime-vars skip, because pointer
            #     variables (used in tl.load/tl.store) are runtime vars.
            if len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                result = adapter.transform_assignment(stmt, eval_env, printer)
                if result is not None:
                    new_node, updates = result
                    ast.copy_location(new_node, stmt)
                    new_body.append(new_node)
                    eval_env.update(updates)
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
                    val = eval(code_str, eval_env)  # noqa: S307

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

    lego_code = {}
    compile_time_names = set()

    func_def.body = _process_stmts(
        func_def.body, lego_code, eval_env, printer,
        runtime_vars, compile_time_names, adapter,
    )

    adapter.post_process_body(func_def)

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
