import ast
import inspect
import textwrap
import sympy as sp
import sys
import os
import contextlib
import io
from .lego_python import LEGOPythonCodePrinter
from .lego import *


def jit(fn=None, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in Triton kernels.

    This decorator uses AST transformation to evaluate LEGO/SymPy expressions
    at decoration time and replaces them with generated Triton code.

    Features:
    - Auto-symbolization: Kernel arguments and loop variables are treated
      as SymPy symbols for layout calculations.
    - Layout Algebra: Supports defining and operating on Layout objects.
    - Code Generation: Simplifies and generates efficient pointer arithmetic.

    Usage:
        @lego.jit
        @triton.jit
        def kernel(M, N, K, ...):
            # M, N, K are implicitly symbols

            # Runtime variables (pid) are tracked as symbols for layout math
            pid = tl.program_id(0)

            # Define Layouts
            L_pid = OrderBy(...).TileBy(...)

            # Layout operations (Compile-time evaluation)
            pid_m, pid_n = L_pid.inv(pid)

            # Generated offsets
            offset = L_A[pid_m, s_k, :, :]
            ...
    """
    def decorator(fn):
        # Unwrap through Triton decorator layers (Autotuner -> JITFunction -> function)
        original_fn = fn
        wrappers = []  # Track wrappers to re-apply later
        while hasattr(original_fn, 'fn'):
            wrappers.append(original_fn)
            original_fn = original_fn.fn
            
        # Get source code from the original unwrapped function
        source = textwrap.dedent(inspect.getsource(original_fn))
        
        # Parse into AST
        tree = ast.parse(source)
        func_def = tree.body[0]
        
        # Remove all decorators from the function
        func_def.decorator_list = []
        
        # Normalize line numbers to 1-based for the new file
        lineno_offset = func_def.lineno - 1
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                node.lineno -= lineno_offset
            if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                node.end_lineno -= lineno_offset
        
        # Override Symbol to add integer/positive assumptions (critical for simplification speed)
        def _symbol_with_assumptions(name, **kw):
            kw.setdefault('integer', True)
            kw.setdefault('positive', True)
            return sp.Symbol(name, **kw)
        
        # Build evaluation environment
        eval_env = {**original_fn.__globals__, 'sp': sp, 'Symbol': _symbol_with_assumptions}
        printer = LEGOPythonCodePrinter()
        
        # Auto-create SymPy symbols for all kernel parameters
        for arg in func_def.args.args:
            param_name = arg.arg
            eval_env[param_name] = _symbol_with_assumptions(param_name)
        
        # Track LEGO symbols: name -> generated code string
        lego_code = {}
        # Names that are compile-time only (Symbols, Layouts)
        compile_time_names = set()
        
        # Helper to find variables used as pointers in tl.load/store/atomic
        def _find_pointer_vars(node):
            pointers = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    func_name = ""
                    if isinstance(n.func, ast.Attribute):
                        if isinstance(n.func.value, ast.Name) and n.func.value.id == 'tl':
                            func_name = n.func.attr
                    elif isinstance(n.func, ast.Name):
                         func_name = n.func.id
                    
                    if func_name in ('load', 'store', 'atomic_add', 'atomic_max', 'atomic_min', 
                                     'atomic_and', 'atomic_or', 'atomic_xor', 'atomic_xchg', 'atomic_cas'):
                        if n.args and isinstance(n.args[0], ast.Name):
                            pointers.add(n.args[0].id)
            return pointers

        runtime_pointers = _find_pointer_vars(func_def)
        
        class LEGOASTTransformer(ast.NodeTransformer):
            def __init__(self, lego_code, eval_env, printer):
                self.lego_code = lego_code
                self.eval_env = eval_env
                self.printer = printer

            def visit_Name(self, node):
                if node.id in self.lego_code:
                    # Replace LEGO variable with its generated code
                    code = self.lego_code[node.id]
                    return ast.parse(f"({code})").body[0].value
                return node

            def visit_Subscript(self, node):
                # Try to evaluate subscript (e.g. L_A[...]) to see if it reduces to a LEGO expression
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

        def _process_stmts(stmts):
            new_body = []
            transformer = LEGOASTTransformer(lego_code, eval_env, printer)
            
            for stmt in stmts:
                # 1. Skip docstrings/constants
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
                    new_body.append(stmt)
                    continue
                
                # 2. Handle For Loops
                if isinstance(stmt, ast.For):
                    if isinstance(stmt.target, ast.Name):
                        loop_var = stmt.target.id
                        eval_env[loop_var] = _symbol_with_assumptions(loop_var)
                    stmt.body = _process_stmts(stmt.body)
                    new_body.append(stmt)
                    continue
                
                # 3. Handle While Loops
                if isinstance(stmt, ast.While):
                    # Replace LEGO vars in the test
                    stmt.test = transformer.visit(stmt.test)
                    # Recurse into body
                    stmt.body = _process_stmts(stmt.body)
                    new_body.append(stmt)
                    continue

                # 4. Handle If blocks
                if isinstance(stmt, ast.If):
                    # Replace LEGO vars in the test
                    stmt.test = transformer.visit(stmt.test)
                    # Recurse into bodies
                    stmt.body = _process_stmts(stmt.body)
                    stmt.orelse = _process_stmts(stmt.orelse)
                    new_body.append(stmt)
                    continue
                
                # 5. Handle Assignments
                if isinstance(stmt, ast.Assign):
                    # Check for explicit Symbol declaration
                    value_node = stmt.value
                    if (isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name)
                            and value_node.func.id == 'Symbol' and not value_node.args):
                        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                            var_name = stmt.targets[0].id
                            sym_name = var_name[2:] if var_name.startswith('s_') else var_name
                            value_node.args = [ast.Constant(value=sym_name)]

                    # Skip evaluation for runtime pointers
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        target_name = stmt.targets[0].id
                        if target_name in runtime_pointers:
                            # Transform RHS to resolve any LEGO vars, but don't eval the whole assignment
                            stmt.value = transformer.visit(stmt.value)
                            new_body.append(stmt)
                            # Ensure we have a symbol for this pointer
                            eval_env[target_name] = _symbol_with_assumptions(target_name)
                            continue

                    try:
                        # Try to evaluate
                        code_str = ast.unparse(stmt.value)
                        with contextlib.redirect_stdout(io.StringIO()):
                            val = eval(code_str, eval_env)
                        
                        # Handle Single Target
                        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                            name = stmt.targets[0].id
                            
                            if isinstance(val, LayoutBlock):
                                # Pure layout object -> remove stmt, track in env for compile-time use
                                eval_env[name] = val
                                compile_time_names.add(name)
                                continue
                            elif isinstance(val, sp.Expr):
                                # Compile-time expression (includes sp.Symbol) -> Emit assignment if meaningful
                                simplified = sp.simplify(val)
                                code = printer.doprint(simplified)
                                if code != name:
                                    new_stmt = ast.parse(f"{name} = {code}").body[0]
                                    ast.copy_location(new_stmt, stmt)
                                    new_body.append(new_stmt)
                                    # Subsequent expressions should use the variable name
                                    eval_env[name] = _symbol_with_assumptions(name)
                                else:
                                    # Self-alias (e.g. n_rows = n_rows symbol) -> just track in env
                                    eval_env[name] = val
                                continue
                            else:
                                # Runtime result (e.g. tl.program_id) -> Keep stmt
                                eval_env[name] = _symbol_with_assumptions(name)
                                new_body.append(stmt)
                                continue

                        # Handle Tuple Unpacking
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
                                            # Emit assignment, do NOT inline
                                            new_stmt = ast.parse(f"{var_name} = {code}").body[0]
                                            ast.copy_location(new_stmt, stmt)
                                            new_body.append(new_stmt)
                                        eval_env[var_name] = _symbol_with_assumptions(var_name)
                                    continue
                    except Exception:
                        # Evaluation failed -> Runtime Statement
                        pass

                    # If we reached here (Exception or Fallthrough), it's a runtime assignment
                    # Transform RHS and keep it
                    stmt.value = transformer.visit(stmt.value)
                    
                    # Ensure symbols exist for targets
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            eval_env[target.id] = _symbol_with_assumptions(target.id)
                        elif isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    eval_env[elt.id] = _symbol_with_assumptions(elt.id)
                    
                    new_body.append(stmt)
                    continue

                # 6. Other Statements
                # Transform to replace LEGO vars
                new_stmt = transformer.visit(stmt)
                new_body.append(new_stmt)
            
            return new_body
        
        func_def.body = _process_stmts(func_def.body)
        ast.fix_missing_locations(tree)
        
        # Generate the source code
        new_source = ast.unparse(tree)
        
        # Write to file so Triton can inspect it
        _debug = os.environ.get('LEGO_DEBUG', False)
        _save = os.environ.get('LEGO_SAVE_KERNEL', False)
        temp_dir = os.environ.get("LEGO_TEMP_DIR", "/tmp/lego_kernels")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{original_fn.__name__}_{id(original_fn)}.py")
        
        with open(temp_file, 'w') as f:
            f.write(new_source)
        
        # Print generated source if LEGO_DEBUG is set
        if _debug:
            print(f"=== LEGO Generated Kernel ({temp_file}) ===")
            print(new_source)
            print("=== End Generated Kernel ===")
        
        # Compile and execute
        code_obj = compile(tree, filename=temp_file, mode='exec')
        namespace = original_fn.__globals__.copy()
        exec(code_obj, namespace)
        
        # Triton reads source lazily via inspect.getsource(), so file must persist.
        # Register cleanup at exit unless LEGO_SAVE_KERNEL is set.
        if not _save:
            import atexit
            atexit.register(lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)
        
        # Get the transformed function
        transformed_fn = namespace[original_fn.__name__]
        
        # Update filename for Triton's inspection
        transformed_fn.__code__ = transformed_fn.__code__.replace(co_filename=temp_file)
        
        # Re-apply Triton wrappers in reverse order (innermost first)
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
    
    if fn is not None:
        return decorator(fn)
    return decorator
