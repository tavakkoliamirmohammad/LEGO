import ast
import inspect
import textwrap
import sympy as sp
import sys
import os
from .lego_python import LEGOPythonCodePrinter
from .lego import *


def jit(fn=None, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in Triton kernels.
    
    Evaluates LEGO/SymPy expressions at decoration time and replaces them
    with generated Triton code. Uses AST for proper multi-line handling.
    
    Usage:
        @lego.jit
        @triton.jit
        def kernel(...):
            s_pid = Symbol('pid')
            L_pid = OrderBy(...).TileBy(...)
            pid_m, pid_n = L_pid.inv(s_pid)
            offset = L_A[pid_m, s_k, :, :]
            ...
    """
    def decorator(fn):
        # If fn is wrapped by Triton, extract the original function
        original_fn = fn
        if hasattr(fn, 'fn'):  # JITFunction or Autotuner
            original_fn = fn.fn
            
        # Get source code from the original unwrapped function
        source = textwrap.dedent(inspect.getsource(original_fn))
        
        # Parse into AST
        tree = ast.parse(source)
        func_def = tree.body[0]
        
        # Remove all decorators from the function
        func_def.decorator_list = []
        
        # Override Symbol to add integer/positive assumptions (critical for simplification speed)
        def _symbol_with_assumptions(name, **kw):
            kw.setdefault('integer', True)
            kw.setdefault('positive', True)
            return sp.Symbol(name, **kw)
        
        # Build evaluation environment
        eval_env = {**original_fn.__globals__, 'sp': sp, 'Symbol': _symbol_with_assumptions}
        printer = LEGOPythonCodePrinter()
        
        # Track LEGO symbols: name -> generated code string
        lego_code = {}
        # Names that are compile-time only (Symbols, Layouts)
        compile_time_names = set()
        new_body = []
        
        for stmt in func_def.body:
            # Skip docstrings
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
                new_body.append(stmt)
                continue
            
            # Try to evaluate assignments
            if isinstance(stmt, ast.Assign):
                try:
                    code_str = ast.unparse(stmt.value)
                    val = eval(code_str, eval_env)
                    
                    # Single target assignment
                    if len(stmt.targets) == 1:
                        target = stmt.targets[0]
                        
                        if isinstance(target, ast.Name):
                            name = target.id
                            
                            if isinstance(val, sp.Symbol) and not isinstance(val, sp.Expr.__class__):
                                # Bare Symbol definition - compile-time only, remove
                                eval_env[name] = val
                                compile_time_names.add(name)
                                continue
                            
                            elif isinstance(val, LayoutBlock):
                                # Layout object - compile-time only, remove
                                eval_env[name] = val
                                compile_time_names.add(name)
                                continue
                            
                            elif isinstance(val, sp.Expr):
                                # SymPy expression - store for inlining at usage sites
                                # (don't emit as assignment - may reference loop vars like k)
                                simplified = sp.simplify(val)
                                code = printer.doprint(simplified)
                                lego_code[name] = code
                                eval_env[name] = val
                                continue
                            
                        elif isinstance(target, ast.Tuple):
                            # Tuple unpacking (e.g., pid_m, pid_n = L_pid.inv(s_pid))
                            var_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
                            
                            if isinstance(val, (tuple, list)) and all(isinstance(v, (sp.Expr, sp.Symbol)) for v in val):
                                # Tuple-unpacked SymPy vars (e.g. pid_m, pid_n)
                                # These are intermediates - emit as assignments AND store for inlining
                                for var_name, v in zip(var_names, val):
                                    if isinstance(v, sp.Expr):
                                        simplified = sp.simplify(v)
                                        code = printer.doprint(simplified)
                                        lego_code[var_name] = code
                                        
                                        # Emit as assignment (pid_m, pid_n are used by the kernel)
                                        new_stmt = ast.parse(f"{var_name} = {code}").body[0]
                                        ast.copy_location(new_stmt, stmt)
                                        new_body.append(new_stmt)
                                    
                                    # Replace with fresh Symbol so subsequent LEGO expressions
                                    # reference by name (e.g. offset_a uses "pid_m" not the full expr)
                                    eval_env[var_name] = _symbol_with_assumptions(var_name)
                                
                                continue
                    
                except Exception as e:
                    # Evaluation failed - this is a runtime statement, keep it
                    pass
            
            # For non-LEGO statements, replace any references to LEGO offset variables
            # that should be inlined
            stmt_code = ast.unparse(stmt)
            modified = False
            import re
            for name, code in lego_code.items():
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, stmt_code):
                    stmt_code = re.sub(pattern, f'({code})', stmt_code)
                    modified = True
            
            if modified:
                try:
                    new_stmts = ast.parse(stmt_code).body
                    for s in new_stmts:
                        ast.copy_location(s, stmt)
                    new_body.extend(new_stmts)
                except:
                    new_body.append(stmt)
            else:
                new_body.append(stmt)
        
        func_def.body = new_body
        ast.fix_missing_locations(tree)
        
        # Generate the source code
        new_source = ast.unparse(tree)
        
        # Write to file so Triton can inspect it
        temp_dir = "/tmp/lego_kernels"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{original_fn.__name__}_{id(original_fn)}.py")
        
        with open(temp_file, 'w') as f:
            f.write(new_source)
        
        # Compile and execute
        code_obj = compile(tree, filename=temp_file, mode='exec')
        namespace = original_fn.__globals__.copy()
        exec(code_obj, namespace)
        
        # Get the transformed function
        transformed_fn = namespace[original_fn.__name__]
        
        # Update filename for Triton's inspection
        transformed_fn.__code__ = transformed_fn.__code__.replace(co_filename=temp_file)
        
        # If input was wrapped by @triton.jit, re-wrap the transformed function
        if fn is not original_fn:
            import triton
            transformed_fn = triton.jit(transformed_fn)
        
        return transformed_fn
    
    if fn is not None:
        return decorator(fn)
    return decorator
