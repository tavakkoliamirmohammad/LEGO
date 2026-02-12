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
        
        def _process_stmts(stmts):
            """Process a list of statements, evaluating LEGO expressions and keeping runtime code."""
            new_body = []
            
            for stmt in stmts:
                # Skip docstrings
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Str)):
                    new_body.append(stmt)
                    continue
                
                # Handle for loops - create symbol for loop var and process body
                if isinstance(stmt, ast.For):
                    if isinstance(stmt.target, ast.Name):
                        loop_var = stmt.target.id
                        eval_env[loop_var] = _symbol_with_assumptions(loop_var)
                    
                    # Process the loop body recursively
                    stmt.body = _process_stmts(stmt.body)
                    new_body.append(stmt)
                    continue
                
                # Try to evaluate assignments
                if isinstance(stmt, ast.Assign):
                    try:
                        # If Symbol() with no args, infer name from variable (strip s_ prefix)
                        value_node = stmt.value
                        if (isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name)
                                and value_node.func.id == 'Symbol' and not value_node.args):
                            target = stmt.targets[0]
                            if isinstance(target, ast.Name):
                                var_name = target.id
                                sym_name = var_name[2:] if var_name.startswith('s_') else var_name
                                value_node.args = [ast.Constant(value=sym_name)]
                        
                        code_str = ast.unparse(stmt.value)
                        # Suppress stdout during eval because triton.language functions 
                        # (like program_id) might print to stdout when called in eager mode
                        with contextlib.redirect_stdout(io.StringIO()):
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
                                    simplified = sp.simplify(val)
                                    code = printer.doprint(simplified)
                                    lego_code[name] = code
                                    eval_env[name] = val
                                    continue
                                
                                else:
                                    # Non-LEGO result (e.g., tl.program_id returns dict)
                                    # Keep as runtime, ensure symbol exists for this name
                                    eval_env[name] = _symbol_with_assumptions(name)
                                    # Fall through to keep statement
                                
                            elif isinstance(target, ast.Tuple):
                                # Tuple unpacking (e.g., pid_m, pid_n = L_pid.inv(pid))
                                var_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
                                
                                if isinstance(val, (tuple, list)) and all(isinstance(v, (sp.Expr, sp.Symbol)) for v in val):
                                    for var_name, v in zip(var_names, val):
                                        if isinstance(v, sp.Expr):
                                            simplified = sp.simplify(v)
                                            code = printer.doprint(simplified)
                                            lego_code[var_name] = code
                                            
                                            # Emit as assignment
                                            new_stmt = ast.parse(f"{var_name} = {code}").body[0]
                                            ast.copy_location(new_stmt, stmt)
                                            new_body.append(new_stmt)
                                        
                                        # Replace with fresh Symbol for subsequent expressions
                                        eval_env[var_name] = _symbol_with_assumptions(var_name)
                                    
                                    continue
                        
                    except Exception as e:
                        # Evaluation failed - this is a runtime statement
                        # Ensure symbols exist for target variables  
                        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                            target = stmt.targets[0]
                            if isinstance(target, ast.Name):
                                eval_env[target.id] = _symbol_with_assumptions(target.id)
                
                # For non-LEGO statements, replace any references to LEGO variables
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
            
            return new_body
        
        func_def.body = _process_stmts(func_def.body)
        ast.fix_missing_locations(tree)
        
        # Generate the source code
        new_source = ast.unparse(tree)
        
        # Write to file so Triton can inspect it
        _debug = os.environ.get('LEGO_DEBUG')
        _save = os.environ.get('LEGO_SAVE_KERNEL')
        temp_dir = "/tmp/lego_kernels"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"{original_fn.__name__}_{id(original_fn)}.py")
        
        with open(temp_file, 'w') as f:
            f.write(new_source)
        
        # Print generated source if LEGO_DEBUG is set
        if _debug:
            print(f"=== LEGO Generated Kernel ({temp_file}) ===")
            print(new_source)
            print("=== End Generated Kernel ===")
            if wrappers:
                for w in wrappers:
                    wtype = type(w).__name__
                    if hasattr(w, 'configs'):
                        print(f"  Re-applying @triton.autotune with {len(w.configs)} configs")
                    else:
                        print(f"  Re-applying @triton.{wtype.lower()}")
        
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
