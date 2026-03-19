import ast
import atexit
import os
import sys

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


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

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        _save = os.environ.get('LEGO_SAVE_KERNEL', False)
        temp_dir = os.environ.get("LEGO_TEMP_DIR", "/tmp/lego_kernels")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(
            temp_dir, f"{original_fn.__name__}_{id(original_fn)}.py")

        # Write to file so Triton can use inspect.getsource()
        with open(temp_file, 'w') as f:
            f.write(new_source)

        if return_source:
            if not _save:
                os.remove(temp_file)
            return new_source

        # Compile and execute
        code_obj = compile(tree, filename=temp_file, mode='exec')
        namespace = original_fn.__globals__.copy()
        exec(code_obj, namespace)

        # Register cleanup at exit unless LEGO_SAVE_KERNEL is set
        if not _save:
            atexit.register(
                lambda f=temp_file: os.remove(f) if os.path.exists(f) else None)

        transformed_fn = namespace[original_fn.__name__]
        transformed_fn.__code__ = transformed_fn.__code__.replace(
            co_filename=temp_file)

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

def jit(fn=None, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in Triton kernels.

    Usage:
        @lego.jit
        @triton.jit
        def kernel(M, N, K, ...):
            ...
    """
    def decorator(fn):
        return rewrite(fn, TritonAdapter(), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator


def get_kernel_source(fn):
    """
    Utility function to retrieve the raw generated Triton source
    code for a @lego.jit decorated function without compiling it.
    """
    return jit(fn, return_source=True)
