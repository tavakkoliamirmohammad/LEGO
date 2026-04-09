import ast
import os
import sys

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import DSLAdapter, write_and_exec_temp_file
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# cuTile-specific code printer
# ---------------------------------------------------------------------------

class CutileCodePrinter(LEGOPythonCodePrinter):
    """Renders ``lego_arange`` as ``ct.arange`` for cuTile.

    cuTile's ``ct.arange(n, dtype=)`` takes a single size and returns
    ``[0, n)``, unlike Triton's ``tl.arange(start, stop)``.
    """

    def _print_lego_arange(self, expr):
        start, stop = expr.args[0], expr.args[1]
        if int(start) == 0:
            return f"ct.arange({self._print(stop)}, dtype=ct.int32)"
        return (f"(ct.arange({self._print(stop - start)}, dtype=ct.int32)"
                f" + {self._print(start)})")


# ---------------------------------------------------------------------------
# cuTile adapter
# ---------------------------------------------------------------------------

class CutileAdapter(DSLAdapter):
    # cuTile builtins that take a tensor as first arg
    _TENSOR_FUNCS = frozenset((
        'load', 'store', 'gather', 'scatter',
        'atomic_add', 'atomic_cas', 'atomic_xchg',
    ))

    def unwrap(self, fn):
        original_fn = fn
        wrappers = []

        # Strategy 1: _pyfunc (cuda.tile.kernel stores the raw function here)
        if hasattr(original_fn, '_pyfunc'):
            wrappers.append(original_fn)
            original_fn = original_fn._pyfunc

        # Strategy 2: walk .fn chain (Triton-style)
        if not wrappers:
            while hasattr(original_fn, 'fn'):
                wrappers.append(original_fn)
                original_fn = original_fn.fn

        # Strategy 3: py_func (Numba-style)
        if not wrappers and hasattr(original_fn, 'py_func'):
            wrappers.append(original_fn)
            original_fn = original_fn.py_func

        # Strategy 4: __wrapped__ (functools-style)
        if not wrappers and hasattr(original_fn, '__wrapped__'):
            wrappers.append(original_fn)
            original_fn = original_fn.__wrapped__

        source_fn = original_fn
        while hasattr(source_fn, 'src_fn'):
            source_fn = source_fn.src_fn

        return source_fn, original_fn, wrappers

    def find_runtime_vars(self, func_def):
        tensors = set()
        for n in ast.walk(func_def):
            if isinstance(n, ast.Call):
                func_name = ""
                if isinstance(n.func, ast.Attribute):
                    if isinstance(n.func.value, ast.Name) and n.func.value.id == 'ct':
                        func_name = n.func.attr
                elif isinstance(n.func, ast.Name):
                    func_name = n.func.id

                if func_name in self._TENSOR_FUNCS:
                    if n.args and isinstance(n.args[0], ast.Name):
                        tensors.add(n.args[0].id)
        return tensors

    def get_code_printer(self):
        return CutileCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        result = write_and_exec_temp_file(
            new_source, tree, original_fn, return_source=return_source)

        if return_source:
            source_text, _ = result
            return source_text

        namespace, transformed_fn = result

        # Re-apply cuTile wrappers in reverse order
        if wrappers:
            import cuda.tile as ct_mod
            for wrapper in reversed(wrappers):
                transformed_fn = ct_mod.kernel(transformed_fn)

        return transformed_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cutile_jit(fn=None, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in cuTile kernels.

    Usage:
        @lego.cutile_jit
        @ct.kernel
        def kernel(A, B, C, M, N, ...):
            ...
    """
    def decorator(fn):
        return rewrite(fn, CutileAdapter(), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator


def get_cutile_kernel_source(fn):
    """
    Utility function to retrieve the raw generated cuTile source
    code for a @lego.cutile_jit decorated function without compiling it.
    """
    return cutile_jit(fn, return_source=True)
