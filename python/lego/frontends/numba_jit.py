"""Numba CUDA adapter for the LEGO rewriter."""

import ast
import os

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# Numba CUDA code printer
# ---------------------------------------------------------------------------

class NumbaCUDACodePrinter(LEGOPythonCodePrinter):
    """Renders ``lego_arange`` as a plain ``range`` (Numba has no broadcast arange).

    Numba CUDA kernels use explicit thread indexing, so BroadcastRange is
    rendered without broadcast slicing — just the index expression.
    """

    def _print_lego_arange(self, expr):
        return f"range({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_BroadcastRange(self, expr):
        # Numba CUDA uses scalar thread indices — no broadcasting needed
        return f"({expr.args[0]})"


# ---------------------------------------------------------------------------
# Numba CUDA adapter
# ---------------------------------------------------------------------------

class NumbaCUDAAdapter(DSLAdapter):

    def unwrap(self, fn):
        original_fn = fn
        wrappers = []

        # Numba's @cuda.jit returns a Dispatcher; unwrap via .py_func
        if hasattr(original_fn, 'py_func'):
            wrappers.append(original_fn)
            original_fn = original_fn.py_func

        source_fn = original_fn
        return source_fn, original_fn, wrappers

    def find_runtime_vars(self, func_def):
        # Numba CUDA uses direct array indexing (x[i]) — arrays are passed
        # as arguments and indexed directly.  No tl.load/store equivalent.
        # Return empty set; array parameters are already kernel args.
        return set()

    def get_code_printer(self):
        return NumbaCUDACodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        if return_source:
            return new_source

        code_obj = compile(tree, filename="<lego-numba>", mode='exec')
        namespace = original_fn.__globals__.copy()
        exec(code_obj, namespace)

        transformed_fn = namespace[original_fn.__name__]

        # Re-apply @cuda.jit
        if wrappers:
            from numba import cuda
            transformed_fn = cuda.jit(transformed_fn)

        return transformed_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def jit(fn=None, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in Numba CUDA kernels.

    Usage:
        @lego_jit
        @cuda.jit
        def kernel(x, y, z, N):
            ...
    """
    def decorator(fn):
        return rewrite(fn, NumbaCUDAAdapter(), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator
