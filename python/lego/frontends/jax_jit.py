"""JAX adapter for the LEGO rewriter."""

import ast
import os

from lego.python_printer import LEGOPythonCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# JAX code printer
# ---------------------------------------------------------------------------

class JAXCodePrinter(LEGOPythonCodePrinter):
    """Renders ``lego_arange`` as ``jnp.arange`` and ``BroadcastRange`` with
    NumPy-style broadcasting (``jnp.arange(N)[None, :, None]``)."""

    def _print_lego_arange(self, expr):
        return f"jnp.arange({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_BroadcastRange(self, expr):
        base_expr_str = self._print(expr.args[0])
        try:
            dim_val = int(expr.args[1])
            total_val = int(expr.args[2])
        except (TypeError, ValueError):
            dim_val = expr.args[1]
            total_val = expr.args[2]
        if total_val == 1 or not self.do_broadcast:
            return f"({base_expr_str})"
        slices = []
        if isinstance(total_val, int):
            for i in range(total_val):
                slices.append(":" if i == dim_val else "None")
        else:
            slices.append(f"None at dim {dim_val} and ':' otherwise")
        index_str = ", ".join(slices)
        return f"(({base_expr_str})[{index_str}])"


# ---------------------------------------------------------------------------
# JAX adapter
# ---------------------------------------------------------------------------

class JAXAdapter(DSLAdapter):

    def unwrap(self, fn):
        original_fn = fn
        wrappers = []

        # JAX's @jax.jit wraps via Traced/Staged objects.  For plain
        # functions (the common case) no unwrapping is needed.
        # If already jit-wrapped, pull out the underlying function.
        if hasattr(original_fn, '__wrapped__'):
            wrappers.append(original_fn)
            original_fn = original_fn.__wrapped__

        source_fn = original_fn
        return source_fn, original_fn, wrappers

    def find_runtime_vars(self, func_def):
        # JAX is purely functional — no pointer variables.
        # Array arguments are traced, not eagerly evaluated.
        return set()

    def get_code_printer(self):
        return JAXCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        if return_source:
            return new_source

        code_obj = compile(tree, filename="<lego-jax>", mode='exec')
        namespace = original_fn.__globals__.copy()
        exec(code_obj, namespace)

        transformed_fn = namespace[original_fn.__name__]

        # Re-apply @jax.jit, preserving static_argnums etc.
        if wrappers:
            import jax
            wrapper = wrappers[0]
            jit_kwargs = {}
            if hasattr(wrapper, '_jit_info'):
                info = wrapper._jit_info
                if info.static_argnums:
                    jit_kwargs['static_argnums'] = info.static_argnums
                if info.static_argnames:
                    jit_kwargs['static_argnames'] = info.static_argnames
                if info.donate_argnums:
                    jit_kwargs['donate_argnums'] = info.donate_argnums
            transformed_fn = jax.jit(transformed_fn, **jit_kwargs)

        return transformed_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def jit(fn=None, **kwargs):
    """
    Decorator that transforms LEGO layout expressions in JAX functions.

    Usage:
        @lego_jit
        @jax.jit
        def f(x, y, N):
            ...
    """
    def decorator(fn):
        return rewrite(fn, JAXAdapter(), **kwargs)

    if fn is not None:
        return decorator(fn)
    return decorator
