"""JavaScript source code generation adapter for the LEGO rewriter."""

import ast

from lego.js_printer import LEGOJSCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


class JSAdapter(DSLAdapter):
    """Adapter that emits JavaScript source code from LEGO layout expressions."""

    def unwrap(self, fn):
        return fn, fn, []

    def find_runtime_vars(self, func_def):
        return set()

    def get_code_printer(self):
        return LEGOJSCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        return new_source


def generate(fn, **kwargs):
    """Generate JavaScript source code from a function with LEGO layout expressions.

    Useful for WebGPU / WebAssembly targets.

    Usage::

        js_code = lego.js_gen.generate(index_kernel)
    """
    return rewrite(fn, JSAdapter(), return_source=True, **kwargs)
