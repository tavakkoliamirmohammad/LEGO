"""JavaScript source code generation adapter for the LEGO rewriter."""

from lego.js_printer import LEGOJSCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
JSAdapter = lambda: SourceGenAdapter(LEGOJSCodePrinter())


def generate(fn, **kwargs):
    """Generate JavaScript source code from a function with LEGO layout expressions.

    Usage::

        js_code = lego.js_gen.generate(index_kernel)
    """
    return source_generate(fn, LEGOJSCodePrinter(), **kwargs)
