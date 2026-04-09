"""Julia source code generation adapter for the LEGO rewriter."""

from lego.julia_printer import LEGOJuliaCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
JuliaAdapter = lambda: SourceGenAdapter(LEGOJuliaCodePrinter())


def generate(fn, **kwargs):
    """Generate Julia source code from a function with LEGO layout expressions.

    Usage::

        julia_code = lego.julia_gen.generate(index_kernel)
    """
    return source_generate(fn, LEGOJuliaCodePrinter(), **kwargs)
