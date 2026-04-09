"""C++ source code generation adapter for the LEGO rewriter."""

from lego.cxx_printer import LEGOCXXCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
CXXAdapter = lambda: SourceGenAdapter(LEGOCXXCodePrinter())


def generate(fn, **kwargs):
    """Generate C++ source code from a function with LEGO layout expressions.

    Usage::

        cxx_code = lego.cxx_gen.generate(index)
    """
    return source_generate(fn, LEGOCXXCodePrinter(), **kwargs)
