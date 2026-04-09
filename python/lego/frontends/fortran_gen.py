"""Fortran source code generation adapter for the LEGO rewriter."""

from lego.fortran_printer import LEGOFortranCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
FortranAdapter = lambda: SourceGenAdapter(LEGOFortranCodePrinter())


def generate(fn, **kwargs):
    """Generate Fortran source code from a function with LEGO layout expressions.

    Usage::

        fortran_code = lego.fortran_gen.generate(index)
    """
    return source_generate(fn, LEGOFortranCodePrinter(), **kwargs)
