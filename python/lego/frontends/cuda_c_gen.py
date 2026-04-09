"""CUDA C source code generation adapter for the LEGO rewriter."""

from lego.cuda_c_printer import LEGOCUDACCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
CUDACAdapter = lambda: SourceGenAdapter(LEGOCUDACCodePrinter())


def generate(fn, **kwargs):
    """Generate CUDA C source code from a function with LEGO layout expressions.

    Usage::

        cuda_code = lego.cuda_c_gen.generate(index_kernel)
    """
    return source_generate(fn, LEGOCUDACCodePrinter(), **kwargs)
