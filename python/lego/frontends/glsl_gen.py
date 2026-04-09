"""GLSL source code generation adapter for the LEGO rewriter."""

from lego.glsl_printer import LEGOGLSLCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
GLSLAdapter = lambda: SourceGenAdapter(LEGOGLSLCodePrinter())


def generate(fn, **kwargs):
    """Generate GLSL source code from a function with LEGO layout expressions.

    Usage::

        glsl_code = lego.glsl_gen.generate(index_kernel)
    """
    return source_generate(fn, LEGOGLSLCodePrinter(), **kwargs)
