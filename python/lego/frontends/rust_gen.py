"""Rust source code generation adapter for the LEGO rewriter."""

from lego.rust_printer import LEGORustCodePrinter
from lego.frontends._adapter import SourceGenAdapter, source_generate

# Kept for backwards compatibility (direct adapter use).
RustAdapter = lambda: SourceGenAdapter(LEGORustCodePrinter())


def generate(fn, **kwargs):
    """Generate Rust source code from a function with LEGO layout expressions.

    Usage::

        rust_code = lego.rust_gen.generate(index)
    """
    return source_generate(fn, LEGORustCodePrinter(), **kwargs)
