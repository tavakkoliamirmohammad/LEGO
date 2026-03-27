"""Rust source code generation adapter for the LEGO rewriter."""

import ast

from lego.rust_printer import LEGORustCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# Rust adapter
# ---------------------------------------------------------------------------

class RustAdapter(DSLAdapter):
    """Adapter that emits Rust source code from LEGO layout expressions."""

    def unwrap(self, fn):
        return fn, fn, []

    def find_runtime_vars(self, func_def):
        return set()

    def get_code_printer(self):
        return LEGORustCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        # Always return source — Rust is an ahead-of-time compiled language
        return new_source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(fn, **kwargs):
    """Generate Rust source code from a function with LEGO layout expressions.

    Usage::

        def index(M, N, K):
            L = OrderBy(Row(M, N)).TileBy((M // BM, N // BN), (BM, BN))
            offset = L[pid_m, pid_n, :, :]
            return offset

        rust_code = lego.rust_gen.generate(index)
    """
    return rewrite(fn, RustAdapter(), return_source=True, **kwargs)
