"""Fortran source code generation adapter for the LEGO rewriter."""

import ast

from lego.fortran_printer import LEGOFortranCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


# ---------------------------------------------------------------------------
# Fortran adapter
# ---------------------------------------------------------------------------

class FortranAdapter(DSLAdapter):
    """Adapter that emits Fortran source code from LEGO layout expressions."""

    def unwrap(self, fn):
        return fn, fn, []

    def find_runtime_vars(self, func_def):
        return set()

    def get_code_printer(self):
        return LEGOFortranCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        return new_source


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(fn, **kwargs):
    """Generate Fortran source code from a function with LEGO layout expressions.

    Usage::

        def index(M, N, K):
            L = OrderBy(Row(M, N)).TileBy((M // BM, N // BN), (BM, BN))
            offset = L[pid_m, pid_n, :, :]
            return offset

        fortran_code = lego.fortran_gen.generate(index)
    """
    return rewrite(fn, FortranAdapter(), return_source=True, **kwargs)
