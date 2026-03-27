"""CUDA C source code generation adapter for the LEGO rewriter."""

import ast

from lego.cuda_c_printer import LEGOCUDACCodePrinter
from lego.frontends._adapter import DSLAdapter
from lego.rewriter import rewrite


class CUDACAdapter(DSLAdapter):
    """Adapter that emits CUDA C source code from LEGO layout expressions."""

    def unwrap(self, fn):
        return fn, fn, []

    def find_runtime_vars(self, func_def):
        return set()

    def get_code_printer(self):
        return LEGOCUDACCodePrinter()

    def compile_and_wrap(self, new_source, tree, original_fn, wrappers,
                         return_source=False):
        return new_source


def generate(fn, **kwargs):
    """Generate CUDA C source code from a function with LEGO layout expressions.

    Usage::

        cuda_code = lego.cuda_c_gen.generate(index_kernel)
    """
    return rewrite(fn, CUDACAdapter(), return_source=True, **kwargs)
