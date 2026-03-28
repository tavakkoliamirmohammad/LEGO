"""CUDA C code printer for LEGO layout expressions.

Extends the existing LEGOCCodePrinter with CUDA-specific intrinsics
for thread indexing and range generation.
"""

from .c_printer import LEGOCCodePrinter
from .core import *


class LEGOCUDACCodePrinter(LEGOCCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_BroadcastRange(self, expr):
        # CUDA uses scalar thread indexing — no broadcast
        return f"({self._print(expr.args[0])})"

    def _print_lego_arange(self, expr):
        # In CUDA C kernels, ranges map to thread-local loops
        return f"/* arange({self._print(expr.args[0])}, {self._print(expr.args[1])}) */"

    def _print_Mod(self, expr):
        return f"(({self._print(expr.args[0])}) % ({self._print(expr.args[1])}))"
