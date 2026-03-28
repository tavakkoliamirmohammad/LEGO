"""Julia code printer for LEGO layout expressions."""

from sympy.printing.julia import JuliaCodePrinter
from .core import *


class LEGOJuliaCodePrinter(JuliaCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        # Julia: div(a, b) for integer floor division
        return f"div({self._print(num)}, {self._print(den)})"

    def _print_Mod(self, expr):
        return f"mod({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_BroadcastRange(self, expr):
        return f"({self._print(expr.args[0])})"

    def _print_lego_arange(self, expr):
        # Julia ranges are inclusive and 1-indexed by convention,
        # but for index computation we keep 0-based to match LEGO semantics
        start = self._print(expr.args[0])
        stop = expr.args[1]
        stop_str = self._print(stop - 1) if hasattr(stop, '__sub__') else f"({self._print(stop)} - 1)"
        return f"({start}:{stop_str})"
