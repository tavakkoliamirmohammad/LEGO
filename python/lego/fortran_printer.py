"""Fortran code printer for LEGO layout expressions."""

from sympy.printing.fortran import FCodePrinter
from .core import *


class LEGOFortranCodePrinter(FCodePrinter):
    def __init__(self, *args, **kwargs):
        # Default to Fortran 95 standard
        kwargs.setdefault('standard', 95)
        super().__init__(*args, kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        # Fortran integer division truncates toward zero; for positive values
        # this matches floor division.
        return f"(({self._print(num)}) / ({self._print(den)}))"

    def _print_Pow(self, expr):
        from sympy.core.numbers import equal_valued

        if equal_valued(expr.exp, 2):
            base = self._print(expr.base)
            return f"({base}**2)"
        elif equal_valued(expr.exp, 0.5):
            return f"sqrt(dble({self._print(expr.base)}))"
        else:
            return f"(({self._print(expr.base)})**({self._print(expr.exp)}))"

    def _print_Mod(self, expr):
        return f"mod({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_BroadcastRange(self, expr):
        # Fortran arrays are 1-indexed; emit the base expression
        return f"({self._print(expr.args[0])})"

    def _print_lego_arange(self, expr):
        # Fortran implicit DO / array constructor range
        start = self._print(expr.args[0])
        # Fortran ranges are inclusive on both ends, so stop - 1
        stop = expr.args[1]
        stop_str = self._print(stop - 1) if hasattr(stop, '__sub__') else f"({self._print(stop)} - 1)"
        return f"(/ (i, i = {start}, {stop_str}) /)"
