"""C++ code printer for LEGO layout expressions."""

from sympy.printing.cxx import CXX17CodePrinter
from .core import *


class LEGOCXXCodePrinter(CXX17CodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        return f"(({self._print(num)}) / ({self._print(den)}))"

    def _print_Pow(self, expr):
        from sympy.printing.precedence import precedence
        from sympy.core.numbers import equal_valued

        PREC = precedence(expr)

        if equal_valued(expr.exp, 2):
            base = self.parenthesize(expr.base, PREC)
            return f"{base} * {base}"
        elif equal_valued(expr.exp, -1):
            return f"1.0 / {self.parenthesize(expr.base, PREC)}"
        elif equal_valued(expr.exp, 0.5):
            return f"std::sqrt(static_cast<double>({self._print(expr.base)}))"
        elif equal_valued(expr.exp, 1.0 / 3):
            return f"std::cbrt(static_cast<double>({self._print(expr.base)}))"
        else:
            return f"std::pow(static_cast<double>({self._print(expr.base)}), {self._print(expr.exp)})"

    def _print_Mod(self, expr):
        return f"(({self._print(expr.args[0])}) % ({self._print(expr.args[1])}))"

    def _print_BroadcastRange(self, expr):
        return f"({self._print(expr.args[0])})"

    def _print_lego_arange(self, expr):
        # C++ has no built-in range literal; emit std::views::iota (C++20)
        return f"std::views::iota({self._print(expr.args[0])}, {self._print(expr.args[1])})"
