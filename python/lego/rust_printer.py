"""Rust code printer for LEGO layout expressions."""

from sympy.printing.rust import RustCodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGORustCodePrinter(LEGOStaticLangMixin, RustCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Pow(self, expr):
        from sympy.core.numbers import equal_valued

        if equal_valued(expr.exp, 2):
            base = self._print(expr.base)
            return f"({base} * {base})"
        elif equal_valued(expr.exp, -1):
            return f"(1.0 / {self._print(expr.base)})"
        elif equal_valued(expr.exp, 0.5):
            return f"({self._print(expr.base)} as f64).sqrt()"
        else:
            return f"({self._print(expr.base)} as f64).powi({self._print(expr.exp)})"

    def _print_lego_arange(self, expr):
        return f"({self._print(expr.args[0])}..{self._print(expr.args[1])})"
