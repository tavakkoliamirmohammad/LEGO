"""GLSL code printer for LEGO layout expressions."""

from sympy.printing.glsl import GLSLPrinter
from .core import *


class LEGOGLSLCodePrinter(GLSLPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        # GLSL integer division truncates
        return f"(({self._print(num)}) / ({self._print(den)}))"

    def _print_Mod(self, expr):
        return f"(({self._print(expr.args[0])}) % ({self._print(expr.args[1])}))"

    def _print_BroadcastRange(self, expr):
        return f"({self._print(expr.args[0])})"

    def _print_lego_arange(self, expr):
        # GLSL has no range type; emit a comment with the range bounds
        return f"/* arange({self._print(expr.args[0])}, {self._print(expr.args[1])}) */"
