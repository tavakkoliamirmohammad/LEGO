"""GLSL code printer for LEGO layout expressions."""

from sympy.printing.glsl import GLSLPrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOGLSLCodePrinter(LEGOStaticLangMixin, GLSLPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_lego_arange(self, expr):
        # GLSL has no range type; emit a comment with the range bounds
        return f"/* arange({self._print(expr.args[0])}, {self._print(expr.args[1])}) */"
