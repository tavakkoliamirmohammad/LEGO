"""JavaScript code printer for LEGO layout expressions."""

from sympy.printing.jscode import JavascriptCodePrinter
from ._printer_base import LEGOStaticLangMixin


class LEGOJSCodePrinter(LEGOStaticLangMixin, JavascriptCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        return f"Math.floor(({self._print(num)}) / ({self._print(den)}))"

    def _print_lego_arange(self, expr):
        # Generate Array.from({length: stop - start}, (_, i) => i + start)
        start = self._print(expr.args[0])
        stop = self._print(expr.args[1])
        return f"Array.from({{length: {stop} - {start}}}, (_, i) => i + {start})"
