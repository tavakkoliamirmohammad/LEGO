"""Shared mixin for static-language LEGO code printers.

Provides common implementations of _print_BroadcastRange, _print_floor,
and _print_Mod that most static languages share. Language-specific
printers override only the parts that differ.
"""


class LEGOStaticLangMixin:
    """Mixin providing common print methods for static-language printers.

    Mix this in BEFORE the SymPy printer base class so these methods take
    precedence via MRO. Each method can be overridden in the concrete printer.
    """

    def _print_BroadcastRange(self, expr):
        return f"({self._print(expr.args[0])})"

    def _print_floor(self, expr):
        arg = expr.args[0]
        num, den = arg.as_numer_denom()
        return f"(({self._print(num)}) / ({self._print(den)}))"

    def _print_Mod(self, expr):
        return f"(({self._print(expr.args[0])}) % ({self._print(expr.args[1])}))"
