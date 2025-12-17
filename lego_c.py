from sympy.printing.c import C99CodePrinter
from lego import *


class LEGOCCodePrinter(C99CodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, kwargs)

    def _print_floor(self, expr):
        arg = expr.args[0]
        # Try to extract numerator and denominator from the argument
        num, den = arg.as_numer_denom()
        return f"(({self._print(num)})/({self._print(den)}))"

    def _print_Pow(self, expr):
        from sympy.printing.precedence import precedence
        from sympy.core.numbers import equal_valued
        from sympy.codegen.ast import real

        if "Pow" in self.known_functions:
            return self._print_Function(expr)

        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)

        # Check if expression is pow(x, 2) and convert to x*x
        if equal_valued(expr.exp, 2):
            return '%s*%s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, -1):
            literal_suffix = self._get_literal_suffix(sp.real)
            return '1.0%s/%s' % (literal_suffix, self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                        self._print(expr.exp))
