from lego import *
from sympy.printing.pycode import PythonCodePrinter


class LEGOPythonCodePrinter(PythonCodePrinter):
    def __init__(self, *args, **kwargs):
        self.do_broadcast = kwargs.pop('do_broadcast', True)
        super().__init__(*args, kwargs)

    def _print_floor(self, expr):
        """
        Override printing for floor() so that floor(x/y) becomes (x//y)
        when possible.
        """
        arg = expr.args[0]
        # Try to extract numerator and denominator from the argument
        num, den = arg.as_numer_denom()
        return f"(({self._print(num)})//({self._print(den)}))"

    def _print_TritonRange(self, expr):
        base_expr_str = expr.args[0]
        try:
            dim_val = int(expr.args[1])
            total_val = int(expr.args[2])
        except (TypeError, ValueError):
            dim_val = expr.args[1]
            total_val = expr.args[2]
        if total_val == 1 or not self.do_broadcast:
            return f"({base_expr_str})"
        slices = []
        if isinstance(total_val, int):
            for i in range(total_val):
                slices.append(":" if i == dim_val else "None")
        else:
            slices.append(f"None at dim {dim_val} and ':' otherwise")
        index_str = ", ".join(slices)
        return f"(({base_expr_str})[{index_str}])"
