"""Bitwise primitives in the algebra: BitXor / BitAnd fold on constants, print in every language,
lower to arith.xori / arith.andi, and survive the MLIR round trip."""
import pytest
import sympy as sp

from lego.core import BitXor, BitAnd


def test_bitwise_functions_fold_on_integer_arguments_and_stay_symbolic_otherwise():
    a = sp.Symbol("a", integer=True, nonnegative=True)
    assert BitXor(sp.Integer(5), sp.Integer(3)) == 6
    assert BitAnd(sp.Integer(13), sp.Integer(7)) == 5
    assert isinstance(BitXor(a, sp.Integer(3)), BitXor)
    assert BitXor(a, 3).subs(a, 6) == 5


def test_printers_render_bitwise_ops_in_each_language():
    from lego.c_printer import LEGOCCodePrinter
    from lego.python_printer import LEGOPythonCodePrinter
    from lego.fortran_printer import LEGOFortranCodePrinter
    from lego.julia_printer import LEGOJuliaCodePrinter
    a, b = sp.symbols("a b", integer=True, nonnegative=True)
    assert LEGOCCodePrinter().doprint(BitXor(a, b)) == "((a) ^ (b))"
    assert LEGOCCodePrinter().doprint(BitAnd(a, b)) == "((a) & (b))"
    assert LEGOPythonCodePrinter().doprint(BitXor(a, b)) == "((a) ^ (b))"
    assert LEGOFortranCodePrinter().doprint(BitXor(a, b)).strip() == "ieor(a, b)"
    assert LEGOJuliaCodePrinter().doprint(BitXor(a, b)) == "xor(a, b)"


def test_bitxor_round_trips_through_the_mlir_lowering():
    pytest.importorskip("lego.mlir.ir")
    from lego.core import GenP, OrderBy, GroupBy
    rows, cols = 16, 16

    def f_apply(args):
        i, j = args
        return i * cols + BitXor(j, BitAnd(i, 7))

    def f_inv(f):
        i = sp.floor(f / cols); jp = sp.Mod(f, cols)
        return (i, BitXor(jp, BitAnd(i, 7)))

    L = GroupBy([(rows, cols)], [OrderBy(GenP((rows, cols), f_apply, f_inv))])
    i, j = sp.symbols("i j", integer=True, nonnegative=True)
    expr = L[i, j]
    assert expr.has(BitXor), expr
    assert int(expr.subs({i: 5, j: 3})) == 5 * 16 + (3 ^ 5)
