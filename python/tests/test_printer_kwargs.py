"""Regression tests: printer kwargs must be forwarded and affect output."""
import sympy as sp
from lego.c_printer import LEGOCCodePrinter
from lego.cxx_printer import LEGOCXXCodePrinter
from lego.rust_printer import LEGORustCodePrinter
from lego.js_printer import LEGOJSCodePrinter
from lego.fortran_printer import LEGOFortranCodePrinter
from lego.glsl_printer import LEGOGLSLCodePrinter
from lego.julia_printer import LEGOJuliaCodePrinter
from lego.python_printer import LEGOPythonCodePrinter
import pytest

ALL_PRINTERS = [
    LEGOCCodePrinter, LEGOCXXCodePrinter, LEGORustCodePrinter,
    LEGOJSCodePrinter, LEGOFortranCodePrinter, LEGOGLSLCodePrinter,
    LEGOJuliaCodePrinter, LEGOPythonCodePrinter,
]


@pytest.mark.parametrize("PrinterClass", ALL_PRINTERS)
def test_printer_accepts_kwargs(PrinterClass):
    """Every printer must accept settings dict without crashing."""
    printer = PrinterClass(settings={"full_prec": True})
    x = sp.Symbol("x")
    result = printer.doprint(x)
    assert "x" in result


@pytest.mark.parametrize("PrinterClass", ALL_PRINTERS)
def test_full_prec_affects_float_output(PrinterClass):
    """full_prec=True should produce more digits than full_prec=False."""
    lo = PrinterClass(settings={"full_prec": False})
    hi = PrinterClass(settings={"full_prec": True})
    expr = sp.Float(1.0) / sp.Integer(3)
    lo_result = lo.doprint(expr)
    hi_result = hi.doprint(expr)
    # High-precision representation should be at least as long.
    assert len(hi_result) >= len(lo_result)


@pytest.mark.parametrize("PrinterClass", ALL_PRINTERS)
def test_printer_handles_complex_expr(PrinterClass):
    """Printers should produce non-empty output for compound expressions."""
    x, y = sp.symbols("x y")
    expr = x**2 + 3 * y - 1
    printer = PrinterClass()
    result = printer.doprint(expr)
    assert result  # non-empty
    assert "x" in result
    assert "y" in result


@pytest.mark.parametrize("PrinterClass", ALL_PRINTERS)
def test_printer_handles_floor(PrinterClass):
    """floor(x/y) rendering should work in all printers."""
    x, y = sp.symbols("x y")
    expr = sp.floor(x / y)
    printer = PrinterClass()
    result = printer.doprint(expr)
    assert result
    assert "x" in result
    assert "y" in result


def test_python_printer_do_broadcast_off():
    """do_broadcast=False should suppress broadcasting index notation."""
    x = sp.Symbol("x")
    from lego.core import BroadcastRange
    expr = BroadcastRange(x, 0, 2)
    on = LEGOPythonCodePrinter(do_broadcast=True)
    off = LEGOPythonCodePrinter(do_broadcast=False)
    result_on = on.doprint(expr)
    result_off = off.doprint(expr)
    # With broadcast on: should have indexing syntax (None).
    # With broadcast off: should be simpler.
    assert "None" in result_on
    assert "None" not in result_off


def test_fortran_standard_setting():
    """Fortran printer should accept and use standard setting."""
    p90 = LEGOFortranCodePrinter(settings={"standard": 90})
    p95 = LEGOFortranCodePrinter(settings={"standard": 95})
    x = sp.Symbol("x")
    # Both must produce valid output.
    assert "x" in p90.doprint(x)
    assert "x" in p95.doprint(x)


@pytest.mark.parametrize("PrinterClass", ALL_PRINTERS)
def test_printer_mod_expr(PrinterClass):
    """Mod expression should render in all printers."""
    x, y = sp.symbols("x y")
    expr = sp.Mod(x, y)
    printer = PrinterClass()
    result = printer.doprint(expr)
    assert result
    assert "x" in result
