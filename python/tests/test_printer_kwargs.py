"""Regression test: printer kwargs must be forwarded correctly."""
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

@pytest.mark.parametrize("PrinterClass", [
    LEGOCCodePrinter, LEGOCXXCodePrinter, LEGORustCodePrinter,
    LEGOJSCodePrinter, LEGOFortranCodePrinter, LEGOGLSLCodePrinter,
    LEGOJuliaCodePrinter, LEGOPythonCodePrinter,
])
def test_printer_accepts_kwargs(PrinterClass):
    printer = PrinterClass(settings={"full_prec": True})
    x = sp.Symbol("x")
    result = printer.doprint(x)
    assert "x" in result
