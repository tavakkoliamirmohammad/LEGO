"""Verify that all LEGO Python imports work correctly."""
print("[check-lego-imports] Testing core imports...")
from lego import jit
from lego.core import OrderBy, Row, Col, GroupBy
print("[check-lego-imports] Testing backend imports...")
from lego.backend.compiler import LayoutCompiler
from lego.backend.dialects.lego_dialect import register
print("[check-lego-imports] Testing frontend imports...")
from lego.frontends.python_mlir import Tiled
from lego.core import le_constraint, divisibility_constraint
print("[check-lego-imports] All imports OK")
