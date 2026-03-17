"""
LEGO: Layout Expression Language for Code Generation

  Backend:   lego.backend (compiler, dialects, mlir_roundtrip)
  Frontends: lego.frontends.symbolic, .triton_jit, .python_mlir
"""
from .core import *
from .frontends.triton_jit import jit, get_kernel_source
from .frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, Tiled, Custom,
    row, col, reg_p, order_by, tile_by, group_by, gen_p,
)
