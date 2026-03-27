"""
LEGO: Layout Expression Language for Code Generation

  Backend:   lego.backend (symbolic, codegen, compiler, dialects)
  Frontends: lego.frontends.triton_jit, .cutile_jit, .python_mlir
"""
from .core import *
from .frontends.triton_jit import jit, get_kernel_source
from .frontends.cutile_jit import cutile_jit, get_cutile_kernel_source
from .frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, Tiled, TiledView, Custom,
    Transposed, ZCurve, Swizzle, BlockCyclic,
    Batched, BatchedLayout, LegoArray,
    row, col, reg_p, order_by, tile_by, group_by, gen_p,
)
from .backend.torch_tensor import LegoTensor, as_lego_tensor
from .frontends import rust_gen, fortran_gen, cxx_gen
from .frontends import julia_gen, cuda_c_gen, js_gen, glsl_gen
