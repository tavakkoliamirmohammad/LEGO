"""
LEGO: Layout Expression Language for Code Generation

  Backend:   lego.backend (symbolic, codegen, compiler, dialects)
  Frontends: lego.frontends.triton_jit, .python_mlir
"""
from .core import *
from .frontends.triton_jit import jit, get_kernel_source
from .frontends.python_mlir import (
    LegoLayout, RowMajor, ColMajor, Tiled, TiledPermute, TiledView, Custom,
    Transposed, ZCurve, Swizzle, BlockCyclic,
    Batched, BatchedLayout, LegoArray,
    row, col, reg_p, order_by, tile_by, group_by, gen_p,
)
from .backend.torch_tensor import LegoTensor, as_lego_tensor
from .autotune import autotune

# Register torch.compile "lego" backend (if torch available)
try:
    from .backend import fx_backend as _fx_backend  # noqa: F401
except ImportError:
    pass
