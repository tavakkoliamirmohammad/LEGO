"""
LEGO Tensor API

User-facing API for applying LEGO layout transformations to NumPy arrays
and PyTorch tensors.

Two usage patterns:
  1. Power users: use the composable API (row, col, order_by, tile_by, ...)
  2. Casual users: use convenience constructors (RowMajor, ColMajor, Tiled)

Example:
    import lego
    import numpy as np

    arr = np.random.randn(512, 512).astype(np.float32)
    layout = lego.Tiled((512, 512), tile_shape=(64, 64))
    tiled = layout.transform(arr)
    back = layout.inverse_transform(tiled)
    assert np.allclose(arr, back)
"""

import time
import numpy as np
from lego.core import (
    LayoutBlock, Row, Col, RegP, OrderBy, GroupBy, TileByLayout, GenP,
)
from lego.backend.compiler import LayoutCompiler, _dtype_to_mlir

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ============================================================================
# Composable functional constructors (mirror MLIR ops)
# ============================================================================

def row(*dims):
    """Create a Row layout — mirrors lego.row."""
    return Row(*dims)


def col(*dims):
    """Create a Col layout — mirrors lego.col."""
    return Col(*dims)


def reg_p(dims, perm):
    """Create a RegP layout — mirrors lego.reg_p."""
    return RegP(dims, perm)


def order_by(*ps):
    """Create an OrderBy layout — mirrors lego.order_by."""
    return OrderBy(*ps)


def tile_by(input_layout, *tile_groups):
    """Create a TileByLayout — mirrors lego.tile_by."""
    return TileByLayout(
        input_chain=[input_layout] if isinstance(input_layout, OrderBy) else input_layout.chain,
        tile_groups=list(tile_groups),
        group_dims=[tuple(d for g in tile_groups for d in g)],
    )


def group_by(dims, *objects):
    """Create a GroupBy layout — mirrors lego.group_by."""
    return GroupBy([dims], list(objects))


def gen_p(dims, f_apply, f_inv):
    """Create a GenP layout — mirrors lego.gen_p."""
    return GenP(dims, f_apply, f_inv)


# ============================================================================
# LegoLayout wrapper
# ============================================================================

class LegoLayout:
    """Wrapper around a layout object with convenience methods and caching."""

    def __init__(self, layout, shape=None):
        self._layout = layout
        if shape is None:
            dims = layout._dims if hasattr(layout, '_dims') else layout.dims()
            shape = dims
        self._shape = tuple(int(s) for s in shape)
        self._numel = 1
        for s in self._shape:
            self._numel *= s
        self._cache = {}

    @property
    def shape(self):
        return self._shape

    @property
    def numel(self):
        return self._numel

    def create_tensor(self, dtype):
        arr = np.arange(self._numel, dtype=dtype)
        return self.transform(arr)

    def _get_compiler(self, tensor):
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            dtype_str = _dtype_to_mlir(tensor.dtype)
        elif isinstance(tensor, np.ndarray):
            dtype_str = _dtype_to_mlir(tensor.dtype)
        else:
            dtype_str = "f32"

        key = (tensor.shape, dtype_str)
        if key not in self._cache:
            self._cache[key] = LayoutCompiler(self._layout, self._shape, dtype_str)
        return self._cache[key]

    def transform(self, tensor):
        compiler = self._get_compiler(tensor)
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from lego.backend.torch_ops import LegoTransformFunction
            if LegoTransformFunction is not None:
                return LegoTransformFunction.apply(tensor, compiler).reshape(self._shape)
        if isinstance(tensor, np.ndarray):
            return compiler.transform_numpy(tensor).reshape(self._shape)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def inverse_transform(self, tensor):
        compiler = self._get_compiler(tensor)
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from lego.backend.torch_ops import LegoInverseTransformFunction
            if LegoInverseTransformFunction is not None:
                return LegoInverseTransformFunction.apply(tensor, compiler).reshape(self._shape)
        if isinstance(tensor, np.ndarray):
            return compiler.inverse_transform_numpy(tensor).reshape(self._shape)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def get_mlir(self, dtype="f32"):
        return LayoutCompiler(self._layout, self._shape, dtype).mlir_text

    def benchmark(self, tensor, n_iters=100):
        self.transform(tensor)
        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            self.transform(tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return {
            "mean_ms": np.mean(times), "min_ms": np.min(times),
            "max_ms": np.max(times), "std_ms": np.std(times), "n_iters": n_iters,
        }


# ============================================================================
# Convenience Constructors
# ============================================================================

def RowMajor(shape):
    """Row-major (C/NumPy) layout."""
    layout = GroupBy([shape], [OrderBy(Row(*shape))])
    return LegoLayout(layout, shape)


def ColMajor(shape):
    """Column-major (Fortran) layout."""
    layout = GroupBy([shape], [OrderBy(Col(*shape))])
    return LegoLayout(layout, shape)


def Tiled(shape, tile_shape):
    """Tiled layout — tiles arranged row-major, elements within tiles contiguous."""
    if len(shape) != len(tile_shape):
        raise ValueError("shape and tile_shape must have the same rank")
    tile_grid = tuple(s // t for s, t in zip(shape, tile_shape))
    layout = OrderBy(Row(*shape)).TileBy(tile_grid, tile_shape)
    return LegoLayout(layout, shape)


def Custom(layout_obj, shape):
    """Wrap an existing layout object."""
    return LegoLayout(layout_obj, shape)
