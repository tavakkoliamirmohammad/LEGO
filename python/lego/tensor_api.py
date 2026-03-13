"""
LEGO Tensor API

User-facing API for applying LEGO layout transformations to NumPy arrays
and PyTorch tensors.

Two usage patterns:
  1. Power users: use the composable descriptor API (row, col, order_by, tile_by, ...)
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
from .compiler import (
    RegPDesc, RowDesc, ColDesc, OrderByDesc, GroupByDesc, TileByDesc, GenPDesc,
    _LAYOUT_DESC_TYPES, _layout_from_sympy,
    LayoutCompiler, _dtype_to_mlir,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ============================================================================
# Composable functional constructors (mirror MLIR ops)
# ============================================================================

def row(*dims):
    """Create a RowDesc — mirrors lego.row."""
    return RowDesc(dims)


def col(*dims):
    """Create a ColDesc — mirrors lego.col."""
    return ColDesc(dims)


def reg_p(dims, perm):
    """Create a RegPDesc — mirrors lego.reg_p."""
    return RegPDesc(dims, perm)


def order_by(*ps):
    """Create an OrderByDesc — mirrors lego.order_by."""
    return OrderByDesc(list(ps))


def tile_by(input_layout, *tile_groups):
    """Create a TileByDesc — mirrors lego.tile_by."""
    return TileByDesc(input_layout, list(tile_groups))


def group_by(dims, *objects):
    """Create a GroupByDesc — mirrors lego.group_by."""
    return GroupByDesc(dims, list(objects))


def gen_p(dims, apply_builder, inv_builder):
    """Create a GenPDesc — mirrors lego.gen_p."""
    return GenPDesc(dims, apply_builder, inv_builder)


# ============================================================================
# LegoLayout wrapper
# ============================================================================

class LegoLayout:
    """Wrapper around a layout descriptor with convenience methods and caching.

    Provides transform/inverse_transform for both NumPy arrays and PyTorch
    tensors, with automatic JIT compilation and caching.
    """

    def __init__(self, layout, shape=None):
        """
        Args:
            layout: Any layout descriptor (GroupByDesc, TileByDesc, etc.) or SymPy layout
            shape: Tuple of concrete dimension sizes (optional, inferred from layout.dims)
        """
        if not isinstance(layout, _LAYOUT_DESC_TYPES):
            # Convert SymPy-based layout to descriptor
            layout = _layout_from_sympy(layout)
        self._layout = layout
        if shape is None:
            shape = layout.dims
        self._shape = tuple(int(s) for s in shape)
        self._numel = 1
        for s in self._shape:
            self._numel *= s
        self._cache = {}  # (shape, dtype_str) -> compiler

    @property
    def shape(self):
        """The logical shape of the layout."""
        return self._shape

    @property
    def numel(self):
        """Total number of elements."""
        return self._numel

    def create_tensor(self, dtype):
        """Create a tensor viewed through this layout.

        Generates a flat identity range [0, 1, ..., numel-1], applies the
        layout transform, and returns the result reshaped to the layout's
        logical shape.

        Args:
            dtype: NumPy dtype (e.g. np.float32, "float32")
        """
        arr = np.arange(self._numel, dtype=dtype)
        return self.transform(arr)

    def _get_compiler(self, tensor):
        """Get or create a compiler for the given tensor's dtype."""
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            dtype_str = _dtype_to_mlir(tensor.dtype)
        elif isinstance(tensor, np.ndarray):
            dtype_str = _dtype_to_mlir(tensor.dtype)
        else:
            dtype_str = "f32"

        key = (tensor.shape, dtype_str)
        if key not in self._cache:
            self._cache[key] = LayoutCompiler(
                self._layout, self._shape, dtype_str
            )
        return self._cache[key]

    def transform(self, tensor):
        """Apply the layout transformation to a tensor.

        Args:
            tensor: NumPy array or PyTorch tensor (any shape with matching element count)

        Returns:
            Transformed tensor reshaped to the layout's logical shape
        """
        compiler = self._get_compiler(tensor)

        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from .torch_ops import LegoTransformFunction
            if LegoTransformFunction is not None:
                return LegoTransformFunction.apply(tensor, compiler).reshape(self._shape)

        if isinstance(tensor, np.ndarray):
            return compiler.transform_numpy(tensor).reshape(self._shape)

        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def inverse_transform(self, tensor):
        """Apply the inverse layout transformation.

        Args:
            tensor: Transformed NumPy array or PyTorch tensor

        Returns:
            Original-layout tensor reshaped to the layout's logical shape
        """
        compiler = self._get_compiler(tensor)

        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from .torch_ops import LegoInverseTransformFunction
            if LegoInverseTransformFunction is not None:
                return LegoInverseTransformFunction.apply(tensor, compiler).reshape(self._shape)

        if isinstance(tensor, np.ndarray):
            return compiler.inverse_transform_numpy(tensor).reshape(self._shape)

        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def get_mlir(self, dtype="f32"):
        """Return the generated MLIR module text for inspection."""
        compiler = LayoutCompiler(self._layout, self._shape, dtype)
        return compiler.mlir_text

    def benchmark(self, tensor, n_iters=100):
        """Benchmark the transform on a tensor.

        Args:
            tensor: Input tensor (NumPy or PyTorch)
            n_iters: Number of iterations

        Returns:
            Dict with 'mean_ms', 'min_ms', 'max_ms', 'n_iters'
        """
        # Warm up
        self.transform(tensor)

        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            self.transform(tensor)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return {
            "mean_ms": np.mean(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "std_ms": np.std(times),
            "n_iters": n_iters,
        }


# ============================================================================
# Convenience Constructors
# ============================================================================


def RowMajor(shape):
    """Create a row-major layout for the given shape.

    This is the standard C/NumPy memory layout.

    Args:
        shape: Tuple of dimension sizes, e.g. (512, 512)
    """
    group = GroupByDesc(shape, [OrderByDesc([RowDesc(shape)])])
    return LegoLayout(group, shape)


def ColMajor(shape):
    """Create a column-major (Fortran-order) layout.

    Args:
        shape: Tuple of dimension sizes
    """
    group = GroupByDesc(shape, [OrderByDesc([ColDesc(shape)])])
    return LegoLayout(group, shape)


def Tiled(shape, tile_shape):
    """Create a tiled layout.

    Elements within each tile are stored contiguously, then tiles are
    arranged in row-major order.

    Args:
        shape: Global shape, e.g. (512, 512)
        tile_shape: Tile shape, e.g. (64, 64)
    """
    if len(shape) != len(tile_shape):
        raise ValueError("shape and tile_shape must have the same rank")

    tile_grid = tuple(s // t for s, t in zip(shape, tile_shape))
    ob = OrderByDesc([RowDesc(shape)])
    tb = TileByDesc(ob, [tile_grid, tile_shape])
    return LegoLayout(tb, shape)


def Custom(layout_obj, shape):
    """Wrap an existing layout object (any descriptor type or SymPy layout).

    Args:
        layout_obj: A layout descriptor or SymPy layout object
        shape: Tuple of dimension sizes
    """
    return LegoLayout(layout_obj, shape)
