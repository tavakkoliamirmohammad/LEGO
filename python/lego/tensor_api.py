"""
LEGO Tensor API

User-facing API for applying LEGO layout transformations to NumPy arrays
and PyTorch tensors.

Two usage patterns:
  1. Power users: use the algebra directly (OrderBy, GroupBy) with .transform()
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
from .lego import Row, Col, OrderBy, RegP, GroupBy, get_sigma_perm
from .compiler import LayoutCompiler, _dtype_to_mlir

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class LegoLayout:
    """Wrapper around a GroupBy layout with convenience methods and caching.

    Provides transform/inverse_transform for both NumPy arrays and PyTorch
    tensors, with automatic JIT compilation and caching.
    """

    def __init__(self, layout, shape):
        """
        Args:
            layout: A GroupBy layout object
            shape: Tuple of concrete dimension sizes
        """
        if not isinstance(layout, GroupBy):
            raise TypeError(
                f"Expected GroupBy layout, got {type(layout).__name__}"
            )
        self._layout = layout
        self._shape = tuple(int(s) for s in shape)
        self._cache = {}  # (shape, dtype_str) -> compiler

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
            tensor: NumPy array or PyTorch tensor

        Returns:
            Transformed tensor of the same type, shape, and dtype
        """
        compiler = self._get_compiler(tensor)

        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from .torch_ops import LegoTransformFunction
            if LegoTransformFunction is not None:
                return LegoTransformFunction.apply(tensor, compiler)

        if isinstance(tensor, np.ndarray):
            return compiler.transform_numpy(tensor)

        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def inverse_transform(self, tensor):
        """Apply the inverse layout transformation.

        Args:
            tensor: Transformed NumPy array or PyTorch tensor

        Returns:
            Original-layout tensor
        """
        compiler = self._get_compiler(tensor)

        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from .torch_ops import LegoInverseTransformFunction
            if LegoInverseTransformFunction is not None:
                return LegoInverseTransformFunction.apply(tensor, compiler)

        if isinstance(tensor, np.ndarray):
            return compiler.inverse_transform_numpy(tensor)

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
    L = OrderBy(Row(*shape)).GroupBy([shape])
    return LegoLayout(L, shape)


def ColMajor(shape):
    """Create a column-major (Fortran-order) layout.

    Args:
        shape: Tuple of dimension sizes
    """
    L = OrderBy(Col(*shape)).GroupBy([shape])
    return LegoLayout(L, shape)


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

    # Compute tile grid dimensions
    # shape = tile_grid * tile_shape
    tile_grid = tuple(s // t for s, t in zip(shape, tile_shape))

    # Build: OrderBy(Row(*shape)).TileBy(tile_grid, tile_shape)
    L = OrderBy(Row(*shape)).TileBy(tile_grid, tile_shape)
    return LegoLayout(L, shape)


def Custom(layout_obj, shape):
    """Wrap an existing GroupBy layout object.

    Args:
        layout_obj: A GroupBy layout object
        shape: Tuple of dimension sizes
    """
    return LegoLayout(layout_obj, shape)
