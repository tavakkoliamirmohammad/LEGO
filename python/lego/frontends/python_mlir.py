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

import math
import time
import numpy as np
from lego.core import (
    LayoutBlock, Row, Col, RegP, OrderBy, GroupBy, TileByLayout, GenP,
)
from lego.backend.compiler import LayoutCompiler, _dtype_to_mlir


def _check_layout_invertible(layout):
    """Walk layout tree and raise if any GenP lacks an inverse function."""
    if isinstance(layout, GenP):
        if not layout.has_inverse:
            raise ValueError(
                "Cannot compute inverse: GenP has no inverse function. "
                "Provide f_inv explicitly."
            )
    elif isinstance(layout, OrderBy):
        for p in layout.perms:
            _check_layout_invertible(p)
    elif isinstance(layout, TileByLayout):
        for ob in layout._input_chain:
            _check_layout_invertible(ob)
    elif isinstance(layout, GroupBy):
        for obj in layout.objects:
            _check_layout_invertible(obj)

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
        for i, d in enumerate(self._shape):
            if d <= 0:
                raise ValueError(
                    f"LegoLayout: dimension {i} has non-positive size {d}. "
                    f"All dimensions must be > 0."
                )
        self._numel = 1
        for s in self._shape:
            self._numel *= s
        self._cache = {}
        self._perm_cache = {}
        self._composed_perm = None
        self._compiled_torch = None  # cached (fwd_fn, inv_fns, layout_dims)

    @property
    def shape(self):
        return self._shape

    @property
    def numel(self):
        return self._numel

    @property
    def rank(self):
        """Number of dimensions."""
        return len(self._shape)

    @property
    def is_identity(self):
        """True if the layout is row-major (no-op transform)."""
        compiler = LayoutCompiler(self._layout, self._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        return np.array_equal(fwd, np.arange(len(fwd)))

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

    def _get_perm_tensors(self, device):
        """Get cached (fwd, inv) permutation tensors for a torch device."""
        key = str(device)
        if key not in self._perm_cache:
            if self._composed_perm is not None:
                fwd, inv = self._composed_perm
            else:
                compiler = LayoutCompiler(self._layout, self._shape, "i64")
                fwd, inv = compiler.get_permutation_table()
            fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to(device)
            inv_t = torch.from_numpy(np.ascontiguousarray(inv)).to(device)
            self._perm_cache[key] = (fwd_t, inv_t)
        return self._perm_cache[key]

    def _validate_numel(self, tensor):
        """Validate that tensor numel matches layout numel."""
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            actual = tensor.numel()
        elif isinstance(tensor, np.ndarray):
            actual = tensor.size
        else:
            return
        if actual != self._numel:
            raise ValueError(
                f"Tensor numel ({actual}) != layout numel ({self._numel}). "
                f"Layout expects shape compatible with {self._shape}."
            )

    def _get_compiled_torch(self):
        """Get cached compiled layout transform functions for PyTorch."""
        if self._compiled_torch is None:
            from lego.backend.torch_layout import compile_layout_transform
            self._compiled_torch = compile_layout_transform(self._layout, self._shape)
        return self._compiled_torch

    def transform(self, tensor):
        self._validate_numel(tensor)
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from lego.backend.torch_layout import apply_layout_torch
            return apply_layout_torch(tensor, self._layout, self._shape,
                                      compiled=self._get_compiled_torch())
        if isinstance(tensor, np.ndarray):
            if self._composed_perm is not None:
                fwd, _ = self._composed_perm
                return tensor.ravel()[fwd].reshape(self._shape)
            compiler = self._get_compiler(tensor)
            return compiler.transform_numpy(tensor).reshape(self._shape)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def inverse_transform(self, tensor):
        self._validate_numel(tensor)
        _check_layout_invertible(self._layout)
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            from lego.backend.torch_layout import apply_inverse_layout_torch
            return apply_inverse_layout_torch(tensor, self._layout, self._shape,
                                              compiled=self._get_compiled_torch())
        if isinstance(tensor, np.ndarray):
            if self._composed_perm is not None:
                _, inv = self._composed_perm
                return tensor.ravel()[inv].reshape(self._shape)
            compiler = self._get_compiler(tensor)
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

    def __call__(self, tensor):
        """Shorthand: layout(tensor) == layout.transform(tensor)."""
        return self.transform(tensor)

    def __repr__(self):
        type_name = type(self._layout).__name__
        return f"LegoLayout(shape={self._shape}, layout={type_name})"

    def compose(self, other):
        """Compose layouts: result.transform(x) == other.transform(self.transform(x))."""
        if self._shape != other._shape:
            raise ValueError(f"Shape mismatch: {self._shape} vs {other._shape}")

        fwd_a, inv_a = LayoutCompiler(self._layout, self._shape, "i64").get_permutation_table()
        fwd_b, inv_b = LayoutCompiler(other._layout, other._shape, "i64").get_permutation_table()
        composed_fwd = fwd_b[fwd_a]
        composed_inv = inv_a[inv_b]

        result = LegoLayout(self._layout, self._shape)
        result._composed_perm = (composed_fwd, composed_inv)
        return result

    def __eq__(self, other):
        if not isinstance(other, LegoLayout):
            return NotImplemented
        if self._shape != other._shape:
            return False
        fwd_a, _ = LayoutCompiler(self._layout, self._shape, "i64").get_permutation_table()
        fwd_b, _ = LayoutCompiler(other._layout, other._shape, "i64").get_permutation_table()
        return np.array_equal(fwd_a, fwd_b)

    def __hash__(self):
        compiler = LayoutCompiler(self._layout, self._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        return hash((self._shape, fwd.tobytes()))

    def as_array(self, tensor):
        """Transform and wrap as LegoArray."""
        return LegoArray(self.transform(tensor), self)

    def to_dlpack(self, tensor):
        """Export transformed tensor via DLPack.

        Returns a DLPack capsule if the tensor supports it,
        otherwise returns the transformed tensor directly.
        """
        transformed = self.transform(tensor)
        if hasattr(transformed, '__dlpack__'):
            return transformed.__dlpack__()
        return transformed

    def from_dlpack(self, obj):
        """Import a DLPack-compatible object as a physical-order LegoArray.

        Accepts any object that np.from_dlpack can consume (tensors with
        __dlpack__, or raw DLPack capsules via framework-specific import).
        """
        if hasattr(obj, '__dlpack__'):
            data = np.from_dlpack(obj)
        else:
            # Raw capsule — try torch first, then numpy
            try:
                import torch
                t = torch.from_dlpack(obj)
                data = t.numpy()
            except (ImportError, RuntimeError):
                data = np.from_dlpack(obj)
        return LegoArray(data, self)


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


class TiledView:
    """Tiled multi-index view of a tensor.

    Transforms an (N, M) tensor into a (N//th, M//tw, th, tw) view
    via reshape + transpose — no data movement, just re-indexing.

    Example:
        layout = Tiled((8, 8), tile_shape=(4, 4))
        tiled = layout.transform(data)   # shape (2, 2, 4, 4)
        tiled[0, 1]                      # the (0,1) tile as a 4×4 array
        back = layout.inverse_transform(tiled)  # shape (8, 8)
    """

    def __init__(self, shape, tile_shape):
        if len(shape) != len(tile_shape):
            raise ValueError("shape and tile_shape must have the same rank")
        for i, (s, t) in enumerate(zip(shape, tile_shape)):
            if t <= 0:
                raise ValueError(
                    f"TiledView: tile size at dimension {i} must be > 0, got {t}"
                )
            if s % t != 0:
                raise ValueError(
                    f"TiledView: dimension {i} (size {s}) not divisible by "
                    f"tile size {t}. Consider tile sizes that evenly divide "
                    f"each dimension (e.g., factors of {s}: "
                    f"{[f for f in range(1, s+1) if s % f == 0]})."
                )

        self._original_shape = tuple(shape)
        self._tile_shape = tuple(tile_shape)
        self._tile_grid = tuple(s // t for s, t in zip(shape, tile_shape))
        self._shape = self._tile_grid + self._tile_shape
        self._tile_layout = OrderBy(Row(*shape)).TileBy(self._tile_grid, self._tile_shape)

        self._numel = 1
        for s in shape:
            self._numel *= s

        ndim = len(shape)
        # (g0, t0, g1, t1, ...) → (g0, g1, ..., t0, t1, ...)
        self._fwd_perm = list(range(0, 2 * ndim, 2)) + list(range(1, 2 * ndim, 2))
        # (g0, g1, ..., t0, t1, ...) → (g0, t0, g1, t1, ...)
        self._inv_perm = [0] * (2 * ndim)
        for i in range(ndim):
            self._inv_perm[2 * i] = i
            self._inv_perm[2 * i + 1] = ndim + i

    @property
    def shape(self):
        return self._shape

    @property
    def original_shape(self):
        return self._original_shape

    @property
    def tile_shape(self):
        return self._tile_shape

    @property
    def tile_grid(self):
        return self._tile_grid

    @property
    def numel(self):
        return self._numel

    @property
    def rank(self):
        return len(self._shape)

    def _interleaved_shape(self):
        result = []
        for g, t in zip(self._tile_grid, self._tile_shape):
            result.extend([g, t])
        return result

    def transform(self, tensor):
        """Reshape tensor into tiled multi-index view."""
        interleaved = self._interleaved_shape()
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            return tensor.reshape(interleaved).permute(self._fwd_perm)
        if isinstance(tensor, np.ndarray):
            return tensor.reshape(interleaved).transpose(self._fwd_perm)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def inverse_transform(self, tensor):
        """Reshape tiled multi-index view back to original shape."""
        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            return tensor.permute(self._inv_perm).contiguous().reshape(self._original_shape)
        if isinstance(tensor, np.ndarray):
            return tensor.transpose(self._inv_perm).reshape(self._original_shape)
        raise TypeError(f"Unsupported tensor type: {type(tensor)}")

    def create_tensor(self, dtype):
        arr = np.arange(self._numel, dtype=dtype).reshape(self._original_shape)
        return self.transform(arr)

    def __call__(self, tensor):
        return self.transform(tensor)

    def get_mlir(self, dtype="f32"):
        """Return MLIR text with lego.tile_by ops."""
        return LayoutCompiler(self._tile_layout, self._original_shape, dtype).mlir_text

    def __repr__(self):
        return f"TiledView(shape={self._original_shape}, tile={self._tile_shape})"


def Tiled(shape, tile_shape):
    """Tiled layout — multi-index view with (grid..., tile...) dimensions.

    Reshapes data from shape to (grid..., tile...) via reshape+transpose.
    No physical data reordering — just re-indexing.

    Example:
        Tiled((8, 8), (4, 4)).transform(data)  # shape becomes (2, 2, 4, 4)
    """
    return TiledView(shape, tile_shape)


def TiledPermute(shape, tile_shape):
    """Tiled layout that physically permutes data into tile-contiguous order.

    Unlike Tiled() which reshapes to higher rank (grid..., tile...),
    TiledPermute keeps the same shape but reorders the underlying data
    so that elements within each tile are contiguous in memory.

    This is useful for cache-friendly access patterns in tiled algorithms.

    Example:
        layout = TiledPermute((8, 8), (4, 4))
        result = layout.transform(data)   # shape stays (8, 8)
        # Elements within each 4x4 tile are now contiguous in memory
    """
    if len(shape) != len(tile_shape):
        raise ValueError("shape and tile_shape must have the same rank")
    for i, (s, t) in enumerate(zip(shape, tile_shape)):
        if t <= 0:
            raise ValueError(
                f"TiledPermute: tile size at dimension {i} must be > 0, got {t}"
            )
        if s % t != 0:
            raise ValueError(
                f"TiledPermute: dimension {i} (size {s}) not divisible by "
                f"tile size {t}."
            )
    tile_grid = tuple(s // t for s, t in zip(shape, tile_shape))
    tile_layout = OrderBy(Row(*shape)).TileBy(tile_grid, tile_shape)
    return LegoLayout(tile_layout, shape)


def Custom(layout_obj, shape):
    """Wrap an existing layout object."""
    return LegoLayout(layout_obj, shape)


def Transposed(shape):
    """Transpose layout — reverses dimension order.

    For 2D: equivalent to matrix transpose (same as ColMajor).
    For ND: reverses all dimensions (like np.transpose()).
    """
    perm = list(reversed(range(len(shape))))
    layout = GroupBy([shape], [OrderBy(RegP(shape, perm))])
    return LegoLayout(layout, shape)


def ZCurve(shape):
    """Z-order (Morton) curve layout for 2D arrays.

    Interleaves bits of row and column indices for cache-friendly
    2D spatial locality.
    """
    if len(shape) != 2:
        raise ValueError("ZCurve only supports 2D shapes")
    from sympy import Integer, Mod, floor

    n, m = shape
    nbits = max(math.ceil(math.log2(max(n, m, 2))), 1)

    def _interleave_bits(x, y, nb):
        result = Integer(0)
        for k in range(nb):
            result += Mod(floor(x / Integer(2**k)), Integer(2)) * Integer(1 << (2 * k))
            result += Mod(floor(y / Integer(2**k)), Integer(2)) * Integer(1 << (2 * k + 1))
        return result

    def _deinterleave_bits(z, nb):
        x = Integer(0)
        y = Integer(0)
        for k in range(nb):
            x += Mod(floor(z / Integer(2**(2 * k))), Integer(2)) * Integer(1 << k)
            y += Mod(floor(z / Integer(2**(2 * k + 1))), Integer(2)) * Integer(1 << k)
        return x, y

    def f_apply(args):
        i, j = args
        return _interleave_bits(i, j, nbits)

    def f_inv(z):
        return _deinterleave_bits(z, nbits)

    layout = GenP(shape, f_apply, f_inv)
    return LegoLayout(GroupBy([shape], [OrderBy(layout)]), shape)


def _sympy_xor(a, b, nbits):
    """Compute bitwise XOR using arithmetic decomposition."""
    from sympy import Integer, Mod, floor
    result = Integer(0)
    for k in range(nbits):
        bit_a = Mod(floor(a / Integer(2**k)), Integer(2))
        bit_b = Mod(floor(b / Integer(2**k)), Integer(2))
        xor_bit = Mod(bit_a + bit_b, Integer(2))
        result = result + xor_bit * Integer(2**k)
    return result


def Swizzle(shape, num_bits=None):
    """XOR-based swizzle layout for 2D arrays.

    Remaps columns by XORing with row bits. Used in GPU shared memory
    to avoid bank conflicts (CUTLASS/CuTe style).
    """
    if len(shape) != 2:
        raise ValueError("Swizzle only supports 2D shapes")

    nrows, ncols = shape
    if num_bits is None:
        num_bits = int(math.log2(ncols)) if ncols > 1 else 0
    mask = (1 << num_bits) - 1

    from sympy import Integer, Mod

    col_bits = max(math.ceil(math.log2(max(ncols, 2))), 1)

    def f_apply(args):
        i, j = args
        masked_i = Mod(i, Integer(mask + 1))
        swizzled_j = _sympy_xor(j, masked_i, col_bits)
        return i * Integer(ncols) + swizzled_j

    def f_inv(flat):
        from sympy import floor
        r = floor(flat / Integer(ncols))
        phys_col = Mod(flat, Integer(ncols))
        masked_r = Mod(r, Integer(mask + 1))
        log_col = _sympy_xor(phys_col, masked_r, col_bits)
        return (r, log_col)

    layout = GenP(shape, f_apply, f_inv)
    return LegoLayout(GroupBy([shape], [OrderBy(layout)]), shape)


def BlockCyclic(shape, block_size, num_procs):
    """Block-cyclic distribution layout (1D).

    Used in distributed computing (ScaLAPACK-style).
    """
    if len(shape) != 1:
        raise ValueError("BlockCyclic currently supports 1D only")
    N = shape[0]

    from sympy import floor, Mod, Integer

    local_blocks = (N // block_size + num_procs - 1) // num_procs

    def f_apply(args):
        i = args[0] if hasattr(args, '__len__') else args
        blk = floor(i / Integer(block_size))
        proc = Mod(blk, Integer(num_procs))
        local_blk = floor(blk / Integer(num_procs))
        offset = Mod(i, Integer(block_size))
        return proc * Integer(local_blocks * block_size) + local_blk * Integer(block_size) + offset

    def f_inv(flat):
        proc = floor(flat / Integer(local_blocks * block_size))
        remainder = Mod(flat, Integer(local_blocks * block_size))
        local_blk = floor(remainder / Integer(block_size))
        offset = Mod(remainder, Integer(block_size))
        blk = local_blk * Integer(num_procs) + proc
        return (blk * Integer(block_size) + offset,)

    layout = GenP(shape, f_apply, f_inv)
    return LegoLayout(GroupBy([shape], [OrderBy(layout)]), shape)


# ============================================================================
# BatchedLayout
# ============================================================================

class BatchedLayout:
    """Applies a base layout independently across batch dimensions."""

    def __init__(self, base_layout, batch_shape):
        self._base = base_layout
        self._batch_shape = tuple(batch_shape)
        self._shape = self._batch_shape + base_layout.shape

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return len(self._shape)

    @property
    def numel(self):
        total = 1
        for s in self._shape:
            total *= s
        return total

    def transform(self, tensor):
        batch_dims = len(self._batch_shape)
        inner_shape = self._base.shape

        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            batch_total = 1
            for d in tensor.shape[:batch_dims]:
                batch_total *= d
            flat = tensor.reshape(batch_total, -1)
            fwd, _ = self._base._get_perm_tensors(tensor.device)
            result = flat[:, fwd]
            return result.reshape(self._shape)

        if isinstance(tensor, np.ndarray):
            batch_total = int(np.prod(tensor.shape[:batch_dims]))
            flat = tensor.reshape(batch_total, -1)
            compiler = LayoutCompiler(self._base._layout, inner_shape, "i64")
            fwd, _ = compiler.get_permutation_table()
            result = flat[:, fwd]
            return result.reshape(self._shape)

        raise TypeError(f"Unsupported: {type(tensor)}")

    def inverse_transform(self, tensor):
        batch_dims = len(self._batch_shape)
        inner_shape = self._base.shape

        if _HAS_TORCH and isinstance(tensor, torch.Tensor):
            batch_total = 1
            for d in tensor.shape[:batch_dims]:
                batch_total *= d
            flat = tensor.reshape(batch_total, -1)
            _, inv = self._base._get_perm_tensors(tensor.device)
            result = flat[:, inv]
            return result.reshape(self._shape)

        if isinstance(tensor, np.ndarray):
            batch_total = int(np.prod(tensor.shape[:batch_dims]))
            flat = tensor.reshape(batch_total, -1)
            compiler = LayoutCompiler(self._base._layout, inner_shape, "i64")
            _, inv = compiler.get_permutation_table()
            result = flat[:, inv]
            return result.reshape(self._shape)

        raise TypeError(f"Unsupported: {type(tensor)}")

    def __repr__(self):
        return f"BatchedLayout(batch={self._batch_shape}, base={self._base!r})"

    def __call__(self, tensor):
        return self.transform(tensor)


def Batched(base_layout, batch_shape):
    """Add batch dimensions to a layout."""
    return BatchedLayout(base_layout, batch_shape)


# ============================================================================
# LegoArray (NumPy wrapper)
# ============================================================================

class LegoArray:
    """NumPy array wrapper carrying layout metadata.

    Storage is in physical order. Implements __array__ for
    transparent conversion to logical order.
    """

    def __init__(self, data, layout):
        self._data = np.asarray(data)
        self._layout = layout

    def __array__(self, dtype=None, copy=None):
        result = self._layout.inverse_transform(self._data)
        if dtype is not None:
            result = result.astype(dtype)
        if copy:
            result = result.copy()
        return result

    @property
    def shape(self):
        return self._layout.shape

    @property
    def dtype(self):
        return self._data.dtype

    def to_logical(self):
        return self._layout.inverse_transform(self._data)

    def to_physical(self):
        return self._data.copy()

    def __repr__(self):
        return f"LegoArray(shape={self.shape}, dtype={self.dtype}, layout={self._layout!r})"
