"""
LEGO Tensor Subclass (Layer 2)

A ``__torch_dispatch__``-based tensor subclass that carries layout metadata.
Supports virtual (annotate) and physical (rearrange) modes with 4-tier op
classification.

Tier 1 — Propagate:  pointwise ops run on physical data, layout copied
Tier 2 — Transform:  transpose/permute expressed as layout algebra
Tier 3 — Drop+warn:  unsupported ops strip layout with one-time warning
Tier 4 — LEGO kernel: dispatched to lego::* ops (Phase 1)
"""

import threading
import warnings
import torch

# One-time warning tracker
_warned_ops: set[str] = set()


# ============================================================================
# Helper: unwrap / rewrap
# ============================================================================

def _unwrap(x):
    if isinstance(x, LegoTensor):
        return x._data
    return x


def _unwrap_args(args):
    if isinstance(args, tuple):
        return tuple(_unwrap_args(a) for a in args)
    if isinstance(args, list):
        return [_unwrap_args(a) for a in args]
    return _unwrap(args)


def _first_lego(args):
    """Return the first LegoTensor found in args, or None."""
    if isinstance(args, (tuple, list)):
        for a in args:
            r = _first_lego(a)
            if r is not None:
                return r
    elif isinstance(args, LegoTensor):
        return args
    return None


# ============================================================================
# Op classification tables  (populated at end of module)
# ============================================================================

_TIER1: set = set()
_TIER2: set = set()
_TIER4: set = set()
_FULL_REDUCTIONS: set = set()
_DIM_REDUCTIONS: set = set()


# ============================================================================
# Layout algebra wrappers
# ============================================================================

class TransposedLayout:
    """Layout wrapper recording a dimension permutation on top of a base layout."""

    def __init__(self, base, perm):
        self._base = base
        self._perm = tuple(perm)
        base_shape = base.shape if hasattr(base, "shape") else base._shape
        self._shape = tuple(base_shape[p] for p in perm)

    @property
    def shape(self):
        return self._shape

    def compose(self, other):
        """Compose with another layout.

        Returns other — the new layout takes precedence since the
        transpose permutation is already reflected in the data view.
        """
        return other

    def transpose(self, dim0, dim1):
        perm = list(self._perm)
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return TransposedLayout(self._base, perm)

    def permute(self, dims):
        new_perm = tuple(self._perm[d] for d in dims)
        return TransposedLayout(self._base, new_perm)

    def __repr__(self):
        return f"TransposedLayout(base={self._base!r}, perm={self._perm})"


def _transpose_layout(layout, dim0=0, dim1=1):
    if isinstance(layout, TransposedLayout):
        return layout.transpose(dim0, dim1)
    ndim = len(layout.shape) if hasattr(layout, "shape") else len(layout._shape)
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return TransposedLayout(layout, perm)


def _permute_layout(layout, dims):
    if isinstance(layout, TransposedLayout):
        return layout.permute(dims)
    return TransposedLayout(layout, dims)


# ============================================================================
# LegoTensor
# ============================================================================

class LegoTensor(torch.Tensor):
    """Tensor subclass carrying LEGO layout metadata.

    Uses ``__torch_dispatch__`` (not ``__torch_function__``) so it survives
    ``torch.compile`` tracing.
    """

    @staticmethod
    def __new__(cls, data, layout, is_physical=False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.shape,
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            requires_grad=data.requires_grad,
            storage_offset=data.storage_offset(),
        )

    def __init__(self, data, layout, is_physical=False):
        self._data = data
        self._layout = layout
        self._is_physical = is_physical

    @property
    def lego_layout(self):
        return self._layout

    def __repr__(self):
        mode = "physical" if self._is_physical else "virtual"
        return (
            f"LegoTensor(shape={list(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, layout={self._layout!r}, mode={mode})"
        )

    # -- Serialization (DataLoader, multiprocessing) --

    def __reduce_ex__(self, protocol):
        return (_rebuild, (self._data, self._layout, self._is_physical))

    # -- torch.compile / torch.export compatibility --

    def __tensor_flatten__(self):
        return ["_data"], {"layout": self._layout, "is_physical": self._is_physical}

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls(inner_tensors["_data"], metadata["layout"], metadata["is_physical"])

    # -- Dispatch --

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if func in _TIER1:
            return _dispatch_tier1(func, args, kwargs)

        if func in _TIER2:
            return _dispatch_tier2(func, args, kwargs)

        # Full reductions (no dim arg) — safe on physical data, returns scalar
        if func in _FULL_REDUCTIONS:
            return func(*_unwrap_args(args), **{k: _unwrap(v) for k, v in kwargs.items()})

        # Dim-reductions: must inverse-rearrange physical data first
        if func in _DIM_REDUCTIONS:
            return _dispatch_dim_reduction(func, args, kwargs)

        if func in _TIER4:
            return _dispatch_tier4(func, args, kwargs)

        # Tier 3 — fallback
        return _dispatch_tier3(func, args, kwargs)


def _rebuild(data, layout, is_physical):
    return LegoTensor(data, layout, is_physical)


# ============================================================================
# Tier dispatchers
# ============================================================================

def _dispatch_tier1(func, args, kwargs):
    lt = _first_lego(args)
    layout = lt._layout
    phys = lt._is_physical
    result = func(*_unwrap_args(args), **{k: _unwrap(v) for k, v in kwargs.items()})
    # Only propagate layout when output shape matches input (pointwise).
    # Dim-reductions change shape, so layout doesn't propagate, but
    # the computation is still correct on the physical data.
    if isinstance(result, torch.Tensor) and result.shape == lt.shape:
        return LegoTensor(result, layout, phys)
    return result


def _dispatch_dim_reduction(func, args, kwargs):
    """Dim-reductions: correct on virtual data, need inverse on physical."""
    lt = _first_lego(args)
    if lt is not None and lt._is_physical:
        # Physical data is in layout order — inverse-rearrange to logical
        # order before reducing along a dimension.
        from lego.backend.compiler import LayoutCompiler
        import numpy as np
        layout = lt._layout
        base = layout._base if hasattr(layout, "_base") else layout
        compiler = LayoutCompiler(base._layout, base._shape, "i64")
        _, inv = compiler.get_permutation_table()
        inv_t = torch.from_numpy(np.ascontiguousarray(inv)).to(lt._data.device)
        logical = lt._data.reshape(-1)[inv_t].reshape(lt._data.shape)
        new_args = list(args)
        new_args[0] = logical
        return func(*new_args, **{k: _unwrap(v) for k, v in kwargs.items()})
    # Virtual data: data is still in logical order, safe to reduce directly
    return func(*_unwrap_args(args), **{k: _unwrap(v) for k, v in kwargs.items()})


def _dispatch_tier2(func, args, kwargs):
    lt = _first_lego(args)
    if lt is None:
        return func(*_unwrap_args(args), **kwargs)

    layout = lt._layout
    aten = torch.ops.aten

    if func == aten.t.default:
        new_layout = _transpose_layout(layout)
    elif func == aten.transpose.int:
        new_layout = _transpose_layout(layout, args[1], args[2])
    elif func == aten.permute.default:
        new_layout = _permute_layout(layout, args[1])
    else:
        return _dispatch_tier3(func, args, kwargs)

    result = func(*_unwrap_args(args), **{k: _unwrap(v) for k, v in kwargs.items()})
    return LegoTensor(result, new_layout, lt._is_physical)


def _dispatch_tier3(func, args, kwargs):
    name = str(func.name()) if hasattr(func, "name") else str(func)
    if name not in _warned_ops:
        _warned_ops.add(name)
        warnings.warn(f"'{name}' is not layout-aware. Layout metadata dropped.", stacklevel=4)
    return func(*_unwrap_args(args), **{k: _unwrap(v) for k, v in kwargs.items()})


_TIER4_MAP: dict = {}
_TIER4_LOCK = threading.Lock()


def _dispatch_tier4(func, args, kwargs):
    """Tier 4: redirect to lego::* op if registered, else standard op."""
    if not _TIER4_MAP:
        with _TIER4_LOCK:
            if not _TIER4_MAP:
                _populate_tier4_map()
    lego_op = _TIER4_MAP.get(func)
    raw_args = _unwrap_args(args)
    raw_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
    if lego_op is not None:
        return lego_op(*raw_args, **raw_kwargs)
    return func(*raw_args, **raw_kwargs)


def _populate_tier4_map():
    import lego.torch.ops  # noqa: F401 — ensures lego::mm/bmm are registered
    aten = torch.ops.aten
    _TIER4_MAP[aten.mm.default] = torch.ops.lego.mm
    _TIER4_MAP[aten.bmm.default] = torch.ops.lego.bmm
    _TIER4_MAP[aten.addmm.default] = None  # no lego:: equivalent yet


# ============================================================================
# Populate op tables
# ============================================================================

def _populate():
    aten = torch.ops.aten
    _TIER1.update([
        aten.add.Tensor, aten.add.Scalar,
        aten.sub.Tensor, aten.sub.Scalar,
        aten.mul.Tensor, aten.mul.Scalar,
        aten.div.Tensor, aten.div.Scalar,
        aten.neg.default,
        aten.abs.default,
        aten.relu.default,
        aten.gelu.default,
        aten.silu.default,
        aten.sigmoid.default,
        aten.tanh.default,
        aten.sin.default, aten.cos.default,
        aten.exp.default, aten.log.default,
        aten.sqrt.default, aten.rsqrt.default,
        aten.clone.default,
        aten.copy_.default,
        aten.fill_.Scalar,
        aten.zero_.default,
        aten.eq.Tensor, aten.eq.Scalar,
        aten.ne.Tensor, aten.ne.Scalar,
        aten.lt.Tensor, aten.lt.Scalar,
        aten.le.Tensor, aten.le.Scalar,
        aten.gt.Tensor, aten.gt.Scalar,
        aten.ge.Tensor, aten.ge.Scalar,
        aten._to_copy.default,
        aten.dropout.default,
        aten.native_dropout.default,
    ])

    _TIER2.update([
        aten.t.default,
        aten.transpose.int,
        aten.permute.default,
    ])

    _TIER4.update([
        aten.mm.default,
        aten.bmm.default,
        aten.addmm.default,
    ])

    _FULL_REDUCTIONS.update([
        aten.sum.default,
        aten.mean.default,
        aten.prod.default,
        aten.amax.default,
        aten.amin.default,
    ])

    _DIM_REDUCTIONS.update([
        aten.sum.dim_IntList,
        aten.mean.dim,
        aten.prod.dim_int,
    ])


_populate()
