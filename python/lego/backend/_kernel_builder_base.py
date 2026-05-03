"""Shared base classes for CPU and GPU kernel builders.

This module factors out the infrastructure that is duplicated between
:mod:`cpu_builder` and :mod:`gpu_builder`:

- :class:`_KernelContextBase` — arithmetic helpers, load_flat/store_flat,
  constants, math ops, and the ``for_range`` helper that are identical in
  both ``CPUKernelContext`` and ``KernelContext``.

- :data:`_DESC_CACHE` / :data:`_DESC_CACHE_DESC` — module-level memref
  descriptor cache shared by CPU (and optionally GPU-JIT) fast-paths.

- :func:`_cached_memref_ptr` — the cache lookup / fill helper.

- :func:`clear_descriptor_cache` — public eviction API.

- :func:`_make_callable_base` — the hot-path ``jit_fn`` builder that wraps
  an MLIR ``ExecutionEngine`` in a Python callable with descriptor caching,
  scalar marshalling, and pre-cast packed-array dispatch.  CPU uses this
  directly; GPU JIT may adopt it in future.

GPU-specific features (``block_id``, ``thread_id``, ``barrier``,
``tensor_core``, MMA, warp shuffles) stay in ``gpu_builder.KernelContext``.

CPU-specific features (``tile_id``, ``set_layout``) stay in
``cpu_builder.CPUKernelContext``.
"""

import ctypes
from typing import Dict

import numpy as np

from lego.mlir.ir import (
    F32Type, IntegerType, InsertionPoint,
)
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import scf as scf_dialect
from lego.mlir.dialects import memref as memref_dialect

from lego.backend._ops import _index_const

# ============================================================================
# Memref descriptor cache
# ============================================================================
# ``get_ranked_memref_descriptor()`` costs ~23 µs per call (Python + ctypes
# struct allocation).  For benchmarks that call the same kernel 1000× with
# identical numpy arrays, this adds ~0.1 ms of overhead per call — dominating
# the measured kernel time and masking vectorization wins.
#
# Cache key: (data_ptr, shape, strides).  Using arr.ctypes.data (the actual
# C-level buffer pointer) instead of id(arr) avoids the id-reuse hazard: two
# different numpy arrays that share the same base memory hash identically,
# which is correct — they share the same descriptor.  The key naturally
# invalidates when the underlying buffer moves (reshape to non-contiguous,
# in-place realloc, etc.) because the data pointer or strides change.
#
# We also cache ctypes.pointer(ctypes.pointer(desc)) — the "double-pointer"
# that engine.invoke expects — so the hot path is a single dict lookup + ~1 µs
# of pointer construction (down from ~104 µs raw).
#
# The cache is a plain module-level dict (no WeakRef needed: the descriptor
# struct keeps a reference to the numpy array's memory, so GC will not free
# the underlying buffer while a live descriptor exists).

_DESC_CACHE: Dict[tuple, object] = {}       # key → ctypes double-pointer object
_DESC_CACHE_DESC: Dict[tuple, object] = {}  # key → raw descriptor (keeps alive)


def _cached_memref_ptr(arr: np.ndarray):
    """Return a cached ctypes pointer-of-pointer for *arr*'s ranked memref descriptor.

    The result is suitable for passing directly to ``engine.invoke``.  The
    descriptor and the double-pointer object are kept alive in module-level
    dicts for the lifetime of the process.
    """
    from lego.mlir.runtime import get_ranked_memref_descriptor
    key = (arr.ctypes.data, arr.shape, arr.strides)
    pp = _DESC_CACHE.get(key)
    if pp is None:
        desc = get_ranked_memref_descriptor(arr)
        pp = ctypes.pointer(ctypes.pointer(desc))
        _DESC_CACHE_DESC[key] = desc   # keep descriptor alive
        _DESC_CACHE[key] = pp
    return pp


def clear_descriptor_cache():
    """Evict all cached memref descriptors.

    Call this if you intentionally reuse numpy array memory for a different
    buffer (e.g. ``arr[:] = new_data`` after an in-place realloc).  Under
    normal benchmark patterns (fixed arrays, repeated calls) this is never
    needed.
    """
    _DESC_CACHE.clear()
    _DESC_CACHE_DESC.clear()


# ============================================================================
# _KernelContextBase — shared arithmetic / loop helpers
# ============================================================================

class _KernelContextBase:
    """Shared helpers for CPUKernelContext and KernelContext.

    Subclasses must set ``self._buf_vals`` (list of MLIR Values for the
    kernel's buffer arguments) before calling any inherited method.

    GPU-specific methods (block/thread IDs, barriers, MMA, warp shuffles)
    are implemented in ``gpu_builder.KernelContext``.

    CPU-specific methods (``tile_id``, ``set_layout``) are implemented in
    ``cpu_builder.CPUKernelContext``.
    """

    # --- Flat load/store (no layout) ---

    def load_flat(self, buf_index: int, flat_idx):
        """Load from buffer *buf_index* at flat index *flat_idx*."""
        return memref_dialect.LoadOp(self._buf_vals[buf_index], [flat_idx]).result

    def store_flat(self, value, buf_index: int, flat_idx):
        """Store *value* to buffer *buf_index* at flat index *flat_idx*."""
        memref_dialect.StoreOp(value, self._buf_vals[buf_index], [flat_idx])

    # --- Constants ---

    def const_f32(self, value) -> object:
        """Emit an f32 constant MLIR Value."""
        return arith_dialect.ConstantOp(F32Type.get(), float(value)).result

    def const_index(self, value) -> object:
        """Emit an index constant MLIR Value."""
        return _index_const(int(value))

    # --- Arithmetic (float) ---

    def addf(self, a, b):
        return arith_dialect.AddFOp(a, b).result

    def mulf(self, a, b):
        return arith_dialect.MulFOp(a, b).result

    def subf(self, a, b):
        return arith_dialect.SubFOp(a, b).result

    def divf(self, a, b):
        return arith_dialect.DivFOp(a, b).result

    # --- Arithmetic (integer) ---

    def addi(self, a, b):
        return arith_dialect.AddIOp(a, b).result

    def muli(self, a, b):
        return arith_dialect.MulIOp(a, b).result

    def subi(self, a, b):
        return arith_dialect.SubIOp(a, b).result

    # --- Convenience aliases ---

    def add(self, a, b):
        """Alias for addf (float add)."""
        return self.addf(a, b)

    def mul(self, a, b):
        """Alias for mulf (float multiply)."""
        return self.mulf(a, b)

    # --- Math operations ---

    def exp(self, val):
        """Emit math.exp."""
        from lego.mlir.ir import Operation
        return Operation.create("math.exp", results=[val.type], operands=[val]).result

    def sqrt(self, val):
        """Emit math.sqrt."""
        from lego.mlir.ir import Operation
        return Operation.create("math.sqrt", results=[val.type], operands=[val]).result

    def rsqrt(self, val):
        """Emit math.rsqrt."""
        from lego.mlir.ir import Operation
        return Operation.create("math.rsqrt", results=[val.type], operands=[val]).result

    # --- Comparisons ---

    def lt(self, a, b):
        """Unsigned-less-than comparison on index values."""
        return arith_dialect.CmpIOp(arith_dialect.CmpIPredicate.ult, a, b).result

    def eq(self, a, b):
        """Equality comparison on integer/index values."""
        return arith_dialect.CmpIOp(arith_dialect.CmpIPredicate.eq, a, b).result

    # --- SCF loop helper ---

    def for_range(self, n, body_fn, init_vals=None):
        """Emit an ``scf.for`` loop from 0 to *n* (step 1).

        Args:
            n:         Upper bound (Python int or MLIR index Value).
            body_fn:   Called with ``(iv[, *carry_args])``; returns the
                       updated carry values if *init_vals* is not None.
            init_vals: Optional initial carried accumulator values.

        Returns:
            ``None`` if *init_vals* is ``None``, else a list of result Values.
        """
        if isinstance(n, int):
            n = _index_const(n)
        if init_vals is None:
            loop = scf_dialect.ForOp(_index_const(0), n, _index_const(1))
            with InsertionPoint(loop.body):
                body_fn(loop.induction_variable)
                scf_dialect.YieldOp([])
            return None
        loop = scf_dialect.ForOp(_index_const(0), n, _index_const(1), init_vals)
        with InsertionPoint(loop.body):
            iv = loop.induction_variable
            carry_args = list(loop.inner_iter_args)
            updated = body_fn(iv, *carry_args)
            scf_dialect.YieldOp(updated)
        return list(loop.results)

    # --- SCF conditional ---

    def if_(self, cond, then_fn):
        """Emit ``scf.if`` with no else branch."""
        if_op = scf_dialect.IfOp(cond, has_else=False)
        with InsertionPoint(if_op.then_block):
            then_fn()
            scf_dialect.YieldOp([])


# ============================================================================
# _make_callable — hot-path JIT wrapper builder
# ============================================================================

def _make_callable(engine, name: str, scalar_params: list, buffers: list):
    """Build a Python wrapper that dispatches to a JIT-compiled MLIR function.

    This is the hot-path wrapper used by ``CPUKernelBuilder._make_callable``.
    It is factored out here so it can be unit-tested independently and reused
    if GPU-JIT support is added later.

    Hot-path optimisations
    ----------------------
    1. **Descriptor cache** — ``get_ranked_memref_descriptor()`` costs ~23 µs
       per numpy array.  We cache the descriptor *and* the
       ``ctypes.pointer(pointer(desc))`` object keyed by
       ``(data_ptr, shape, strides)`` so repeated calls with identical arrays
       pay only a dict lookup + ~1 µs.

    2. **Pre-built dtype → ctype map** — built once at wrapper-creation time.

    3. **Pre-cast packed array** — ``engine.lookup(name)`` returns a raw
       ctypes callable.  We pre-cast all argument pointers into a
       ``(c_void_p * N)`` array once per (scalar, buffer) combination; the
       hot path is a single direct call to that pre-cast array.

    4. **Single-slot last-call cache** — for the overwhelmingly common
       benchmark pattern (same args × 1000 calls), one equality check is
       faster than any dict lookup.

    Args:
        engine:        MLIR ``ExecutionEngine`` instance (already compiled).
        name:          Kernel function name (used for ``engine.lookup``).
        scalar_params: List of dtype strings (``"f32"``, ``"i64"``, …) for
                       scalar parameters, in declaration order.
        buffers:       List of ``LayoutBuffer`` descriptors (or objects with
                       a ``.numel`` attribute), one per memref parameter.

    Returns:
        A Python callable ``jit_fn(*args)`` where *args* is
        ``(scalar_0, ..., scalar_k, array_0, ..., array_m)``.
    """
    _dtype_to_ctype = {
        "f32": ctypes.c_float,
        "f64": ctypes.c_double,
        "i32": ctypes.c_int32,
        "i64": ctypes.c_int64,
        "i16": ctypes.c_int16,
        "i8":  ctypes.c_int8,
    }

    n_scalars = len(scalar_params)
    n_buffers = len(buffers)
    n_cargs = n_scalars + n_buffers

    # Pre-build the ctypes callable once: avoids per-call symbol lookup
    # and CFUNCTYPE construction inside engine.invoke.
    try:
        _ciface_fn = engine.lookup(name)
    except RuntimeError:
        _ciface_fn = None   # fall back to engine.invoke if lookup fails

    _scalar_cache: dict = {}
    _call_cache: dict = {}
    _last: list = [None, None, None]  # [last_key, last_cargs, last_packed]

    def _build_packed(cargs):
        p = (ctypes.c_void_p * n_cargs)()
        for j, v in enumerate(cargs):
            p[j] = ctypes.cast(v, ctypes.c_void_p)
        return p

    def jit_fn(*args):
        """Invoke the JIT-compiled CPU kernel.

        Args:
            *args: scalar args (matching scalar_params types), then one
                   numpy array per LayoutBuffer (in declaration order).
        """
        if len(args) != n_cargs:
            raise ValueError(
                f"Expected {n_cargs} arg(s) "
                f"({n_scalars} scalar(s) + {n_buffers} buffer(s)), "
                f"got {len(args)}"
            )

        # Single-slot last-call cache
        if n_scalars == 1 and n_buffers == 2:
            key = (args[0], id(args[1]), id(args[2]))
        elif n_scalars == 0 and n_buffers == 2:
            key = (id(args[0]), id(args[1]))
        else:
            key = tuple(args[:n_scalars]) + tuple(id(a) for a in args[n_scalars:])

        if key == _last[0]:
            if _ciface_fn is not None:
                _ciface_fn(_last[2])
            else:
                engine.invoke(name, *_last[1])
            return

        # Full _call_cache
        cached = _call_cache.get(key)
        if cached is not None:
            cargs, packed = cached
        else:
            cargs = [None] * n_cargs

            for i, (dtype_str, py_val) in enumerate(
                    zip(scalar_params, args[:n_scalars])):
                kk = (dtype_str, py_val)
                c_wrap = _scalar_cache.get(kk)
                if c_wrap is None:
                    c_ty = _dtype_to_ctype.get(dtype_str, ctypes.c_float)
                    c_wrap = (c_ty * 1)(py_val)
                    _scalar_cache[kk] = c_wrap
                cargs[i] = c_wrap

            for i, arr in enumerate(args[n_scalars:]):
                if isinstance(arr, np.ndarray):
                    arr_c = arr if arr.flags['C_CONTIGUOUS'] else np.ascontiguousarray(arr)
                    cargs[n_scalars + i] = _cached_memref_ptr(arr_c)
                else:
                    cargs[n_scalars + i] = arr

            packed = _build_packed(cargs) if _ciface_fn is not None else None
            _call_cache[key] = (cargs, packed)

        _last[0] = key
        _last[1] = cargs
        _last[2] = packed
        if _ciface_fn is not None:
            _ciface_fn(packed)
        else:
            engine.invoke(name, *cargs)

    return jit_fn
