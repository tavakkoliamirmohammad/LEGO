"""
CPU kernel builder for the LEGO CPU vector pipeline.

Mirrors gpu_builder.py's KernelBuilder but targets the CPU:
  - Builds a func.func (no gpu.launch) with scf.for loops over the grid.
  - Compiles through ``lego-to-x86-vector`` (or ``lego-to-arm-neon``) via
    the registered pass pipelines.
  - JIT-executes via MLIR ExecutionEngine.

Usage::

    from lego.backend.cpu_builder import CPUKernelBuilder, LayoutBuffer
    from lego.core import Row
    from lego.backend.compiler import DType

    a = LayoutBuffer(Row(N), shape=(N,))
    b = LayoutBuffer(Row(N), shape=(N,))

    def axpy_body(ctx):
        ...  # emit MLIR via ctx

    builder = CPUKernelBuilder(buffers=[a, b], kernel_body=axpy_body, name="axpy")
    jit_fn = builder.compile_and_run()
"""

import ctypes
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from lego.mlir.ir import (
    Context, Location, Module, InsertionPoint,
    IndexType, MemRefType, FunctionType, StringAttr, UnitAttr,
    F32Type,
)
from lego.mlir.dialects import func as func_dialect
from lego.mlir.dialects import scf as scf_dialect
from lego.mlir.dialects import arith as arith_dialect
from lego.mlir.dialects import memref as memref_dialect
from lego.mlir.passmanager import PassManager
from lego.mlir.execution_engine import ExecutionEngine
from lego.mlir.runtime import get_ranked_memref_descriptor

from lego.backend.dialects.lego_dialect import register as register_lego
from lego.backend._ops import _index_const, _emit_apply
from lego.backend.compiler import DType, _get_mlir_element_type
from lego.backend.symbolic import emit_layout_from_python
from lego.backend.gpu_builder import LayoutBuffer  # reuse the same descriptor


# ============================================================================
# Memref descriptor cache
# ============================================================================
# get_ranked_memref_descriptor() takes ~23 µs per call (Python + ctypes struct
# allocation).  For benchmarks that call the same kernel 1000× with the same
# numpy arrays, this adds ~0.1 ms of pure Python overhead per call — dominating
# the measured kernel time and masking vectorization wins.
#
# Cache key: (data_ptr, shape, strides).  Using arr.ctypes.data (the actual
# C-level buffer pointer) instead of id(arr) avoids the id-reuse hazard: two
# different numpy arrays that share the same base memory will hash identically,
# which is correct — they share the same descriptor.  The key naturally
# invalidates when the underlying buffer moves (reshape to non-contiguous,
# in-place realloc, etc.) because the data pointer or strides change.
#
# We also cache ctypes.pointer(ctypes.pointer(desc)) — the "double-pointer"
# that engine.invoke expects — so the hot path is a single dict lookup + ~1 µs
# of pointer construction (from 104 µs raw).
#
# The cache is a plain module-level dict (no WeakRef needed: the descriptor
# struct keeps a reference to the numpy array's memory, so GC will not free
# the underlying buffer while a live descriptor exists).

_DESC_CACHE: Dict[tuple, object] = {}   # key → ctypes double-pointer object
_DESC_CACHE_DESC: Dict[tuple, object] = {}  # key → raw descriptor (keeps it alive)


def _cached_memref_ptr(arr: np.ndarray):
    """Return a cached ctypes pointer-of-pointer for *arr*'s ranked memref descriptor.

    The result is suitable for passing directly to ``engine.invoke``.  The
    descriptor and the double-pointer object are kept alive in module-level
    dicts for the lifetime of the process (or until the cache is manually
    cleared with ``clear_descriptor_cache()``).
    """
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
# CPU target registry (mirrors GPUTarget in gpu_builder.py)
# ============================================================================

_CPU_TARGETS: Dict[str, "CPUTarget"] = {}


@dataclass
class CPUTarget:
    """Describes a CPU compilation backend (pipeline + default options)."""
    name: str
    pipeline: str
    default_cpu: Optional[str] = None

    def pipeline_string(self, cpu: Optional[str] = None) -> str:
        c = cpu or self.default_cpu or ""
        if c:
            return f"builtin.module({self.pipeline}{{cpu={c}}})"
        return f"builtin.module({self.pipeline})"

    def register(self):
        _CPU_TARGETS[self.name] = self
        return self


CPUTarget(
    name="x86",
    pipeline="lego-to-x86-vector",
    default_cpu="skx",
).register()

CPUTarget(
    name="arm-neon",
    pipeline="lego-to-arm-neon",
    default_cpu="cortex-a76",
).register()

# Alias: plain "cpu" → x86 by default (auto-detected later if needed)
CPUTarget(
    name="cpu",
    pipeline="lego-to-x86-vector",
    default_cpu="skx",
).register()

# Scalar (no vectorization) baseline — same calling convention as x86/arm-neon
# but compiled through lego-to-llvm (scalar SCF loops, no vector ops).
# Used for apples-to-apples speedup isolation: compare scalar-JIT vs vec-JIT
# to measure the vectorization benefit independently of JIT startup overhead.
CPUTarget(
    name="scalar",
    pipeline="lego-to-llvm",
    default_cpu=None,
).register()


# ============================================================================
# Kernel context — CPU equivalent of KernelContext in gpu_builder.py
# ============================================================================

class CPUKernelContext:
    """Context passed to a user kernel body when building CPU MLIR.

    Provides layout-aware load/store, arithmetic helpers, and loop helpers.
    Mirrors KernelContext in gpu_builder.py but without GPU-specific ops
    (no block_id/thread_id/barrier/tensor_core/shared memory).

    Additionally exposes ``tile_id``: the current outer-loop induction
    variable (the "which tile are we computing" index).
    """

    def __init__(self, buf_vals, buf_descs, tile_id=None):
        self._buf_vals = buf_vals
        self._buf_descs = list(buf_descs)
        self._tile_id = tile_id

    @property
    def tile_id(self):
        """Outer-loop tile index (MLIR index Value, or None if no outer loop)."""
        return self._tile_id

    # --- Layout-aware load/store ---

    def _get_buf_layout_ir(self, buf_index):
        desc = self._buf_descs[buf_index]
        return emit_layout_from_python(desc.layout, {})

    def load(self, buf_index: int, indices: list):
        layout_val = self._get_buf_layout_ir(buf_index)
        memref = self._buf_vals[buf_index]
        flat_idx = _emit_apply(layout_val, indices)
        return memref_dialect.LoadOp(memref, [flat_idx]).result

    def store(self, value, buf_index: int, indices: list):
        layout_val = self._get_buf_layout_ir(buf_index)
        memref = self._buf_vals[buf_index]
        flat_idx = _emit_apply(layout_val, indices)
        memref_dialect.StoreOp(value, memref, [flat_idx])

    def load_flat(self, buf_index: int, flat_idx):
        return memref_dialect.LoadOp(self._buf_vals[buf_index], [flat_idx]).result

    def store_flat(self, value, buf_index: int, flat_idx):
        memref_dialect.StoreOp(value, self._buf_vals[buf_index], [flat_idx])

    def set_layout(self, buf_index: int, new_layout):
        from lego.backend.gpu_builder import LayoutBuffer as _LB
        self._buf_descs[buf_index] = _LB(
            layout=new_layout,
            shape=self._buf_descs[buf_index].shape,
            dtype=self._buf_descs[buf_index].dtype,
            shared=self._buf_descs[buf_index].shared,
        )

    # --- Arithmetic helpers (same as KernelContext) ---

    def addf(self, a, b):
        return arith_dialect.AddFOp(a, b).result

    def mulf(self, a, b):
        return arith_dialect.MulFOp(a, b).result

    def subf(self, a, b):
        return arith_dialect.SubFOp(a, b).result

    def addi(self, a, b):
        return arith_dialect.AddIOp(a, b).result

    def muli(self, a, b):
        return arith_dialect.MulIOp(a, b).result

    def add(self, a, b):
        return self.addf(a, b)

    def mul(self, a, b):
        return self.mulf(a, b)

    # --- Constants ---

    def const_f32(self, value):
        return arith_dialect.ConstantOp(F32Type.get(), float(value)).result

    def const_index(self, value):
        return _index_const(int(value))

    # --- Math ops ---

    def exp(self, val):
        from lego.mlir.ir import Operation
        return Operation.create("math.exp", results=[val.type], operands=[val]).result

    def sqrt(self, val):
        from lego.mlir.ir import Operation
        return Operation.create("math.sqrt", results=[val.type], operands=[val]).result

    def rsqrt(self, val):
        from lego.mlir.ir import Operation
        return Operation.create("math.rsqrt", results=[val.type], operands=[val]).result

    # --- For-range loop with optional accumulator ---

    def for_range(self, n, body_fn, init_vals=None):
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


# ============================================================================
# CPUKernelBuilder
# ============================================================================

class CPUKernelBuilder:
    """Build a CPU kernel with multiple layout-aware buffers.

    Mirrors gpu_builder.KernelBuilder but emits ``func.func`` + ``scf.for``
    loops (no ``gpu.launch``), and compiles through
    ``lego-to-x86-vector`` / ``lego-to-arm-neon``.

    Example — SAXPY::

        from lego.core import Row
        from lego.backend.cpu_builder import CPUKernelBuilder, LayoutBuffer
        from lego.backend.compiler import DType

        N = 1024
        x_buf = LayoutBuffer(Row(N), shape=(N,))
        y_buf = LayoutBuffer(Row(N), shape=(N,))
        # (scalar args are NOT LayoutBuffers — they are passed as a separate list)

        def saxpy_body(ctx):
            # tile_id: which outer tile we're computing
            ...

        builder = CPUKernelBuilder(
            buffers=[x_buf, y_buf],
            kernel_body=saxpy_body,
            name="saxpy",
        )

    The kernel function signature emitted is::

        func.func @name(%a: f32, %X: memref<?xf32>, %Y: memref<?xf32>)

    where scalar params are listed first (in declaration order), then the
    flat memref for each LayoutBuffer.

    ``compile(target='cpu')`` returns a callable ``jit_fn(scalar_args..., arrays...)``.
    """

    def __init__(
        self,
        buffers: Sequence[LayoutBuffer],
        kernel_body: Callable[["CPUKernelContext"], None],
        name: str = "kernel",
        scalar_params: Optional[Sequence[str]] = None,
    ):
        """
        Parameters
        ----------
        buffers:
            List of LayoutBuffer descriptors for memref parameters.
        kernel_body:
            Callable(ctx: CPUKernelContext) that emits the kernel body MLIR.
        name:
            MLIR function name.
        scalar_params:
            Optional list of scalar parameter types.  Each entry is an MLIR
            type string such as ``"f32"`` or ``"i64"``.  Scalars precede all
            memref arguments in the function signature.
        """
        self._buffers = list(buffers)
        self._kernel_body = kernel_body
        self._name = name
        self._scalar_params = list(scalar_params or [])
        # JIT state — keyed by (target, cpu) so the same builder can be
        # compiled to multiple targets (e.g. 'scalar' and 'x86' for the
        # apples-to-apples isolation harness).
        self._engines: Dict[tuple, object] = {}

    def build_module(self):
        """Build the MLIR module (func.func with LEGO dialect ops)."""
        ctx = Context()
        register_lego(ctx)
        ctx.load_all_available_dialects()

        with ctx, Location.unknown():
            module = Module.create()

            # Build argument type list: scalars first, then memrefs.
            scalar_types = []
            for dtype_str in self._scalar_params:
                scalar_types.append(_get_mlir_element_type(ctx, dtype_str))

            memref_types = []
            for buf in self._buffers:
                elem_ty = _get_mlir_element_type(ctx, buf.dtype)
                memref_types.append(MemRefType.get([buf.numel], elem_ty))

            all_arg_types = scalar_types + memref_types

            with InsertionPoint(module.body):
                self._build_func(ctx, all_arg_types, len(scalar_types))

        return ctx, module

    def _build_func(self, mlir_ctx, all_arg_types, n_scalars):
        func_ty = FunctionType.get(all_arg_types, [])
        f = func_dialect.FuncOp(self._name, func_ty)
        f.sym_visibility = StringAttr.get("public")
        # The c-interface attribute is required so ExecutionEngine can find
        # the function by name with the standard memref ABI.
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        entry = f.add_entry_block()
        all_args = list(entry.arguments)

        scalar_vals = all_args[:n_scalars]
        buf_vals = all_args[n_scalars:]

        with InsertionPoint(entry):
            ctx = CPUKernelContext(buf_vals, self._buffers)
            # Expose scalar args on the context for the kernel body
            ctx._scalar_vals = scalar_vals
            self._kernel_body(ctx)
            func_dialect.ReturnOp([])

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, target: str = "cpu", cpu: Optional[str] = None) -> Callable:
        """Compile to the given CPU target and return a JIT-callable.

        The same builder instance can be compiled to multiple targets
        (e.g. ``'scalar'`` and ``'x86'``) — each produces an independent
        JIT-compiled function, enabling apples-to-apples isolation of the
        vectorization benefit vs JIT startup overhead.

        Args:
            target: ``"cpu"`` / ``"x86"`` / ``"arm-neon"`` / ``"scalar"`` —
                    must be registered in ``_CPU_TARGETS``.
                    ``"scalar"`` compiles via ``lego-to-llvm`` (no vector ops)
                    and serves as the JIT baseline for isolation harnesses.
            cpu:    Optional CPU override (e.g. ``"znver3"``, ``"skl"``).
                    Ignored for ``target="scalar"`` (lego-to-llvm has no cpu
                    option).

        Returns:
            A Python callable with the same signature as the kernel
            (scalars first, then numpy arrays for each LayoutBuffer).
        """
        cache_key = (target, cpu)
        if cache_key not in self._engines:
            cpu_target = _CPU_TARGETS.get(target)
            if cpu_target is None:
                raise ValueError(
                    f"Unknown CPU target '{target}'. "
                    f"Available: {list(_CPU_TARGETS)}"
                )
            ctx, module = self.build_module()
            pipeline_str = cpu_target.pipeline_string(cpu=cpu)
            with ctx:
                try:
                    pm = PassManager.parse(pipeline_str)
                    pm.run(module.operation)
                except Exception as e:
                    raise RuntimeError(
                        f"CPU compilation failed ({cpu_target.pipeline}):\n{e}"
                    ) from e
                engine = ExecutionEngine(module, opt_level=2)
            self._engines[cache_key] = engine

        return self._make_callable(self._engines[cache_key])

    def _make_callable(self, engine):
        """Build a Python wrapper that invokes the JIT-compiled function.

        Hot-path optimisations
        ----------------------
        1. **Descriptor cache** — ``get_ranked_memref_descriptor()`` costs
           ~23 µs per numpy array (Python + ctypes struct allocation).  For
           benchmarks that call the same kernel 1000× with the same arrays
           this adds ~0.1 ms overhead per call, masking vectorization wins.
           We cache the descriptor *and* the ``ctypes.pointer(pointer(desc))``
           object keyed by ``(data_ptr, shape, strides)`` so repeated calls
           with identical arrays pay only a dict lookup + ~1 µs.

        2. **Pre-built dtype → ctype map** — the ``_dtype_to_ctype`` dict is
           built once at wrapper-creation time rather than on every call.

        3. **np.ascontiguousarray guard** — skipped for arrays that are
           already C-contiguous (the common case); only invoked when needed,
           and its result is cached via the pointer key.
        """
        name = self._name
        scalar_params = self._scalar_params
        buffers = self._buffers

        # Build dtype→ctype map once (not per-call).
        _dtype_to_ctype = {
            "f32": ctypes.c_float,
            "f64": ctypes.c_double,
            "i32": ctypes.c_int32,
            "i64": ctypes.c_int64,
            "i16": ctypes.c_int16,
            "i8":  ctypes.c_int8,
        }

        def jit_fn(*args):
            """Invoke the JIT-compiled CPU kernel.

            Args:
                *args: scalar args (matching scalar_params types), then one
                       numpy array per LayoutBuffer (in declaration order).
            """
            n_scalars = len(scalar_params)
            scalar_args_py = args[:n_scalars]
            buf_args = args[n_scalars:]

            if len(buf_args) != len(buffers):
                raise ValueError(
                    f"Expected {len(buffers)} buffer(s), got {len(buf_args)}"
                )

            # Build ctypes args: scalars first, then ranked-memref pointers.
            cargs = []
            for dtype_str, py_val in zip(scalar_params, scalar_args_py):
                c_ty = _dtype_to_ctype.get(dtype_str, ctypes.c_float)
                cargs.append((c_ty * 1)(py_val))

            for arr in buf_args:
                if isinstance(arr, np.ndarray):
                    # Ensure C-contiguous layout before building descriptor.
                    # np.ascontiguousarray is cheap (~0.1 µs) when already
                    # contiguous; only copies on non-contiguous input.
                    arr_c = arr if arr.flags['C_CONTIGUOUS'] else np.ascontiguousarray(arr)
                    cargs.append(_cached_memref_ptr(arr_c))
                else:
                    # Non-numpy argument (e.g. raw ctypes pointer) — pass through.
                    cargs.append(arr)

            engine.invoke(name, *cargs)

        return jit_fn
