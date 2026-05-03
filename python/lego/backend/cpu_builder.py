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
    F32Type, F64Type, IntegerType,
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
                # Use opt_level=0 for the 'scalar' target so that LLVM's own
                # auto-vectorizer does NOT kick in on the simple scalar loop IR.
                # If we compile scalar JIT at opt_level=2, LLVM's LoopVectorize
                # pass vectorizes the scalar loop too — making scalar_jit ≈
                # vec_jit in speed and collapsing the isolation signal to ~1×.
                # opt_level=0 gives us a clean, un-optimized scalar baseline;
                # the speedup scalar_jit/vec_jit then measures our vectorizer's
                # contribution versus LLVM's un-aided scalar code generation.
                #
                # For all other targets (x86, arm-neon, cpu): use opt_level=3
                # (not 2) so that LLVM aggressively CSEs the struct-field
                # extractvalue chains that the memref-to-LLVM lowering emits.
                # With opt_level=2, repeated `llvm.extractvalue %struct[1]` ops
                # in the inner loop body survive as separate instructions because
                # LLVM's O2 GVN/CSE stops short of hoisting them. opt_level=3
                # runs the full optimizer (including AggressiveInstCombine and
                # the full LoopOptimizer suite) which eliminates these redundant
                # struct-field extractions and hoists the base-pointer into a
                # register before the loop — matching what gcc -O3 does naturally.
                jit_opt = 0 if target == "scalar" else 3
                engine = ExecutionEngine(module, opt_level=jit_opt)
            self._engines[cache_key] = engine

        return self._make_callable(self._engines[cache_key])

    def compile_aot(self, output_path: str = "kernel.o",
                   target: str = "cpu", cpu: Optional[str] = None) -> str:
        """Compile to the given CPU target and emit a relocatable .o file.

        Uses the same MLIR pass pipeline as ``compile()``, then writes the
        compiled kernel to an object file via
        ``ExecutionEngine.dump_to_object_file()``.  The resulting ``.o`` can be
        linked into a static or shared binary without any JIT runtime overhead.

        Args:
            output_path: Destination for the ``.o`` file.  Parent directory
                         must exist.  Default: ``"kernel.o"``.
            target:      CPU backend — same values as ``compile()``.
            cpu:         Optional CPU string (e.g. ``"skx"``, ``"znver3"``).

        Returns:
            The resolved absolute path to the written object file.

        Example::

            obj_path = builder.compile_aot(output_path='/tmp/saxpy.o')
            # Link: gcc -O2 harness.c /tmp/saxpy.o -o harness
        """
        import os
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
                    f"AOT compilation failed ({cpu_target.pipeline}):\n{e}"
                ) from e
            # opt_level=3: same as compile() for non-scalar targets.
            jit_opt = 0 if target == "scalar" else 3
            engine = ExecutionEngine(module, opt_level=jit_opt)
            engine.dump_to_object_file(output_path)

        return os.path.abspath(output_path)

    def bench(self, *args, n_iters: int = 1000, target: str = "cpu",
              cpu: Optional[str] = None) -> float:
        """Run the kernel *n_iters* times in a **single** JIT-level call.

        The usual ``compile()`` path pays ~4-5µs of Python/ctypes dispatch per
        call.  For small-N kernels (N ≤ 16 K) that cost dominates the
        measurement.  ``bench()`` works around this by wrapping the kernel in a
        ``scf.for`` loop at the MLIR level so that ONE ``engine.invoke()`` call
        performs all *n_iters* iterations.  The total wall-clock time is then
        divided by *n_iters* to get the per-iteration kernel time.

        Parameters
        ----------
        *args:
            Same arguments as the compiled kernel (scalars first, then numpy
            arrays for each LayoutBuffer).
        n_iters:
            Number of kernel iterations to perform inside a single invoke.
            Default 1000.
        target / cpu:
            Compilation target (forwarded to ``compile()``).

        Returns
        -------
        float
            Per-iteration time in **milliseconds**.
        """
        import time as _time
        bench_key = (target, cpu, n_iters, "bench")
        if bench_key not in self._engines:
            # Build a wrapper module: an outer scf.for [0, n_iters) that calls
            # the original kernel body on each iteration.  We do this by
            # building a *fresh* CPUKernelBuilder whose body emits the outer
            # loop and then delegates to the original kernel_body.
            orig_kernel_body = self._kernel_body
            orig_scalar_params = list(self._scalar_params)
            orig_buffers = list(self._buffers)
            n_iters_const = n_iters

            def bench_body(ctx):
                """Wrap original kernel body in scf.for n_iters loop."""
                n_iv = _index_const(n_iters_const)
                lb   = _index_const(0)
                step = _index_const(1)
                loop = scf_dialect.ForOp(lb, n_iv, step)
                with InsertionPoint(loop.body):
                    # Inside the loop body the context is the same ctx — scalar
                    # vals and buf_vals don't change across iterations.
                    orig_kernel_body(ctx)
                    scf_dialect.YieldOp([])

            bench_builder = CPUKernelBuilder(
                buffers=orig_buffers,
                kernel_body=bench_body,
                name=self._name + "_bench",
                scalar_params=orig_scalar_params,
            )
            # Compile and cache only the engine (not the callable wrapper).
            cpu_target = _CPU_TARGETS.get(target)
            if cpu_target is None:
                raise ValueError(
                    f"Unknown CPU target '{target}'. "
                    f"Available: {list(_CPU_TARGETS)}"
                )
            ctx_mlir, module = bench_builder.build_module()
            pipeline_str = cpu_target.pipeline_string(cpu=cpu)
            with ctx_mlir:
                try:
                    pm = PassManager.parse(pipeline_str)
                    pm.run(module.operation)
                except Exception as e:
                    raise RuntimeError(
                        f"bench() compilation failed ({cpu_target.pipeline}):\n{e}"
                    ) from e
                jit_opt = 0 if target == "scalar" else 3
                bench_engine = ExecutionEngine(module, opt_level=jit_opt)
            # Store both engine and callable.
            bench_fn = bench_builder._make_callable(bench_engine)
            self._engines[bench_key] = bench_fn

        bench_fn = self._engines[bench_key]
        t0 = _time.perf_counter_ns()
        bench_fn(*args)
        elapsed_ns = _time.perf_counter_ns() - t0
        return elapsed_ns / n_iters / 1e6   # ms per iteration

    def bench_self_timed(self, *args, n_iters: int = 1000, n_warmup: int = 100,
                         target: str = "cpu",
                         cpu: Optional[str] = None) -> float:
        """Timing harness whose measurement happens INSIDE the JIT'd MLIR module.

        Unlike ``bench()``, which wraps Python's ``time.perf_counter_ns()``
        around a single ``engine.invoke()`` call, ``bench_self_timed`` emits
        ``clock_gettime(CLOCK_MONOTONIC)`` calls as MLIR ops inside the JIT'd
        module:

        1. Warmup loop: *n_warmup* iterations of the kernel body (warms icache,
           branch predictor, TLB — eliminates first-call bias).
        2. Start timestamp: ``clock_gettime(CLOCK_MONOTONIC)`` → (sec, nsec).
        3. Timed loop: *n_iters* iterations of the kernel body.
        4. End timestamp: ``clock_gettime(CLOCK_MONOTONIC)`` → (sec, nsec).
        5. Elapsed nanoseconds = (end_sec - start_sec) * 1e9 + (end_nsec - start_nsec).
        6. Per-iteration time (ms) = elapsed_ns / n_iters / 1e6.
        7. Result written to an output memref<1xf64> that Python reads back.

        This eliminates ALL Python-level timing overhead — the number returned
        is the actual JIT'd kernel's per-iteration cost as measured by the
        hardware's monotonic clock from within the running JIT'd code.

        The function calls ``func.func private @clock_gettime(i32, memref<2xi64>)``
        which the MLIR ExecutionEngine resolves to the libc symbol at JIT time
        (libc is already loaded in the Python process).

        Parameters
        ----------
        *args:
            Same arguments as the compiled kernel (scalars first, then numpy
            arrays for each LayoutBuffer).
        n_iters:
            Number of timed iterations.  Default 1000.
        n_warmup:
            Number of warmup iterations (excluded from timing).  Default 100.
        target / cpu:
            Compilation target (forwarded to ``compile()``).

        Returns
        -------
        float
            Per-iteration time in **milliseconds** as measured by
            ``clock_gettime(CLOCK_MONOTONIC)`` inside the JIT'd code.
        """
        import numpy as _np

        bench_key = (target, cpu, n_iters, n_warmup, "bench_self_timed")
        if bench_key not in self._engines:
            orig_kernel_body = self._kernel_body
            orig_scalar_params = list(self._scalar_params)
            orig_buffers = list(self._buffers)
            n_warmup_const = n_warmup
            n_iters_const = n_iters

            def self_timed_body(ctx):
                """Emit warmup + clock_gettime-timed loop + write result."""
                # CLOCK_MONOTONIC = 1 on Linux.
                i32_ty = IntegerType.get(32)
                i64_ty = IntegerType.get(64)
                f64_ty = F64Type.get()
                ts_ty  = MemRefType.get([2], i64_ty)

                clk_id = arith_dialect.ConstantOp(i32_ty, 1).result

                # Allocate two timespecs (start and end).
                start_ts = memref_dialect.AllocaOp(ts_ty, [], []).result
                end_ts   = memref_dialect.AllocaOp(ts_ty, [], []).result

                # Warmup loop.
                nw = _index_const(n_warmup_const)
                c0 = _index_const(0)
                c1 = _index_const(1)
                warmup = scf_dialect.ForOp(c0, nw, c1)
                with InsertionPoint(warmup.body):
                    orig_kernel_body(ctx)
                    scf_dialect.YieldOp([])

                # clock_gettime(CLOCK_MONOTONIC, &start_ts).
                func_dialect.CallOp(
                    [i32_ty], "clock_gettime",
                    [clk_id, start_ts])

                # Timed loop.
                ni = _index_const(n_iters_const)
                timed = scf_dialect.ForOp(c0, ni, c1)
                with InsertionPoint(timed.body):
                    orig_kernel_body(ctx)
                    scf_dialect.YieldOp([])

                # clock_gettime(CLOCK_MONOTONIC, &end_ts).
                func_dialect.CallOp(
                    [i32_ty], "clock_gettime",
                    [clk_id, end_ts])

                # Compute elapsed nanoseconds:
                #   elapsed_sec  = end_ts[0] - start_ts[0]
                #   elapsed_nsec = end_ts[1] - start_ts[1]
                #   elapsed_ns   = elapsed_sec * 1_000_000_000 + elapsed_nsec
                idx0 = _index_const(0)
                idx1 = _index_const(1)
                start_sec  = memref_dialect.LoadOp(start_ts, [idx0]).result
                start_nsec = memref_dialect.LoadOp(start_ts, [idx1]).result
                end_sec    = memref_dialect.LoadOp(end_ts,   [idx0]).result
                end_nsec   = memref_dialect.LoadOp(end_ts,   [idx1]).result

                ns_per_sec = arith_dialect.ConstantOp(i64_ty, 1_000_000_000).result
                diff_sec   = arith_dialect.SubIOp(end_sec, start_sec).result
                diff_nsec  = arith_dialect.SubIOp(end_nsec, start_nsec).result
                sec_ns     = arith_dialect.MulIOp(diff_sec, ns_per_sec).result
                elapsed_ns = arith_dialect.AddIOp(sec_ns, diff_nsec).result

                # Convert to f64 and divide: per_iter_ms = elapsed_ns / n_iters / 1e6.
                elapsed_f64 = arith_dialect.SIToFPOp(f64_ty, elapsed_ns).result
                n_iters_f64 = arith_dialect.ConstantOp(
                    f64_ty, float(n_iters_const)).result
                ns_per_ms   = arith_dialect.ConstantOp(f64_ty, 1_000_000.0).result
                per_iter_ns = arith_dialect.DivFOp(elapsed_f64, n_iters_f64).result
                per_iter_ms = arith_dialect.DivFOp(per_iter_ns, ns_per_ms).result

                # Write result to the last buffer (a special output memref<1xf64>).
                # The output buffer is the last element in ctx._buf_vals.
                out_buf = ctx._buf_vals[-1]
                out_idx = _index_const(0)
                memref_dialect.StoreOp(per_iter_ms, out_buf, [out_idx])

            # The self-timed builder adds one extra buffer for the f64 result.
            # We use a LayoutBuffer with dtype="f64" and numel=1 for the output.
            result_buf = LayoutBuffer(
                layout=None, shape=(1,), dtype="f64", shared=False)
            # Override numel so CPUKernelBuilder uses memref<1xf64>.
            result_buf = type('_ResultBuf', (), {
                'numel': 1, 'dtype': 'f64', 'layout': None,
                'shape': (1,), 'shared': False,
            })()

            timed_buffers = orig_buffers + [result_buf]

            timed_builder = CPUKernelBuilder(
                buffers=timed_buffers,
                kernel_body=self_timed_body,
                name=self._name + "_self_timed",
                scalar_params=orig_scalar_params,
            )

            cpu_target = _CPU_TARGETS.get(target)
            if cpu_target is None:
                raise ValueError(
                    f"Unknown CPU target '{target}'. "
                    f"Available: {list(_CPU_TARGETS)}"
                )

            # Build the module and inject the clock_gettime extern declaration.
            # build_module() emits the main function; we then prepend the extern
            # @clock_gettime so that the func.call in self_timed_body resolves.
            ctx_mlir, module = timed_builder.build_module()
            pipeline_str = cpu_target.pipeline_string(cpu=cpu)

            with ctx_mlir, Location.unknown():
                # Inject clock_gettime extern at the BEGINNING of the module body
                # (before the main function) using InsertionPoint.at_block_begin.
                # Ordering: extern first, then user func — required by MLIR verifier.
                i32_ty = IntegerType.get(32)
                i64_ty = IntegerType.get(64)
                ts_ty  = MemRefType.get([2], i64_ty)
                clk_fn_ty = FunctionType.get([i32_ty, ts_ty], [i32_ty])
                with InsertionPoint.at_block_begin(module.body):
                    clk_fn = func_dialect.FuncOp("clock_gettime", clk_fn_ty)
                    clk_fn.sym_visibility = StringAttr.get("private")

                try:
                    pm = PassManager.parse(pipeline_str)
                    pm.run(module.operation)
                except Exception as e:
                    raise RuntimeError(
                        f"bench_self_timed compilation failed ({pipeline_str}):\n{e}"
                    ) from e
                jit_opt = 0 if target == "scalar" else 3
                timed_engine = ExecutionEngine(module, opt_level=jit_opt)

            # Build the callable for the self-timed function.
            timed_fn = timed_builder._make_callable(timed_engine)
            self._engines[bench_key] = timed_fn

        timed_fn = self._engines[bench_key]

        # Create the output result buffer.
        result_arr = _np.zeros(1, dtype=_np.float64)
        timed_fn(*args, result_arr)
        return float(result_arr[0])

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

        # Hot-path optimisations
        # ───────────────────────
        # 4. **Pre-cached function pointer + packed array** — engine.invoke
        #    does per-call: symbol lookup, CFUNCTYPE creation, packed-array
        #    allocation, and N ctypes.cast calls.  Total: ~6-8 µs overhead.
        #    Instead:
        #      a) Call engine.lookup(name) ONCE at compile time → _ciface_fn.
        #      b) For each (scalar, buffer) combination that appears in the
        #         benchmark loop, build and cache a (c_void_p * N) packed
        #         array with the addresses pre-computed.
        #      c) On every hot-path call: key check + _ciface_fn(_pre_packed).
        #    This reduces the per-call Python overhead from ~7 µs to ~1 µs.
        #
        # 5. **Single-slot last-call cache** — for the overwhelmingly common
        #    benchmark pattern (same scalars + same arrays × 1000 calls),
        #    one key equality check is faster than a dict lookup.
        n_scalars = len(scalar_params)
        n_buffers = len(buffers)
        n_cargs = n_scalars + n_buffers

        # Pre-build the ctypes callable once: avoids per-call symbol lookup
        # and CFUNCTYPE construction inside engine.invoke.
        try:
            _ciface_fn = engine.lookup(name)   # ctypes callable for hot path
        except RuntimeError:
            _ciface_fn = None   # fall back to engine.invoke if lookup fails

        _scalar_cache: dict = {}   # (dtype_str, py_val) → ctypes-array-1
        # Cache maps key → (cargs_list, pre_packed_c_void_p_array)
        # The cargs list keeps ctypes objects alive (prevents GC of descriptors).
        # The pre_packed array is passed directly to _ciface_fn — no per-call cast.
        _call_cache: dict = {}     # key → (cargs_list, packed_array | None)
        _last: list = [None, None, None]  # [last_key, last_cargs, last_packed]

        def _build_packed(cargs):
            """Build a pre-cast (c_void_p * N) array from *cargs*."""
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
            # Validate arg count once.
            if len(args) != n_cargs:
                raise ValueError(
                    f"Expected {n_cargs} arg(s) "
                    f"({n_scalars} scalar(s) + {n_buffers} buffer(s)), "
                    f"got {len(args)}"
                )

            # ── Single-slot last-call cache ──────────────────────────────────
            # Build a cheap key using id() for O(1) object identity on arrays.
            # On a hit: call _ciface_fn with the pre-built packed array directly
            # — zero Python allocation, zero dict lookup, zero per-buffer work.
            if n_scalars == 1 and n_buffers == 2:
                key = (args[0], id(args[1]), id(args[2]))
            elif n_scalars == 0 and n_buffers == 2:
                key = (id(args[0]), id(args[1]))
            else:
                key = tuple(args[:n_scalars]) + tuple(id(a) for a in args[n_scalars:])

            if key == _last[0]:
                if _ciface_fn is not None:
                    _ciface_fn(_last[2])   # pre-packed, zero-overhead dispatch
                else:
                    engine.invoke(name, *_last[1])
                return

            # ── Full _call_cache (multi-array sweeps, new combos) ────────────
            cached = _call_cache.get(key)
            if cached is not None:
                cargs, packed = cached
            else:
                # Build cargs from scratch.
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

                # Pre-cast addresses into the packed array once.
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
