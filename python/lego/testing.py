"""Benchmarking and verification utilities for cpu_dsl kernels.

Provides a Triton-style ``@benchmark`` decorator and ``do_bench`` function
so each candidate kernel can be written as a single self-contained file that:

- Defines the NumPy reference and the ``@cpu_kernel``-decorated kernel.
- Runs timing when executed as a script (``python saxpy.py``).
- Exposes ``.measure()`` and ``.verify()`` for programmatic use (run_all.py).

Example
-------
::

    from lego.testing import benchmark
    from lego.backend.cpu_dsl import cpu_kernel, Buffer
    import numpy as np

    N = 1 << 20

    def saxpy_ref(a, X, Y):
        Y[:] = a * X + Y

    @benchmark(reference=saxpy_ref, n_iters=1000, warmup=100)
    @cpu_kernel
    def saxpy(a: float, X: Buffer[N], Y: Buffer[N]):
        for i in range(N):
            Y[i] = a * X[i] + Y[i]

    if __name__ == "__main__":
        rng = np.random.default_rng(0)
        a = np.float32(2.5)
        X = rng.standard_normal(N).astype(np.float32)
        Y = rng.standard_normal(N).astype(np.float32)
        print(saxpy.measure(a, X, Y))
        print("verified:", saxpy.verify(a, X, Y))
"""

import json
import time as _time
from typing import Callable, Optional


__all__ = ["benchmark", "BenchmarkedKernel", "do_bench"]


def _detect_alignment() -> int:
    """Detect the target alignment for vectorised loads/stores.

    The right value is ``max(cache_line_size, max_vector_register_bytes)``:

    - **Cache line** dominates spatial locality on every load. A vector load
      that crosses a cache line boundary fetches two lines, doubling traffic.
    - **Vector register** width is the largest aligned load the ISA emits.
      x86 AVX-512 = 64 B, AVX-2 = 32 B, ARM NEON = 16 B, ARM SVE = variable.
      We floor at 64 (covers every current LEGO target).

    Empirically observed values:

    =============================  =========  ==========  =============
    Target                         Vector     Cache line  Alignment
    =============================  =========  ==========  =============
    x86_64 AVX-512                 64         64          64
    x86_64 AVX-2 only              32         64          64
    ARM NEON (Linux server)        16         64          64
    ARM NEON (Apple M-series)      16         128         128
    ARM SVE (256-bit, Graviton 3)  32         64          64
    ARM SVE (Fugaku, 512-bit)      64         256         256
    IBM POWER9 / POWER10           16         128         128
    =============================  =========  ==========  =============

    Detection sources, in order:

    1. **Linux**: ``/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size``
       (most reliable; reflects what the kernel sees)
    2. **macOS**: ``sysctl -n hw.cachelinesize`` (returns 128 on M1/M2/M3,
       64 on older Intel Macs)
    3. **POSIX fallback**: ``os.sysconf("SC_LEVEL1_DCACHE_LINESIZE")``
       (works on Linux, glibc; macOS/BSD may not implement it)
    4. **Default**: 64 if everything above fails (Windows, exotic platforms)

    Returned value is rounded up to a power of two so it can be used as a
    bitmask in :func:`_to_aligned`'s ``(-base) & (align - 1)``.
    """
    import os
    import platform
    import subprocess

    line = 0

    # 1. Linux /sys
    sys_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    try:
        with open(sys_path) as f:
            line = int(f.read().strip())
    except (FileNotFoundError, ValueError, PermissionError, OSError):
        pass

    # 2. macOS / Darwin
    if line <= 0 and platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            line = int(out)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, OSError):
            pass

    # 3. POSIX sysconf
    if line <= 0:
        try:
            line = os.sysconf("SC_LEVEL1_DCACHE_LINESIZE")
        except (ValueError, OSError, AttributeError):
            line = 0

    # 4. Default
    if line <= 0:
        line = 64

    # Floor at 64 (covers AVX-512 register width even where cache line is 32).
    align = max(line, 64)
    # Round up to next power of two.
    p = 1
    while p < align:
        p <<= 1
    return p


_ALIGN = _detect_alignment()


def _to_aligned(arr, align: int = 0):
    """Return ``arr`` if its data pointer is already aligned, otherwise return
    an aligned copy.

    The default alignment is auto-detected by :func:`_detect_alignment` —
    typically 64 on x86_64 / Linux ARM, 128 on Apple ARM, larger on ARM SVE
    machines. Pass ``align`` explicitly to override.

    Why this exists
    ---------------
    Small numpy arrays (≲128 KiB) are allocated via ``sbrk``, which returns
    16-byte aligned addresses but rarely cache-line aligned. Vector loads on
    misaligned data cross cache lines, doubling memory traffic. Empirically
    this produces a ~2.4× bimodal slowdown on tiny (N≤64) GEMM kernels.
    Stable benchmarks need stable alignment.

    The returned array shares no storage with ``arr``; modifications won't be
    reflected back. This is fine for timing (output is discarded).
    """
    import numpy as np
    if not isinstance(arr, np.ndarray):
        return arr
    if align <= 0:
        align = _ALIGN
    if arr.ctypes.data % align == 0:
        return arr
    # Over-allocate, then take an aligned slice. The view chain holds a ref
    # to `raw` for as long as the result is alive.
    raw = np.empty(arr.nbytes + align, dtype=np.uint8)
    base = raw.ctypes.data
    offset = (-base) & (align - 1)
    result = raw[offset:offset + arr.nbytes].view(arr.dtype).reshape(arr.shape)
    result[:] = arr
    return result


class BenchmarkedKernel:
    """A ``@cpu_kernel``-decorated function augmented with timing and verification.

    Created by the :func:`benchmark` decorator.  Calling the object directly
    invokes the underlying JIT-compiled kernel (after compiling on first call).

    Attributes
    ----------
    kernel : CPUKernelBuilder
        The raw ``@cpu_kernel`` builder (for direct access to compile/bench).
    reference : Callable
        The NumPy reference function for correctness checking.
    n_iters : int
        Number of timed iterations for :meth:`measure`.
    warmup : int
        Number of warmup iterations (excluded from timing).
    name : str
        Kernel name (from the function's ``__name__``).
    """

    def __init__(
        self,
        kernel,
        reference: Callable,
        n_iters: int = 1000,
        warmup: int = 100,
        name: str = "",
        rtol: float = 1e-3,
        atol: float = 1e-5,
        meta: Optional[dict] = None,
    ):
        self.kernel = kernel
        self.reference = reference
        self.n_iters = n_iters
        self.warmup = warmup
        self.name = name or getattr(kernel, "_name", "kernel")
        self.rtol = rtol
        self.atol = atol
        self.meta = dict(meta) if meta else {}
        self._compiled = {}   # target → compiled callable

    def _get_compiled(self, target: str = "x86"):
        """Return (and cache) the compiled callable for *target*."""
        if target not in self._compiled:
            self._compiled[target] = self.kernel.compile(target=target)
        return self._compiled[target]

    def __call__(self, *args, target: str = "x86", **kwargs):
        """Call the compiled kernel directly."""
        return self._get_compiled(target)(*args, **kwargs)

    def measure(self, *args, target: str = "x86") -> dict:
        """Time the kernel using MLIR-internal clock_gettime.

        Returns a JSON-serializable dict with keys:
        ``name``, ``numpy_ms``, ``scalar_jit_ms``, ``vec_jit_ms``,
        ``speedup_isolated_jit``, ``speedup_vs_numpy``, ``verdict``,
        ``notes``.

        Numpy ndarray inputs are silently re-aligned to 64-byte boundaries
        before timing — see :func:`_to_aligned` for the rationale. This
        eliminates bimodal slowdowns on tiny kernels.
        """
        import numpy as np
        import math

        def _safe_ratio(a, b):
            if (isinstance(a, float) and not math.isnan(a)
                    and isinstance(b, float) and not math.isnan(b) and b > 0):
                return round(a / b, 4)
            return float("nan")

        notes = ""

        # Align ndarray inputs once; reuse aligned copies for all three
        # measurement paths (numpy / scalar JIT / vec JIT) so they're
        # comparing the same data placement.
        aligned = tuple(_to_aligned(a) for a in args)

        # NumPy baseline
        t_numpy = float("nan")
        try:
            # Warm + timed
            self.reference(*aligned)
            t0 = _time.perf_counter_ns()
            for _ in range(self.n_iters):
                self.reference(*aligned)
            t1 = _time.perf_counter_ns()
            t_numpy = (t1 - t0) / self.n_iters / 1e6
        except Exception as e:
            notes = f"numpy: {e}"

        # Scalar JIT baseline
        t_scalar = float("nan")
        try:
            t_scalar = self.kernel.bench_self_timed(
                *aligned, n_iters=self.n_iters, n_warmup=self.warmup, target="scalar"
            )
        except Exception as e:
            if not notes:
                notes = f"scalar_jit: {e}"

        # Vectorized JIT
        t_vec = float("nan")
        try:
            t_vec = self.kernel.bench_self_timed(
                *aligned, n_iters=self.n_iters, n_warmup=self.warmup, target=target
            )
        except Exception as e:
            notes = f"vec_jit: {e}"

        sp_iso = _safe_ratio(t_scalar, t_vec)
        sp_np  = _safe_ratio(t_numpy,  t_vec)

        if not math.isnan(t_vec) and not notes:
            if sp_iso > 1.05:
                verdict = "WIN"
            elif sp_iso > 0.95:
                verdict = "PARITY"
            else:
                verdict = "LOSS"
        else:
            verdict = "ERROR"

        def _round_or_nan(v):
            return round(v, 4) if (isinstance(v, float) and not math.isnan(v)) else v

        rec = {
            "name":                  self.name,
            "numpy_ms":              _round_or_nan(t_numpy),
            "scalar_jit_ms":         _round_or_nan(t_scalar),
            "vec_jit_ms":            _round_or_nan(t_vec),
            "speedup_isolated_jit":  sp_iso,
            "speedup_vs_numpy":      sp_np,
            "verdict":               verdict,
            "notes":                 notes,
        }
        # Merge metadata (prior_verdict, layout_class, N, etc.) for dashboards.
        for k, v in self.meta.items():
            rec.setdefault(k, v)
        return rec

    def verify(self, *args, rtol: Optional[float] = None,
               atol: Optional[float] = None, target: str = "x86") -> bool:
        """Check that the JIT kernel matches the NumPy reference.

        Compares with ``numpy.testing.assert_allclose(rtol, atol)`` —
        ``|jit - ref| <= atol + rtol * |ref|``. The ``atol`` floor matters
        for kernels whose output contains near-zero values (e.g. SAXPY on
        random ``standard_normal`` data, GEMM with results near zero), where
        a tight ``rtol`` alone trips on FMA-vs-non-FMA precision drift.

        Returns ``True`` on success, prints an error message and returns
        ``False`` on any mismatch or exception.
        """
        import numpy as np

        if rtol is None:
            rtol = self.rtol
        if atol is None:
            atol = self.atol

        # Make mutable copies so both branches start from the same state.
        ref_args   = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
        check_args = [a.copy() if isinstance(a, np.ndarray) else a for a in args]

        try:
            self.reference(*ref_args)
        except Exception as e:
            print(f"[verify] reference failed: {e}")
            return False

        try:
            jit_fn = self._get_compiled(target)
            jit_fn(*check_args)
        except Exception as e:
            print(f"[verify] jit ({target}) failed: {e}")
            return False

        # Compare outputs (in-place-modified arrays).
        for ref_a, check_a in zip(ref_args, check_args):
            if isinstance(ref_a, np.ndarray):
                try:
                    np.testing.assert_allclose(check_a, ref_a, rtol=rtol, atol=atol)
                except AssertionError as e:
                    print(f"[verify] mismatch: {e}")
                    return False

        return True

    def print_summary(self, *args, target: str = "x86"):
        """Convenience: run measure + verify and print a one-liner."""
        rec = self.measure(*args, target=target)
        verified = self.verify(*args, target=target)
        print(json.dumps({**rec, "verified": verified}))


def benchmark(
    reference: Callable,
    n_iters: int = 1000,
    warmup: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    meta: Optional[dict] = None,
):
    """Decorator that wraps a ``@cpu_kernel`` with timing and verification.

    Apply **after** ``@cpu_kernel``::

        @benchmark(reference=my_ref, meta={"N": N, "layout_class": "Unit",
                                           "prior_verdict": "WIN"})
        @cpu_kernel(grid=(N,), tile=(16,))
        def my_kernel(X: Buffer[N], Y: Buffer[N]):
            ...

    The decorated object is a :class:`BenchmarkedKernel` with:
    - ``kernel.measure(*args)`` → timing dict (JSON-serializable).
    - ``kernel.verify(*args)``  → ``True``/``False``.
    - ``kernel(*args)``         → direct JIT invocation.

    Args:
        reference: NumPy (or any Python) reference callable.  Must have the
                   same signature as the kernel function.
        n_iters:   Number of timed iterations for ``measure()``.
        warmup:    Warmup iterations excluded from timing.
        rtol:      Tolerance for ``verify()`` (numpy.testing.assert_allclose).
        meta:      Dict of extra fields merged into the ``measure()`` output
                   dict (e.g. ``N``, ``layout_class``, ``prior_verdict``) so
                   downstream dashboards can read them without per-candidate
                   boilerplate.
    """
    def decorator(kernel):
        name = getattr(kernel, "_name", getattr(kernel, "__name__", "kernel"))
        return BenchmarkedKernel(
            kernel=kernel,
            reference=reference,
            n_iters=n_iters,
            warmup=warmup,
            name=name,
            rtol=rtol,
            atol=atol,
            meta=meta,
        )
    return decorator


def do_bench(fn: Callable, *args, n_iters: int = 1000, warmup: int = 100) -> float:
    """Standalone timing function (Triton-style).

    Calls *fn* with *args* ``warmup`` times, then ``n_iters`` times and
    returns the average wall-clock time per call in **milliseconds**.

    This measures Python-level overhead.  For pure kernel timing without
    Python dispatch overhead, use ``CPUKernelBuilder.bench_self_timed``.

    Numpy ndarray inputs are re-aligned to 64-byte boundaries before timing
    — see :func:`_to_aligned`.

    Args:
        fn:      Callable to time.
        *args:   Arguments to pass on each call.
        n_iters: Number of timed iterations (default 1000).
        warmup:  Number of warmup iterations (default 100).

    Returns:
        Average time per call in milliseconds.
    """
    aligned = tuple(_to_aligned(a) for a in args)
    for _ in range(warmup):
        fn(*aligned)
    t0 = _time.perf_counter_ns()
    for _ in range(n_iters):
        fn(*aligned)
    elapsed_ns = _time.perf_counter_ns() - t0
    return elapsed_ns / n_iters / 1e6
