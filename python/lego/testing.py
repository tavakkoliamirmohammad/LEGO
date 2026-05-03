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
    TILE = 16

    def saxpy_ref(a, X, Y):
        Y[:] = a * X + Y

    @benchmark(reference=saxpy_ref, n_iters=1000, warmup=100)
    @cpu_kernel(grid=(N,), tile=(TILE,))
    def saxpy(a: float, X: Buffer[N], Y: Buffer[N]):
        for i in tile_range:
            Y[i] = a * X[i] + Y[i]

    if __name__ == "__main__":
        import numpy as np
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
    ):
        self.kernel = kernel
        self.reference = reference
        self.n_iters = n_iters
        self.warmup = warmup
        self.name = name or getattr(kernel, "_name", "kernel")
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
        """
        import numpy as np
        import math

        def _safe_ratio(a, b):
            if (isinstance(a, float) and not math.isnan(a)
                    and isinstance(b, float) and not math.isnan(b) and b > 0):
                return round(a / b, 4)
            return float("nan")

        notes = ""

        # NumPy baseline
        t_numpy = float("nan")
        try:
            # Warm + timed
            self.reference(*args)
            t0 = _time.perf_counter_ns()
            for _ in range(self.n_iters):
                self.reference(*args)
            t1 = _time.perf_counter_ns()
            t_numpy = (t1 - t0) / self.n_iters / 1e6
        except Exception as e:
            notes = f"numpy: {e}"

        # Scalar JIT baseline
        t_scalar = float("nan")
        try:
            t_scalar = self.kernel.bench_self_timed(
                *args, n_iters=self.n_iters, n_warmup=self.warmup, target="scalar"
            )
        except Exception as e:
            if not notes:
                notes = f"scalar_jit: {e}"

        # Vectorized JIT
        t_vec = float("nan")
        try:
            t_vec = self.kernel.bench_self_timed(
                *args, n_iters=self.n_iters, n_warmup=self.warmup, target=target
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

        return {
            "name":                  self.name,
            "numpy_ms":              _round_or_nan(t_numpy),
            "scalar_jit_ms":         _round_or_nan(t_scalar),
            "vec_jit_ms":            _round_or_nan(t_vec),
            "speedup_isolated_jit":  sp_iso,
            "speedup_vs_numpy":      sp_np,
            "verdict":               verdict,
            "notes":                 notes,
        }

    def verify(self, *args, rtol: float = 1e-4, target: str = "x86") -> bool:
        """Check that the JIT kernel matches the NumPy reference.

        Returns ``True`` on success, prints an error message and returns
        ``False`` on any mismatch or exception.
        """
        import numpy as np
        import copy

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
                    np.testing.assert_allclose(check_a, ref_a, rtol=rtol)
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
):
    """Decorator that wraps a ``@cpu_kernel`` with timing and verification.

    Apply **after** ``@cpu_kernel``::

        @benchmark(reference=my_ref, n_iters=1000, warmup=100)
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
    """
    def decorator(kernel):
        name = getattr(kernel, "_name", getattr(kernel, "__name__", "kernel"))
        return BenchmarkedKernel(
            kernel=kernel,
            reference=reference,
            n_iters=n_iters,
            warmup=warmup,
            name=name,
        )
    return decorator


def do_bench(fn: Callable, *args, n_iters: int = 1000, warmup: int = 100) -> float:
    """Standalone timing function (Triton-style).

    Calls *fn* with *args* ``warmup`` times, then ``n_iters`` times and
    returns the average wall-clock time per call in **milliseconds**.

    This measures Python-level overhead.  For pure kernel timing without
    Python dispatch overhead, use ``CPUKernelBuilder.bench_self_timed``.

    Args:
        fn:      Callable to time.
        *args:   Arguments to pass on each call.
        n_iters: Number of timed iterations (default 1000).
        warmup:  Number of warmup iterations (default 100).

    Returns:
        Average time per call in milliseconds.
    """
    for _ in range(warmup):
        fn(*args)
    t0 = _time.perf_counter_ns()
    for _ in range(n_iters):
        fn(*args)
    elapsed_ns = _time.perf_counter_ns() - t0
    return elapsed_ns / n_iters / 1e6
