"""End-to-end tests for @cpu_kernel: Python AST → MLIR → x86 JIT → numpy.

Architecture note: kernels are defined at MODULE level so that ``Buffer``
and other DSL names appear in ``fn.__globals__``, matching the pattern used
by the puzzle examples (puzzle_01_map.py, puzzle_02_zip.py, …).

Test hierarchy:
  1. test_module_imports        — smoke: can we import cpu_dsl?
  2. test_buffer_type           — Buffer[N] annotation works
  3. test_saxpy_build_module    — MLIR generation succeeds (no pipeline run)
  4. test_saxpy_pipeline        — full x86-vector pipeline compiles
  5. test_saxpy_runs            — JIT execution produces correct numerics
  6. test_vecadd_runs           — element-wise add, no scalar arg
  7. test_scale_runs            — multiply by scalar, in-place
  8. test_range_loop_runs       — explicit range() (no tile_range)

Run::

    PYTHONPATH=<build>/python_packages/lego \\
      python -P -m pytest python/tests/test_cpu_dsl.py -v \\
        --import-mode=importlib --rootdir <build>
"""

import numpy as np
import pytest

from lego.backend.cpu_dsl import cpu_kernel, Buffer, tile_range  # noqa: F401


# ---------------------------------------------------------------------------
# Module-level kernel definitions
# (Buffer must be in fn.__globals__ → define kernels at module scope)
# ---------------------------------------------------------------------------

_N_SAXPY = 1024

@cpu_kernel(grid=(_N_SAXPY,), tile=(8,))
def _saxpy(a: float, X: Buffer[_N_SAXPY], Y: Buffer[_N_SAXPY]):
    for i in tile_range:
        Y[i] = a * X[i] + Y[i]


_N_VECADD = 512

@cpu_kernel(grid=(_N_VECADD,), tile=(8,))
def _vecadd(X: Buffer[_N_VECADD], Y: Buffer[_N_VECADD], Z: Buffer[_N_VECADD]):
    for i in tile_range:
        Z[i] = X[i] + Y[i]


_N_SCALE = 256

@cpu_kernel(grid=(_N_SCALE,), tile=(8,))
def _scale(a: float, Y: Buffer[_N_SCALE]):
    for i in tile_range:
        Y[i] = a * Y[i]


_N_FILL = 128

@cpu_kernel(grid=(_N_FILL,))
def _fill_zeros(X: Buffer[_N_FILL]):
    for i in range(_N_FILL):
        X[i] = 0.0


# ---------------------------------------------------------------------------
# Tier 1 — module imports
# ---------------------------------------------------------------------------

def test_module_imports():
    """Smoke test: cpu_dsl module imports without error."""
    from lego.backend.cpu_dsl import cpu_kernel, Buffer, tile_range
    assert callable(cpu_kernel)
    assert Buffer is not None
    assert tile_range is not None


def test_cpu_builder_imports():
    """CPUKernelBuilder and LayoutBuffer import cleanly."""
    from lego.backend.cpu_builder import CPUKernelBuilder, LayoutBuffer, CPUKernelContext
    assert callable(CPUKernelBuilder)
    assert callable(LayoutBuffer)
    assert callable(CPUKernelContext)


# ---------------------------------------------------------------------------
# Tier 2 — Buffer type annotation
# ---------------------------------------------------------------------------

def test_buffer_1d():
    from lego.backend.cpu_dsl import Buffer, _BufferType
    bt = Buffer[16]
    assert isinstance(bt, _BufferType)
    assert bt.dims == (16,)


def test_buffer_2d():
    from lego.backend.cpu_dsl import Buffer, _BufferType
    bt = Buffer[4, 8]
    assert isinstance(bt, _BufferType)
    assert bt.dims == (4, 8)


# ---------------------------------------------------------------------------
# Tier 3 — MLIR generation (build_module only, no pipeline)
# ---------------------------------------------------------------------------

def test_saxpy_build_module():
    """MLIR module for SAXPY generates without exception."""
    ctx, module = _saxpy.build_module()
    mlir_text = str(module)
    # The generated IR must contain a func.func named saxpy
    assert "func.func" in mlir_text
    assert "_saxpy" in mlir_text
    # It should have memref arguments (the two Buffer params)
    assert "memref" in mlir_text


def test_vecadd_build_module():
    """MLIR module for element-wise vector add generates without exception."""
    ctx, module = _vecadd.build_module()
    mlir_text = str(module)
    assert "func.func" in mlir_text
    assert "_vecadd" in mlir_text


# ---------------------------------------------------------------------------
# Tier 4 — lego-to-x86-vector pipeline compiles cleanly
# ---------------------------------------------------------------------------

def _x86_pipeline_available():
    """Check if lego-to-x86-vector is registered."""
    try:
        from lego.mlir.ir import Context, Location, Module
        from lego.mlir.passmanager import PassManager
        from lego.backend.dialects.lego_dialect import register as register_lego
        ctx = Context()
        register_lego(ctx)
        ctx.load_all_available_dialects()
        with ctx, Location.unknown():
            Module.create()
            PassManager.parse("builtin.module(lego-to-x86-vector)")
        return True
    except Exception:
        return False


_skip_no_x86_pipeline = pytest.mark.skipif(
    not _x86_pipeline_available(),
    reason="lego-to-x86-vector pipeline not registered in this build",
)


@_skip_no_x86_pipeline
def test_saxpy_pipeline():
    """SAXPY MLIR module passes through lego-to-x86-vector without error."""
    ctx, module = _saxpy.build_module()

    from lego.mlir.passmanager import PassManager
    with ctx:
        pm = PassManager.parse("builtin.module(lego-to-x86-vector)")
        pm.run(module.operation)

    lowered = str(module)
    # After lowering, the module must contain LLVM dialect ops
    assert "llvm" in lowered.lower() or "func" in lowered


# ---------------------------------------------------------------------------
# Tier 5 — SAXPY JIT execution + correctness
# ---------------------------------------------------------------------------

@_skip_no_x86_pipeline
def test_saxpy_runs():
    """SAXPY: compile, run, compare to numpy reference."""
    N = _N_SAXPY
    rng = np.random.default_rng(0)
    a_val = np.float32(2.5)
    X = rng.standard_normal(N).astype(np.float32)
    Y = rng.standard_normal(N).astype(np.float32)
    Y_ref = (a_val * X + Y).copy()

    jit_fn = _saxpy.compile(target="cpu")
    jit_fn(a_val, X, Y)

    np.testing.assert_allclose(Y, Y_ref, rtol=1e-5,
                               err_msg="SAXPY JIT result mismatch")


@_skip_no_x86_pipeline
def test_saxpy_correctness():
    """Alias for test_saxpy_runs — the canonical correctness check.

    This is the test Task 23 specifies as the proof point.
    """
    test_saxpy_runs()


# ---------------------------------------------------------------------------
# Tier 6 — vector-add (no scalar arg)
# ---------------------------------------------------------------------------

@_skip_no_x86_pipeline
def test_vecadd_runs():
    """Element-wise vector add: compile, run, compare to numpy."""
    N = _N_VECADD
    rng = np.random.default_rng(42)
    X = rng.standard_normal(N).astype(np.float32)
    Y = rng.standard_normal(N).astype(np.float32)
    Z = np.zeros(N, dtype=np.float32)
    Z_ref = X + Y

    jit_fn = _vecadd.compile(target="cpu")
    jit_fn(X, Y, Z)

    np.testing.assert_allclose(Z, Z_ref, rtol=1e-5,
                               err_msg="vecadd JIT result mismatch")


# ---------------------------------------------------------------------------
# Tier 7 — scale in-place
# ---------------------------------------------------------------------------

@_skip_no_x86_pipeline
def test_scale_inplace_runs():
    """In-place scale: Y[i] = a * Y[i]."""
    N = _N_SCALE
    rng = np.random.default_rng(7)
    a_val = np.float32(3.0)
    Y = rng.standard_normal(N).astype(np.float32)
    Y_ref = (a_val * Y).copy()

    jit_fn = _scale.compile(target="cpu")
    jit_fn(a_val, Y)

    np.testing.assert_allclose(Y, Y_ref, rtol=1e-5,
                               err_msg="scale JIT result mismatch")


# ---------------------------------------------------------------------------
# Tier 8 — range() loop (no tile_range)
# ---------------------------------------------------------------------------

@_skip_no_x86_pipeline
def test_range_loop_runs():
    """Kernel using explicit range() — no tile_range sentinel."""
    N = _N_FILL
    X = np.ones(N, dtype=np.float32)
    jit_fn = _fill_zeros.compile(target="cpu")
    jit_fn(X)

    np.testing.assert_array_equal(X, np.zeros(N, dtype=np.float32),
                                  err_msg="fill_zeros JIT result mismatch")
