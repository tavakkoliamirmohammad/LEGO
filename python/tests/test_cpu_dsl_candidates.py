"""Compile-only smoke tests for the cpu_dsl_comparison candidate kernels.

Each ``evaluation/cpu_dsl_comparison/candidates/<name>/*.py`` file defines
a single ``@cpu_kernel``-decorated function (wrapped by ``@benchmark``).
These tests import each candidate as a module and call ``.compile(target='cpu')``
on the kernel — no execution, no numeric verification — so CI gets fast
coverage that the full lego-to-x86-vector pipeline still lowers each
candidate without crashing.

Run::

    PYTHONPATH=<build>/python_packages/lego \\
      python -m pytest python/tests/test_cpu_dsl_candidates.py -v
"""

import importlib.util
from pathlib import Path

import pytest

# Resolve the candidates directory relative to the repo root (3 levels up
# from this file: python/tests/ → python/ → repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES_DIR = _REPO_ROOT / "evaluation" / "cpu_dsl_comparison" / "candidates"


def _x86_pipeline_available() -> bool:
    """Check if lego-to-x86-vector is registered (matches test_cpu_dsl.py)."""
    try:
        from lego.mlir.ir import Context, Module, Location
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


def _discover_candidates():
    """Yield (candidate_name, .py_path) for every candidate dir."""
    if not _CANDIDATES_DIR.is_dir():
        return
    for cand_dir in sorted(_CANDIDATES_DIR.iterdir()):
        if not cand_dir.is_dir():
            continue
        for py in sorted(cand_dir.glob("*.py")):
            if py.name == "__init__.py":
                continue
            yield (cand_dir.name, py)
            break


def _import_candidate(name: str, py_path: Path):
    """Import the candidate .py as an isolated module."""
    spec = importlib.util.spec_from_file_location(f"_cand_{name}", py_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_compiled_kernel(mod):
    """Return the underlying CPUKernelBuilder from the candidate module.

    Each candidate's ``@benchmark`` + ``@cpu_kernel`` chain produces a
    BenchmarkedKernel whose ``.kernel`` attribute is the CPUKernelBuilder
    that exposes ``.compile(target=...)``.  Some older candidates may
    just have a bare CPUKernelBuilder.
    """
    for attr in vars(mod).values():
        cls_name = type(attr).__name__
        if cls_name == "BenchmarkedKernel":
            inner = getattr(attr, "kernel", None)
            if inner is not None and callable(getattr(inner, "compile", None)):
                return inner
        if cls_name == "CPUKernelBuilder":
            return attr
    raise AssertionError("no @cpu_kernel/@benchmark object found in module")


_CANDIDATES = list(_discover_candidates())


@_skip_no_x86_pipeline
@pytest.mark.skipif(not _CANDIDATES,
                    reason="cpu_dsl_comparison candidates dir is missing")
@pytest.mark.parametrize("name,py_path", _CANDIDATES,
                         ids=[c[0] for c in _CANDIDATES])
def test_candidate_compiles(name: str, py_path: Path):
    """Each candidate's @cpu_kernel function compiles via lego-to-x86-vector.

    This is a compile-only check — no execution, no numeric verification.
    A pass means the candidate lowers cleanly through buildLegoLowerPipeline
    + convert-lego-to-linalg + linalg::vectorize + the LLVM tail.
    """
    mod = _import_candidate(name, py_path)
    kernel = _find_compiled_kernel(mod)
    fn = kernel.compile(target="cpu")
    assert callable(fn)
