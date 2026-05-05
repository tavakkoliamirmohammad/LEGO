"""Compile-only smoke tests for the cpu_dsl example kernels.

Each ``evaluation/cpu_dsl_examples/candidates/<name>/*.py`` file defines
a single ``@cpu_kernel``-decorated function (wrapped by ``@benchmark``).
For every candidate × every CPU target (``x86``, ``arm-neon``, ``arm-sve``)
this test runs the LEGO MLIR pass pipeline on the candidate and asserts
that lowering completes without error.

Compile-only: the pipeline runs to LLVM dialect but the result is *not*
JIT-compiled or executed.  ARM targets work on an x86 host because we
stop before ``ExecutionEngine``; the produced LLVM IR can be inspected
or handed to ``llc -mtriple=aarch64-…`` separately for actual ARM
execution.

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
_CANDIDATES_DIR = _REPO_ROOT / "evaluation" / "cpu_dsl_examples" / "candidates"

_TARGETS = ("x86", "arm-neon", "arm-sve")


def _pipeline_available(target: str) -> bool:
    """Check that the lego-to-<target> pipeline is registered and parses."""
    try:
        from lego.mlir.ir import Context, Module, Location
        from lego.mlir.passmanager import PassManager
        from lego.backend.dialects.lego_dialect import register as register_lego
        from lego.backend.cpu_builder import _CPU_TARGETS
        cpu_target = _CPU_TARGETS.get(target)
        if cpu_target is None:
            return False
        ctx = Context()
        register_lego(ctx)
        ctx.load_all_available_dialects()
        with ctx, Location.unknown():
            Module.create()
            PassManager.parse(cpu_target.pipeline_string(cpu_target.default_cpu))
        return True
    except Exception:
        return False


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


def _find_kernel_builder(mod):
    """Return the underlying CPUKernelBuilder from the candidate module.

    Each candidate's ``@benchmark`` + ``@cpu_kernel`` chain produces a
    BenchmarkedKernel whose ``.kernel`` attribute is the CPUKernelBuilder
    that exposes ``.build_module()``.  Some candidates may have a bare
    CPUKernelBuilder.
    """
    for attr in vars(mod).values():
        cls_name = type(attr).__name__
        if cls_name == "BenchmarkedKernel":
            inner = getattr(attr, "kernel", None)
            if inner is not None and callable(getattr(inner, "build_module", None)):
                return inner
        if cls_name == "CPUKernelBuilder":
            return attr
    raise AssertionError("no @cpu_kernel/@benchmark object found in module")


_CANDIDATES = list(_discover_candidates())


def _run_pipeline_only(builder, target: str) -> str:
    """Run the lego-to-<target> pipeline; return the lowered IR as text.

    Stops before ``ExecutionEngine`` so this works on a host whose CPU
    doesn't match the target (e.g. running the ARM pipelines on x86).
    """
    from lego.mlir.passmanager import PassManager
    from lego.backend.cpu_builder import _CPU_TARGETS
    cpu_target = _CPU_TARGETS[target]
    ctx, module = builder.build_module()
    with ctx:
        pm = PassManager.parse(cpu_target.pipeline_string(cpu_target.default_cpu))
        pm.run(module.operation)
        return str(module)


_param_ids = [f"{c[0]}-{t}" for c in _CANDIDATES for t in _TARGETS]
_params = [(c[0], c[1], t) for c in _CANDIDATES for t in _TARGETS]


@pytest.mark.skipif(not _CANDIDATES,
                    reason="cpu_dsl_examples candidates dir is missing")
@pytest.mark.parametrize("name,py_path,target", _params, ids=_param_ids)
def test_candidate_pipeline_lowers(name: str, py_path: Path, target: str):
    """Pipeline lowers each candidate to LLVM dialect on each target.

    Compile-only — does not JIT-execute.  A pass means
    buildLegoLowerPipeline + (optionally) convert-lego-to-linalg +
    linalg::vectorize + the LLVM tail produced a valid module without
    errors.  Skip the (candidate, target) pair if the pipeline isn't
    registered in this build.
    """
    if not _pipeline_available(target):
        pytest.skip(f"lego-to-{target} pipeline not registered in this build")

    mod = _import_candidate(name, py_path)
    builder = _find_kernel_builder(mod)
    lowered = _run_pipeline_only(builder, target)
    # The lowered IR must contain LLVM dialect ops — verify by checking
    # that some lego/scf/arith ops have been turned into llvm.* form.
    assert "llvm." in lowered, (
        f"{name} on {target}: pipeline ran but produced no llvm.* ops; "
        f"lowering may have stalled mid-pipeline")
