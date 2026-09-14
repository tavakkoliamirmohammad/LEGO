"""Bridge to the LEGO algebra package.

`load_core()` returns `lego.core` when the package (and its native extension) is
importable, otherwise loads `lego/core.py` by path: constructing layouts needs only
SymPy, while evaluating them through the algebra needs the MLIR round trip.
"""
import importlib.util
import pathlib

_CORE = None


def load_core():
    global _CORE
    if _CORE is not None:
        return _CORE
    try:
        import lego.core as core  # noqa: F401
        _CORE = core
    except Exception:
        path = pathlib.Path(__file__).resolve().parents[1] / "lego" / "core.py"
        spec = importlib.util.spec_from_file_location("lego_core_standalone", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CORE = mod
    return _CORE


def from_lego(layout, rows: int, cols: int):
    """Concrete evaluator view of a LEGO layout (needs the native extension)."""
    raise NotImplementedError("requires the MLIR build; see tests/test_synth_lego_adapter.py")
