"""
LEGO MLIR dialect Python overlay.

Provides dialect registration and re-exports auto-generated op classes.

Usage:
    from mlir.ir import Context
    from lego.dialects import lego_dialect

    ctx = Context()
    lego_dialect.register(ctx)
"""

try:
    from ._lego_ops_gen import *  # noqa: F401,F403 — auto-generated op classes
except ImportError:
    pass  # Generated bindings not yet built

try:
    from mlir._mlir_libs._legoDialects import register_lego_dialect as _register
except ImportError:
    try:
        from _mlir_libs._legoDialects import register_lego_dialect as _register
    except ImportError:
        _register = None


def register(ctx):
    """Register and load the LEGO dialect into a MLIR Context."""
    if _register is None:
        raise RuntimeError(
            "LEGO MLIR Python bindings not built. "
            "Build with -DMLIR_ENABLE_BINDINGS_PYTHON=ON."
        )
    _register(ctx)
