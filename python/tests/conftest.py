"""pytest conftest: stub out compiled MLIR bindings for pure-Python tests.

The ``lego`` package eagerly imports ``lego.mlir.*`` and
``lego.backend.dialects.lego_dialect`` which require compiled C-extensions.
Since the printer / codegen tests never call MLIR APIs, we intercept those
imports and hand back lightweight stub modules.
"""

import sys
import types
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec


class _StubLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__all__ = []
        module.__getattr__ = lambda name: type("_Stub", (), {
            "__getattr__": lambda s, n: s,
            "__call__": lambda s, *a, **kw: s,
            "__iter__": lambda s: iter([]),
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
            "__setattr__": lambda s, n, v: None,
            "__getitem__": lambda s, i: s,
        })()


class _MLIRFinder(MetaPathFinder):
    """Intercept ``lego.mlir`` and ``lego.mlir.*`` imports."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "lego.mlir" or fullname.startswith("lego.mlir."):
            return ModuleSpec(fullname, _StubLoader())
        return None


# Register the MLIR stub finder.
sys.meta_path.insert(0, _MLIRFinder())

# Stub ``lego.backend.dialects.lego_dialect`` so that symbolic.py can
# import op classes (ApplyOp, etc.) without the auto-generated bindings.
_fake = types.ModuleType("lego.backend.dialects.lego_dialect")
_fake.__getattr__ = lambda name: type(name, (), {})
sys.modules["lego.backend.dialects.lego_dialect"] = _fake
