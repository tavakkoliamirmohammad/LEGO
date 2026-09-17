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


class ConcreteLegoLayout:
    """Evaluator view of a LEGO layout: `flat(i, j)` over logical coordinates."""

    def __init__(self, fn, rows: int, cols: int, expr):
        self._fn, self.rows, self.cols, self.expr = fn, rows, cols, expr

    def flat(self, i, j):
        import numpy as np
        i = np.asarray(i, dtype=np.int64); j = np.asarray(j, dtype=np.int64)
        return np.asarray(self._fn(i, j), dtype=np.int64)

    @property
    def footprint(self) -> int:
        return self.rows * self.cols


def from_lego(layout, rows: int, cols: int) -> ConcreteLegoLayout:
    """Evaluate a 2-D LEGO layout (needs the native extension for the MLIR round trip).

    Multi-level tilings take one index per level per logical dimension, ordered
    level-major with the last level fastest; logical (i, j) are decomposed with
    successive divmod over the per-level radices.
    """
    import sympy as sp
    I, J = sp.symbols("I J", integer=True, nonnegative=True)
    levels = list(getattr(layout, "_tile_groups", None) or [tuple(layout._dims)])
    radices_i = [int(lv[0]) for lv in levels]
    radices_j = [int(lv[1]) for lv in levels]

    def decompose(sym, radices):
        parts, rest = [], sym
        for r in reversed(radices[1:]):
            parts.append(sp.Mod(rest, r)); rest = sp.floor(rest / r)
        parts.append(rest)
        return list(reversed(parts))

    ii, jj = decompose(I, radices_i), decompose(J, radices_j)
    idx = [x for pair in zip(ii, jj) for x in pair]
    expr = sp.sympify(layout[tuple(idx)])
    fn = sp.lambdify((I, J), expr, modules="numpy")
    return ConcreteLegoLayout(fn, rows, cols, expr)
