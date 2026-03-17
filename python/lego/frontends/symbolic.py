"""
Symbolic LEGO — SymPy-based layout algebra frontend.

Re-exports the core layout building blocks for symbolic index computation.
All layout operations produce SymPy expressions that can be inspected,
printed, or lowered to MLIR via the backend.

Example:
    import sympy as sp
    from lego.frontends.symbolic import OrderBy, Row, GroupBy

    M, N = sp.symbols('M N', integer=True, positive=True)
    i, j = sp.symbols('i j', integer=True, positive=True)
    L = OrderBy(Row(M, N)).GroupBy([(M, N)])
    print(L[i, j])
"""

import sympy as sp
from lego.core import (
    LayoutBlock,
    GenP,
    RegP,
    Row,
    Col,
    OrderBy,
    GroupBy,
    TileByLayout,
    TritonRange,
    get_arange,
    product,
    antidiag,
    antidiag_inv,
    new_antidiag,
)
from lego.backend.mlir_roundtrip import simplify_via_mlir


# ---------------------------------------------------------------------------
# Symbolic utility helpers (used by benchmarks and tests, not by MLIR core)
# ---------------------------------------------------------------------------

def divisibility_constraint(lhs, rhs):
    return sp.Eq(lhs % rhs, 0, evaluate=False)


def le_constraint(lhs, rhs):
    return sp.StrictLessThan(lhs, rhs, evaluate=False)


