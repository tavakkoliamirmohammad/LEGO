"""cuTile vector addition with LEGO layout algebra.

Demonstrates using LEGO's OrderBy/TileBy to compute gather/scatter indices
for a 1D tile-parallel vector addition.  LEGO replaces the manual offset
arithmetic with a single layout expression.

Without LEGO (manual cuTile):
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    a = ct.gather(A, offsets)
    ...

With LEGO:
    bid = ct.bid(0)
    L = OrderBy(Row(N)).TileBy([N // TILE], [TILE])
    offsets = L[bid, :]        # LEGO simplifies to: TILE * bid + ct.arange(TILE)
    a = ct.gather(A, offsets)
    ...

Usage (source verification only — no GPU required):
    python vecadd.py
"""

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lego.core import OrderBy, Row
from lego.frontends.cutile_jit import get_cutile_kernel_source


# ---------------------------------------------------------------------------
# cuTile kernel with LEGO layout algebra
# ---------------------------------------------------------------------------

def vecadd_kernel(A, B, C, N, TILE):
    bid = 0  # placeholder for ct.bid(0)
    L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
    offsets = L[bid, :]
    # In a real kernel:
    #   a = ct.gather(A, offsets)
    #   b = ct.gather(B, offsets)
    #   ct.scatter(C, offsets, a + b)


# ---------------------------------------------------------------------------
# Show generated source
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = get_cutile_kernel_source(vecadd_kernel)
    print("=== LEGO-generated cuTile vecadd ===")
    print(src)
    print()

    # Verify layout is eliminated and ct.arange is present
    assert 'OrderBy' not in src, "Layout expression not eliminated"
    assert 'TileBy' not in src, "TileBy not eliminated"
    assert 'ct.arange' in src, "ct.arange not generated"
    assert 'TILE * bid' in src, "Offset expression not simplified"
    print("All assertions passed.")
