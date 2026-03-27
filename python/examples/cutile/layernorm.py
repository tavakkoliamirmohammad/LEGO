"""cuTile layer normalization with LEGO-computed column offsets.

Demonstrates using LEGO layout algebra for the inner-loop tile offset
computation in a multi-pass normalization kernel.  Each block processes
one row; the inner loop iterates over column tiles.

Without LEGO:
    col_offsets = j * TILE_D + ct.arange(TILE_D, dtype=ct.int32)

With LEGO:
    L_cols = OrderBy(Row(D)).TileBy([D // TILE_D], [TILE_D])
    col_offsets = L_cols[j, :]   # LEGO simplifies to: TILE_D * j + ct.arange(TILE_D)

Usage (source verification only — no GPU required):
    python layernorm.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lego.core import OrderBy, Row
from lego.frontends.cutile_jit import get_cutile_kernel_source


# ---------------------------------------------------------------------------
# cuTile kernel with LEGO column offset layout
# ---------------------------------------------------------------------------

def layernorm_kernel(X, Y, W, B, N, D, eps, TILE_D):
    bid = 0     # placeholder for ct.bid(0) — one block per row
    j = 0       # placeholder for loop variable

    # LEGO layout for column-tile offsets
    L_cols = OrderBy(Row(D)).TileBy([D / TILE_D], [TILE_D])
    col_offsets = L_cols[j, :]

    # In a real kernel, the inner loop would be:
    #   mean = 0.0
    #   for j in range(D // TILE_D):
    #       col_offsets = L_cols[j, :]   # simplified by LEGO
    #       x_tile = ct.gather(X, bid * D + col_offsets)
    #       mean += ct.sum(x_tile)
    #   mean /= D
    #   ... (variance, normalize, apply affine) ...


# ---------------------------------------------------------------------------
# Show generated source
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = get_cutile_kernel_source(layernorm_kernel)
    print("=== LEGO-generated cuTile layernorm ===")
    print(src)
    print()

    assert 'OrderBy' not in src, "Layout expression not eliminated"
    assert 'ct.arange' in src, "ct.arange not generated"
    assert 'TILE_D * j' in src, "Column offset not simplified"
    print("All assertions passed.")
