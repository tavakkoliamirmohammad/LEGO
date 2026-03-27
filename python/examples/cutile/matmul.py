"""cuTile matrix multiplication with LEGO-computed block swizzling.

Demonstrates using LEGO's layout algebra to replace manual 2D swizzle
scheduling logic.  The ``L_pid.inv(bid)`` call computes (bidx, bidy)
tile coordinates from a 1D block ID, with L2-cache-friendly group
ordering.

Without LEGO (manual cuTile):
    bid = ct.bid(0)
    num_pid_n = (N + tn - 1) // tn
    group_id = bid // (GM * num_pid_n)
    first_pid_m = group_id * GM
    group_size_m = min(num_pid_m - first_pid_m, GM)
    bidx = first_pid_m + (bid % (group_size_m * num_pid_n)) // num_pid_n
    bidy = (bid % (group_size_m * num_pid_n)) % num_pid_n

With LEGO:
    bid = ct.bid(0)
    L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                    Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                        [num_pid_m, num_pid_n])
    bidx, bidy = L_pid.inv(bid)   # LEGO simplifies to divmod arithmetic

Usage (source verification only — no GPU required):
    python matmul.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sympy import Max, Min
from lego.core import OrderBy, Col
from lego.frontends.cutile_jit import get_cutile_kernel_source


# ---------------------------------------------------------------------------
# cuTile kernel with LEGO swizzle layout
# ---------------------------------------------------------------------------

def matmul_kernel(A, B, C, M, N, K, tm, tn, tk, GM):
    bid = 0  # placeholder for ct.bid(0)
    num_pid_m = M // tm
    num_pid_n = N // tn

    # LEGO layout for L2-cache-friendly block scheduling:
    # Groups GM consecutive m-blocks together, traversing n-blocks within
    # each group.  This is identical to Triton's grouped PID remapping.
    L_pid = OrderBy(Col(Max(num_pid_m // GM, 1), 1),
                    Col(Min(num_pid_m, GM), num_pid_n)).TileBy(
                        [num_pid_m, num_pid_n])
    bidx, bidy = L_pid.inv(bid)

    # In a real kernel:
    #   acc = 0.0
    #   for k in range(K // tk):
    #       a_tile = ct.load(A, index=(bidx, k), shape=(tm, tk))
    #       b_tile = ct.load(B, index=(k, bidy), shape=(tk, tn))
    #       acc = ct.mma(a_tile, b_tile, acc)
    #   ct.store(C, index=(bidx, bidy), tile=acc.astype(C.dtype))


# ---------------------------------------------------------------------------
# Show generated source
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src = get_cutile_kernel_source(matmul_kernel)
    print("=== LEGO-generated cuTile matmul (swizzled) ===")
    print(src)
    print()

    # Verify layout expressions are eliminated
    assert 'L_pid' not in src, "Layout variable not eliminated"
    assert 'OrderBy' not in src, "OrderBy not eliminated"
    assert 'Col(' not in src, "Col not eliminated"
    assert 'bidx =' in src, "bidx not computed"
    assert 'bidy =' in src, "bidy not computed"
    print("All assertions passed.")
