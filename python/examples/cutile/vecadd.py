"""cuTile vector addition with LEGO layout algebra.

Demonstrates using LEGO's OrderBy/TileBy to compute gather/scatter indices
for a 1D tile-parallel vector addition.  LEGO replaces the manual offset
arithmetic with a single layout expression.

Without LEGO (manual cuTile):
    bid = ct.bid(0)
    offsets = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    a_tile = ct.gather(a, offsets)
    ...

With LEGO:
    bid = ct.bid(0)
    L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
    offsets = L[bid, :]        # LEGO simplifies to: TILE * bid + ct.arange(TILE)
    a_tile = ct.gather(a, offsets)
    ...

Usage:
    python vecadd.py                  # source verification + GPU test
    LEGO_SAVE_KERNEL=1 python vecadd.py  # also saves generated kernel
"""

import math
import torch
import cuda.tile as ct

from lego.core import OrderBy, Row
from lego.frontends.cutile_jit import cutile_jit, get_cutile_kernel_source

ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Reference cuTile kernel (no LEGO)
# ---------------------------------------------------------------------------

@ct.kernel
def vecadd_cutile(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)
    indices = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    ct.scatter(c, indices, a_tile + b_tile)


# ---------------------------------------------------------------------------
# LEGO cuTile kernel — layout algebra replaces manual index computation
# ---------------------------------------------------------------------------

@cutile_jit
@ct.kernel
def vecadd_lego(a, b, c, N: int, TILE: ConstInt):
    bid = ct.bid(0)
    L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
    offsets = L[bid, :]
    a_tile = ct.gather(a, offsets)
    b_tile = ct.gather(b, offsets)
    ct.scatter(c, offsets, a_tile + b_tile)


# ---------------------------------------------------------------------------
# Verification & source inspection
# ---------------------------------------------------------------------------

def verify_source():
    """Check that LEGO eliminates layout expressions and produces ct.arange."""
    def vecadd_raw(a, b, c, N, TILE):
        bid = ct.bid(0)
        L = OrderBy(Row(N)).TileBy([N / TILE], [TILE])
        offsets = L[bid, :]
        a_tile = ct.gather(a, offsets)
        b_tile = ct.gather(b, offsets)
        ct.scatter(c, offsets, a_tile + b_tile)

    src = get_cutile_kernel_source(vecadd_raw)
    print("=== LEGO-generated cuTile vecadd ===")
    print(src)
    print()

    assert 'OrderBy' not in src, "Layout expression not eliminated"
    assert 'TileBy' not in src, "TileBy not eliminated"
    assert 'ct.arange' in src, "ct.arange not generated"
    assert 'TILE * bid' in src, "Offset expression not simplified"
    print("Source verification passed.\n")


def verify_gpu():
    """Run both kernels on GPU and compare against PyTorch."""
    N = 2 ** 16
    TILE = 1024
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    expected = a + b

    stream = torch.cuda.current_stream()
    grid = (math.ceil(N / TILE), 1, 1)

    # Reference cuTile kernel
    c_ref = torch.empty_like(a)
    ct.launch(stream, grid, vecadd_cutile, (a, b, c_ref, TILE))
    assert torch.allclose(c_ref, expected, atol=1e-5), "Reference cuTile vecadd mismatch!"
    print("Reference cuTile vecadd: PASS")

    # LEGO cuTile kernel
    c_lego = torch.empty_like(a)
    ct.launch(stream, grid, vecadd_lego, (a, b, c_lego, N, TILE))
    assert torch.allclose(c_lego, expected, atol=1e-5), "LEGO cuTile vecadd mismatch!"
    print("LEGO cuTile vecadd:      PASS")

    # Multiple sizes
    for size in [128, 1024, 8192, 2 ** 16]:
        a = torch.randn(size, device='cuda')
        b = torch.randn(size, device='cuda')
        c = torch.empty_like(a)
        g = (math.ceil(size / TILE), 1, 1)
        ct.launch(stream, g, vecadd_lego, (a, b, c, size, TILE))
        assert torch.allclose(c, a + b, atol=1e-5), f"Failed for N={size}"
    print("Multi-size verification:  PASS\n")


if __name__ == "__main__":
    verify_source()
    if torch.cuda.is_available():
        verify_gpu()
    else:
        print("No GPU available, skipping GPU verification.")
