"""cuTile layer normalization with LEGO-computed column offsets.

Demonstrates using LEGO layout algebra for the inner-loop tile offset
computation in a multi-pass normalization kernel.  Each block processes
one row; the inner loop iterates over column tiles.

Without LEGO:
    col_offsets = j * TILE_D + ct.arange(TILE_D, dtype=ct.int32)

With LEGO:
    L_cols = OrderBy(Row(D)).TileBy([D / TILE_D], [TILE_D])
    col_offsets = L_cols[j, :]   # LEGO simplifies to: TILE_D * j + ct.arange(TILE_D)

Usage:
    python layernorm.py               # source verification + GPU test
"""

import math
import torch
import cuda.tile as ct

from lego.core import OrderBy, Row
from lego.frontends.cutile_jit import cutile_jit, get_cutile_kernel_source

ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Reference cuTile layernorm (no LEGO)
# ---------------------------------------------------------------------------

@ct.kernel
def layernorm_cutile(X, Y, W, B, D: int, eps: float, TILE_D: ConstInt):
    bid = ct.bid(0)
    num_col_tiles = ct.cdiv(D, TILE_D)
    zero_pad = ct.PaddingMode.ZERO

    # Pass 1: compute mean
    mean_acc = ct.full((1, TILE_D), 0, dtype=ct.float32)
    for j in range(num_col_tiles):
        x_tile = ct.load(X, index=(bid, j), shape=(1, TILE_D),
                         padding_mode=zero_pad).astype(ct.float32)
        mean_acc = mean_acc + x_tile
    mean = ct.sum(mean_acc) / D

    # Pass 2: compute variance
    var_acc = ct.full((1, TILE_D), 0, dtype=ct.float32)
    for j in range(num_col_tiles):
        x_tile = ct.load(X, index=(bid, j), shape=(1, TILE_D),
                         padding_mode=zero_pad).astype(ct.float32)
        diff = x_tile - mean
        var_acc = var_acc + diff * diff
    var = ct.sum(var_acc) / D
    rstd = 1.0 / ct.sqrt(var + eps)

    # Pass 3: normalize + affine
    for j in range(num_col_tiles):
        x_tile = ct.load(X, index=(bid, j), shape=(1, TILE_D),
                         padding_mode=zero_pad).astype(ct.float32)
        w_tile = ct.load(W, index=(j,), shape=(TILE_D,),
                         padding_mode=zero_pad).astype(ct.float32)
        b_tile = ct.load(B, index=(j,), shape=(TILE_D,),
                         padding_mode=zero_pad).astype(ct.float32)
        y_tile = (x_tile - mean) * rstd * w_tile + b_tile
        ct.store(Y, index=(bid, j), tile=y_tile.astype(Y.dtype))


# ---------------------------------------------------------------------------
# LEGO cuTile layernorm — gather variant with layout-computed offsets
# ---------------------------------------------------------------------------

@cutile_jit
@ct.kernel
def layernorm_lego(X, Y, W, B, N: int, D: int, eps: float, TILE_D: ConstInt):
    bid = ct.bid(0)
    num_col_tiles = ct.cdiv(D, TILE_D)
    zero_pad = ct.PaddingMode.ZERO

    # LEGO layout for column tile offsets
    L_cols = OrderBy(Row(D)).TileBy([D / TILE_D], [TILE_D])

    # Pass 1: compute mean
    mean_acc = ct.full((1, TILE_D), 0, dtype=ct.float32)
    for j in range(num_col_tiles):
        x_tile = ct.load(X, index=(bid, j), shape=(1, TILE_D),
                         padding_mode=zero_pad).astype(ct.float32)
        mean_acc = mean_acc + x_tile
    mean = ct.sum(mean_acc) / D

    # Pass 2: compute variance
    var_acc = ct.full((1, TILE_D), 0, dtype=ct.float32)
    for j in range(num_col_tiles):
        x_tile = ct.load(X, index=(bid, j), shape=(1, TILE_D),
                         padding_mode=zero_pad).astype(ct.float32)
        diff = x_tile - mean
        var_acc = var_acc + diff * diff
    var = ct.sum(var_acc) / D
    rstd = 1.0 / ct.sqrt(var + eps)

    # Pass 3: normalize + affine using LEGO-computed column offsets for W/B
    for j in range(num_col_tiles):
        x_tile = ct.load(X, index=(bid, j), shape=(1, TILE_D),
                         padding_mode=zero_pad).astype(ct.float32)
        col_offs = L_cols[j, :]
        w_tile = ct.gather(W, col_offs).astype(ct.float32)
        b_tile = ct.gather(B, col_offs).astype(ct.float32)
        y_tile = (x_tile - mean) * rstd * w_tile + b_tile
        ct.store(Y, index=(bid, j), tile=y_tile.astype(Y.dtype))


# ---------------------------------------------------------------------------
# Verification & source inspection
# ---------------------------------------------------------------------------

def verify_source():
    """Check that LEGO eliminates layout expressions."""
    def ln_raw(X, Y, W, B, N, D, eps, TILE_D):
        bid = ct.bid(0)
        j = 0
        L_cols = OrderBy(Row(D)).TileBy([D / TILE_D], [TILE_D])
        col_offs = L_cols[j, :]

    src = get_cutile_kernel_source(ln_raw)
    print("=== LEGO-generated cuTile layernorm (offset fragment) ===")
    print(src)
    print()

    assert 'OrderBy' not in src, "Layout expression not eliminated"
    assert 'ct.arange' in src, "ct.arange not generated"
    assert 'TILE_D * j' in src, "Column offset not simplified"
    print("Source verification passed.\n")


def verify_gpu():
    """Run both kernels on GPU and compare against PyTorch."""
    N, D = 128, 256
    TILE_D = 64
    eps = 1e-5

    X = torch.randn(N, D, device='cuda', dtype=torch.float32)
    W = torch.ones(D, device='cuda', dtype=torch.float32)
    B = torch.zeros(D, device='cuda', dtype=torch.float32)

    # PyTorch reference
    expected = torch.nn.functional.layer_norm(X, [D], W, B, eps)

    stream = torch.cuda.current_stream()
    grid = (N, 1, 1)

    # Reference cuTile kernel
    Y_ref = torch.empty_like(X)
    ct.launch(stream, grid, layernorm_cutile, (X, Y_ref, W, B, D, eps, TILE_D))
    assert torch.allclose(Y_ref, expected, atol=1e-3, rtol=1e-3), \
        "Reference cuTile layernorm mismatch!"
    print("Reference cuTile layernorm: PASS")

    # LEGO cuTile kernel
    Y_lego = torch.empty_like(X)
    ct.launch(stream, grid, layernorm_lego,
              (X, Y_lego, W, B, N, D, eps, TILE_D))
    assert torch.allclose(Y_lego, expected, atol=1e-3, rtol=1e-3), \
        "LEGO cuTile layernorm mismatch!"
    print("LEGO cuTile layernorm:      PASS")

    assert torch.allclose(Y_ref, Y_lego, atol=1e-5), \
        "Reference and LEGO layernorm differ!"
    print("Ref vs LEGO match:          PASS\n")


if __name__ == "__main__":
    verify_source()
    if torch.cuda.is_available():
        verify_gpu()
    else:
        print("No GPU available, skipping GPU verification.")
