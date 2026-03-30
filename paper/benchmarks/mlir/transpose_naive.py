"""Naive matrix transpose — @gpu_kernel DSL (all GPU backends).

Layouts (3-level TileBy):
  A: OrderBy(Row(N, N)).TileBy(
       [N//TD, N//TD],           — tile grid
       [TD//BRX, TD//BRY],       — inner tiles per thread group
       [BRX, BRY])               — elements per thread
  B: same structure but BRX/BRY swapped in inner levels

Index decomposition via apply_inverse:
  block_id.x  → (tile_row, tile_col) via Row or Col ordering
  thread_id.x → (thread_row, thread_col)

The transpose is achieved by:
  - Reading  tile (rby, rbx) in Row order
  - Writing  tile (wby, wbx) in Col order
  - Swapping inner indices (j, i) → (i, j) in the store
"""
import sys
import numpy as np
from lego.core import OrderBy, Row, Col
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 3:
    print("Usage: python transpose_naive.py NX NY")
    sys.exit(1)

NX = int(sys.argv[1])
NY = int(sys.argv[2])
N = NX

TILE_DIM = 32
BRX = 8    # BLOCK_ROWS_X
BRY = 32   # BLOCK_ROWS_Y

# --- Layouts ---
A_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])
B_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRY, TILE_DIM // BRX],
    [BRY, BRX])


@gpu_kernel(
    grid=(NX // TILE_DIM * NY // TILE_DIM, 1),
    block=(BRY * BRX, 1),
)
def transpose_naive(A: Buffer(A_layout, N, N), B: Buffer(B_layout, N, N)):
    bX = block_id.x
    tX = thread_id.x

    # Decompose block index → tile coordinates
    rby, rbx = apply_inverse(
        OrderBy(Row(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)
    wby, wbx = apply_inverse(
        OrderBy(Col(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)

    # Decompose thread index → element coordinates
    rty, rtx = apply_inverse(
        OrderBy(Row(BRX, BRY)).TileBy([BRX, BRY]), tX)
    wty, wtx = apply_inverse(
        OrderBy(Col(BRY, BRX)).TileBy([BRY, BRX]), tX)

    # Transpose: swap (j, i) indices between load and store
    for j in range(TILE_DIM // BRX):
        for i in range(TILE_DIM // BRY):
            val = A[rby, rbx, j, i, rty, rtx]
            B[wby, wbx, i, j, wty, wtx] = val


from bench_utils import run_transpose_benchmark


if __name__ == "__main__":
    run_transpose_benchmark(
        transpose_naive, {"A": A_layout, "B": B_layout}, N,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
    )
