"""Shared-memory matrix transpose — @gpu_kernel DSL.

Same tiled layouts as transpose_naive, plus a shared memory buffer
whose layout is changed mid-kernel via set_layout() between the
read and write phases.

Phase 1: global A → shared memory (Row-tiled write layout)
Phase 2: shared memory → global B (Row-tiled read layout, indices swapped)
"""
import sys
import numpy as np
from lego.core import OrderBy, Row, Col
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 3:
    print("Usage: python transpose_smem.py NX NY")
    sys.exit(1)

NX = int(sys.argv[1])
NY = int(sys.argv[2])
N = NX

TILE_DIM = 32
BRX = 8
BRY = 32

# --- Global layouts ---
A_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])
B_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])

# --- Shared memory layout (initial) ---
smem_layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy([TILE_DIM, TILE_DIM])

# --- Phase layouts for shared memory ---
smem_write_layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy(
    [TILE_DIM // BRX, TILE_DIM // BRY], [BRX, BRY])
smem_read_layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy(
    [TILE_DIM // BRY, TILE_DIM // BRX], [BRY, BRX])


@gpu_kernel(
    grid=(NX // TILE_DIM * NY // TILE_DIM, 1),
    block=(BRY * BRX, 1),
)
def transpose_smem(A: Buffer(A_layout, N, N), B: Buffer(B_layout, N, N),
                   Smem: Shared(smem_layout, TILE_DIM, TILE_DIM)):
    bX = block_id.x
    tX = thread_id.x

    rby, rbx = apply_inverse(
        OrderBy(Row(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)
    wby, wbx = apply_inverse(
        OrderBy(Col(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)

    rty, rtx = apply_inverse(
        OrderBy(Row(BRX, BRY)).TileBy([BRX, BRY]), tX)
    wty, wtx = apply_inverse(
        OrderBy(Col(BRY, BRX)).TileBy([BRY, BRX]), tX)

    # --- Phase 1: global A → shared memory ---
    set_layout(Smem, smem_write_layout)
    for j in range(TILE_DIM // BRX):
        for i in range(TILE_DIM // BRY):
            val = A[rby, rbx, j, i, rty, rtx]
            Smem[j, i, rty, rtx] = val

    barrier()

    # --- Phase 2: shared memory → global B ---
    set_layout(Smem, smem_read_layout)
    for j in range(TILE_DIM // BRY):
        for i in range(TILE_DIM // BRX):
            val = Smem[j, i, wty, wtx]
            B[wby, wbx, i, j, rty, rtx] = val


from bench_utils import run_transpose_benchmark


if __name__ == "__main__":
    run_transpose_benchmark(
        transpose_smem, {"A": A_layout, "B": B_layout}, N,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
    )
