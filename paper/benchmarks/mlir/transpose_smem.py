"""Shared-memory matrix transpose — unified KernelBuilder API (all GPU backends).

Includes host-side GPU execution verification via CUDA (mlir-runner).
WebGPU not available for this kernel due to shared memory limitation.
"""
import sys
from lego.core import OrderBy, Row, Col
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer

if len(sys.argv) != 3:
    print("Usage: python transpose_smem.py NX NY")
    sys.exit(1)

NX = int(sys.argv[1])
NY = int(sys.argv[2])
N = NX

TILE_DIM = 32
BRX = 8   # BLOCK_ROWS_X
BRY = 32  # BLOCK_ROWS_Y

dimGrid = (NX // TILE_DIM * NY // TILE_DIM, 1, 1)
dimBlock = (BRY * BRX, 1, 1)

# --- Global buffer layouts ---
A_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])
B_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])

# --- Shared memory layout (initial — will be swapped between phases) ---
smem_layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy([TILE_DIM, TILE_DIM])

A = LayoutBuffer(A_layout, shape=(N, N), dtype="f32")
B = LayoutBuffer(B_layout, shape=(N, N), dtype="f32")
Smem = LayoutBuffer(smem_layout, shape=(TILE_DIM, TILE_DIM), dtype="f32", shared=True)


def transpose_smem_kernel(ctx):
    bX = ctx.block_id.x
    tX = ctx.thread_id.x

    rby, rbx = ctx.apply_inverse(
        OrderBy(Row(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)
    wby, wbx = ctx.apply_inverse(
        OrderBy(Col(N // TILE_DIM, N // TILE_DIM))
        .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)

    rty, rtx = ctx.apply_inverse(
        OrderBy(Row(BRX, BRY)).TileBy([BRX, BRY]), tX)
    wty, wtx = ctx.apply_inverse(
        OrderBy(Col(BRY, BRX)).TileBy([BRY, BRX]), tX)

    # --- Phase 1: Read from global A → write to shared memory ---
    ctx.set_layout(2, OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy(
        [TILE_DIM // BRX, TILE_DIM // BRY], [BRX, BRY]))

    read_tile = OrderBy(
        Row(TILE_DIM // BRX, TILE_DIM // BRY)
    ).TileBy([TILE_DIM // BRX, TILE_DIM // BRY])

    def read_body(indices, _):
        j, i = indices
        val = ctx.load(0, [rby, rbx, j, i, rty, rtx])
        ctx.store(val, 2, [j, i, rty, rtx])

    ctx.tile_loop(read_tile, read_body)

    # --- Barrier ---
    ctx.barrier()

    # --- Phase 2: Read from shared memory → write to global B ---
    ctx.set_layout(2, OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy(
        [TILE_DIM // BRY, TILE_DIM // BRX], [BRY, BRX]))

    write_tile = OrderBy(
        Row(TILE_DIM // BRY, TILE_DIM // BRX)
    ).TileBy([TILE_DIM // BRY, TILE_DIM // BRX])

    def write_body(indices, _):
        j, i = indices
        val = ctx.load(2, [j, i, wty, wtx])
        ctx.store(val, 1, [wby, wbx, i, j, rty, rtx])

    ctx.tile_loop(write_tile, write_body)


builder = KernelBuilder(
    buffers=[A, B, Smem],
    kernel_body=transpose_smem_kernel,
    name="transpose_smem",
    grid=dimGrid,
    block=dimBlock,
)


from bench_utils import run_transpose_benchmark


if __name__ == "__main__":
    # SPIR-V targets don't yet support workgroup (shared) memory,
    # so only CUDA is available for the smem transpose.
    run_transpose_benchmark(
        builder, {"A": A_layout, "B": B_layout}, N,
        targets=["cuda"],
    )
