"""Naive matrix transpose — unified KernelBuilder API (all GPU backends).

Includes host-side GPU execution verification via both CUDA (mlir-runner)
and WebGPU (wgpu/Vulkan).
"""
import sys
from lego.core import OrderBy, Row, Col
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer

if len(sys.argv) != 3:
    print("Usage: python transpose_naive.py NX NY")
    sys.exit(1)

NX = int(sys.argv[1])
NY = int(sys.argv[2])
N = NX

TILE_DIM = 32
BRX = 8   # BLOCK_ROWS_X
BRY = 32  # BLOCK_ROWS_Y

dimGrid = (NX // TILE_DIM * NY // TILE_DIM, 1, 1)
dimBlock = (BRY * BRX, 1, 1)

# --- Layouts ---
A_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRX, TILE_DIM // BRY],
    [BRX, BRY])
B_layout = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BRY, TILE_DIM // BRX],
    [BRY, BRX])

A = LayoutBuffer(A_layout, shape=(N, N), dtype=DType.f32)
B = LayoutBuffer(B_layout, shape=(N, N), dtype=DType.f32)


def transpose_kernel(ctx):
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

    tile_loop = OrderBy(
        Row(TILE_DIM // BRX, TILE_DIM // BRY)
    ).TileBy([TILE_DIM // BRX, TILE_DIM // BRY])

    def transpose_body(indices, _):
        j, i = indices
        value = ctx.load(0, [rby, rbx, j, i, rty, rtx])
        ctx.store(value, 1, [wby, wbx, i, j, wty, wtx])

    ctx.tile_loop(tile_loop, transpose_body)


builder = KernelBuilder(
    buffers=[A, B],
    kernel_body=transpose_kernel,
    name="transpose_naive",
    grid=dimGrid,
    block=dimBlock,
)


from bench_utils import run_transpose_benchmark


if __name__ == "__main__":
    run_transpose_benchmark(
        builder, {"A": A_layout, "B": B_layout}, N,
        targets=["cuda", "vulkan", "webgpu", "metal"],
    )
