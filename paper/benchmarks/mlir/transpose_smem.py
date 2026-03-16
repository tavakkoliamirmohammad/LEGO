"""Shared-memory matrix transpose — pure MLIR LEGO dialect (no SymPy index math)."""
from lego.lego_mlir import (
    MLIRTensor, printer,
    mlir_apply_inverse, mlir_load, mlir_store, mlir_loop,
)
from lego.lego import OrderBy, Row, Col, divisibility_constraint
import sys

if len(sys.argv) != 3:
    print("Usage: python script_name.py NX NY")
    sys.exit(1)

NX = int(sys.argv[1])
NY = int(sys.argv[2])
N = NX

TILE_DIM = 32
BLOCK_ROWS_X = 8
BLOCK_ROWS_Y = 32
NUM_REPETITION = 125

dimGrid = (NX // TILE_DIM * NY // TILE_DIM, 1, 1)
dimBlock = (BLOCK_ROWS_Y * BLOCK_ROWS_X, 1, 1)

user_constraints = [
    divisibility_constraint(NX, TILE_DIM),
    divisibility_constraint(NY, TILE_DIM),
    divisibility_constraint(TILE_DIM, BLOCK_ROWS_X),
    divisibility_constraint(TILE_DIM, BLOCK_ROWS_Y),
]

A = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BLOCK_ROWS_X, TILE_DIM // BLOCK_ROWS_Y],
    [BLOCK_ROWS_X, BLOCK_ROWS_Y])
B = OrderBy(Row(N, N)).TileBy(
    [N // TILE_DIM, N // TILE_DIM],
    [TILE_DIM // BLOCK_ROWS_X, TILE_DIM // BLOCK_ROWS_Y],
    [BLOCK_ROWS_X, BLOCK_ROWS_Y])
smem = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy([TILE_DIM, TILE_DIM])

TA = MLIRTensor(A, "f32")
TB = MLIRTensor(B, "f32")
TSmem = MLIRTensor(smem, "f32")


@printer.generate_mlir()
def main():
    @printer.generate_loop(OrderBy(Row(NUM_REPETITION)).TileBy([NUM_REPETITION]))
    def main_body(_, __):
        @printer.generate_gpu_kernel([TA], [TB], dimGrid, dimBlock, workgroup_memory=[TSmem])
        def kernel(args):
            from mlir.dialects import gpu
            bX = gpu.block_id(gpu.Dimension.x)
            tX = gpu.thread_id(gpu.Dimension.x)

            rby, rbx = mlir_apply_inverse(
                OrderBy(Row(N // TILE_DIM, N // TILE_DIM))
                .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)
            wby, wbx = mlir_apply_inverse(
                OrderBy(Col(N // TILE_DIM, N // TILE_DIM))
                .TileBy([N // TILE_DIM, N // TILE_DIM]), bX)

            rty, rtx = mlir_apply_inverse(
                OrderBy(Row(BLOCK_ROWS_X, BLOCK_ROWS_Y))
                .TileBy([BLOCK_ROWS_X, BLOCK_ROWS_Y]), tX)
            wty, wtx = mlir_apply_inverse(
                OrderBy(Col(BLOCK_ROWS_Y, BLOCK_ROWS_X))
                .TileBy([BLOCK_ROWS_Y, BLOCK_ROWS_X]), tX)

            TSmem.layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy(
                [TILE_DIM // BLOCK_ROWS_X, TILE_DIM // BLOCK_ROWS_Y],
                [BLOCK_ROWS_X, BLOCK_ROWS_Y])

            read_tile_loop = OrderBy(
                Row(TILE_DIM // BLOCK_ROWS_X, TILE_DIM // BLOCK_ROWS_Y)
            ).TileBy([TILE_DIM // BLOCK_ROWS_X, TILE_DIM // BLOCK_ROWS_Y])

            def read_body(pargs, _):
                j, i = pargs
                mlir_store(mlir_load(TA, [rby, rbx, j, i, rty, rtx]),
                           TSmem, [j, i, rty, rtx])

            mlir_loop(read_tile_loop, read_body)

            printer.insert_barrier()

            TSmem.layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy(
                [TILE_DIM // BLOCK_ROWS_Y, TILE_DIM // BLOCK_ROWS_X],
                [BLOCK_ROWS_Y, BLOCK_ROWS_X])

            write_tile_loop = OrderBy(
                Row(TILE_DIM // BLOCK_ROWS_Y, TILE_DIM // BLOCK_ROWS_X)
            ).TileBy([TILE_DIM // BLOCK_ROWS_Y, TILE_DIM // BLOCK_ROWS_X])

            def write_body(pargs, _):
                j, i = pargs
                mlir_store(mlir_load(TSmem, [j, i, wty, wtx]),
                           TB, [wby, wbx, i, j, rty, rtx])

            mlir_loop(write_tile_loop, write_body)
