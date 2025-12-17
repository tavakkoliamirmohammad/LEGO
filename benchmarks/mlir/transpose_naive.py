from lego.lego_mlir import *
import sys

if len(sys.argv) != 3:
    print("Usage: python script_name.py NX NY")
    sys.exit(1)

try:
    NX = int(sys.argv[1])
    NY = int(sys.argv[2])
    N = NX
except ValueError:
    print("NX and NY must be integers")
    sys.exit(1)


TILE_DIM = 32
BLOCK_ROWS_X = 8
BLOCK_ROWS_Y = 32
NUM_REPETION = 125

dimGrid = (NX//TILE_DIM * NY//TILE_DIM, 1, 1)
dimBlock = (BLOCK_ROWS_Y * BLOCK_ROWS_X, 1, 1)


if __name__ == "__main__":
    @printer.generate_mlir()
    def main():
        A = OrderBy(Row(N, N)).TileBy(
            [N//TILE_DIM, N//TILE_DIM], [TILE_DIM//BLOCK_ROWS_X, TILE_DIM//BLOCK_ROWS_Y], [BLOCK_ROWS_X, BLOCK_ROWS_Y])
        B = OrderBy(Row(N, N)).TileBy(
            [N//TILE_DIM, N//TILE_DIM], [TILE_DIM//BLOCK_ROWS_Y, TILE_DIM//BLOCK_ROWS_X], [BLOCK_ROWS_Y, BLOCK_ROWS_X])

        TA = MLIRTensor(A, "f32")
        TB = MLIRTensor(B, "f32")

        @printer.generate_loop(OrderBy(Row(NUM_REPETION)).TileBy([NUM_REPETION]))
        def main_body(_, __):
            @printer.generate_gpu_kernel([TA], [TB], dimGrid, dimBlock)
            def kernel(_):
                bX = printer.get_block_linear_index()
                tX = printer.get_thread_linear_index()

                (rby, rbx) = OrderBy(
                    Row(N//TILE_DIM, N//TILE_DIM)).TileBy([N//TILE_DIM, N//TILE_DIM]).inv(bX)
                (wby, wbx) = OrderBy(
                    Col(N//TILE_DIM, N//TILE_DIM)).TileBy([N//TILE_DIM, N//TILE_DIM]).inv(bX)

                (rty, rtx) = OrderBy(
                    Row(BLOCK_ROWS_X, BLOCK_ROWS_Y)).TileBy([BLOCK_ROWS_X, BLOCK_ROWS_Y]).inv(tX)
                (wty, wtx) = OrderBy(
                    Col(BLOCK_ROWS_Y, BLOCK_ROWS_X)).TileBy([BLOCK_ROWS_Y, BLOCK_ROWS_X]).inv(tX)

                @printer.generate_loop(OrderBy(Row(TILE_DIM//BLOCK_ROWS_X, TILE_DIM//BLOCK_ROWS_Y)).TileBy([TILE_DIM//BLOCK_ROWS_X, TILE_DIM//BLOCK_ROWS_Y]))
                def transpose_body(pargs, _):
                    j, i = pargs
                    value_read = TA[rby, rbx, j, i, rty, rtx]
                    TB[wby, wbx, i, j, wty, wtx] = value_read
            # TA.print_matrix_kernel(OrderBy(Row(NX, NY)), NX, NY)
            # TB.print_matrix_kernel(OrderBy(Row(NX, NY)), NY, NX)
