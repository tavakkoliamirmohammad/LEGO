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

user_constraints = [divisibility_constraint(NX, TILE_DIM), divisibility_constraint(NY, TILE_DIM), divisibility_constraint(TILE_DIM, BLOCK_ROWS_X),
                    divisibility_constraint(TILE_DIM, BLOCK_ROWS_Y)]


@printer.generate_mlir()
def main():
    a_order = OrderBy(Row(N, N))
    b_order = OrderBy(Row(N, N))
    A = a_order.TileBy([N//TILE_DIM, N//TILE_DIM], [TILE_DIM//BLOCK_ROWS_X,
                                                    TILE_DIM//BLOCK_ROWS_Y], [BLOCK_ROWS_X, BLOCK_ROWS_Y])
    B = b_order.TileBy([N//TILE_DIM, N//TILE_DIM], [TILE_DIM//BLOCK_ROWS_X,
                       TILE_DIM//BLOCK_ROWS_Y], [BLOCK_ROWS_X, BLOCK_ROWS_Y])
    smem_order = OrderBy(Row(TILE_DIM, TILE_DIM))
    smem = smem_order.TileBy([TILE_DIM, TILE_DIM])

    TA = MLIRTensor(A, "f32")
    TB = MLIRTensor(B, "f32")
    TSmem = MLIRTensor(smem, "f32")

    @printer.generate_loop(OrderBy(Row(NUM_REPETION)).TileBy([NUM_REPETION]))
    def main_body(_, __):
        @printer.generate_gpu_kernel([TA], [TB], dimGrid, dimBlock, workgroup_memory=[TSmem])
        def kernel(args):
            bX = printer.get_block_linear_index()
            tX = printer.get_thread_linear_index()

            (rby, rbx) = OrderBy(Row(N//TILE_DIM, N//TILE_DIM)
                                 ).TileBy([N//TILE_DIM, N//TILE_DIM]).inv(bX)
            (wby, wbx) = OrderBy(
                Col(N//TILE_DIM, N//TILE_DIM)).TileBy([N//TILE_DIM, N//TILE_DIM]).inv(bX)

            (rty, rtx) = OrderBy(
                Row(BLOCK_ROWS_X, BLOCK_ROWS_Y)).TileBy([BLOCK_ROWS_X, BLOCK_ROWS_Y]).inv(tX)
            (wty, wtx) = OrderBy(
                Col(BLOCK_ROWS_Y, BLOCK_ROWS_X)).TileBy([BLOCK_ROWS_Y, BLOCK_ROWS_X]).inv(tX)

            TSmem.layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy([TILE_DIM//BLOCK_ROWS_X, TILE_DIM //
                                                                    BLOCK_ROWS_Y], [BLOCK_ROWS_X, BLOCK_ROWS_Y])

            @printer.generate_loop(OrderBy(Row(TILE_DIM//BLOCK_ROWS_X, TILE_DIM//BLOCK_ROWS_Y)).TileBy([TILE_DIM//BLOCK_ROWS_X, TILE_DIM//BLOCK_ROWS_Y]))
            def _(pargs, _):
                j, i = pargs
                # j, i, rty, rtx, rby, rbx = sp.symbols('j i rty rtx rby rbx', integer=True, postive=True)
                # print(TA[rby, rbx, j, i, rty, rtx])
                # print(TSmem[j, i, rty, rtx])
                TSmem[j, i, rty, rtx] = TA[rby, rbx, j, i, rty, rtx]

            printer.insert_barrier()

            TSmem.layout = OrderBy(Row(TILE_DIM, TILE_DIM)).TileBy([TILE_DIM//BLOCK_ROWS_Y, TILE_DIM //
                                                                    BLOCK_ROWS_X], [BLOCK_ROWS_Y, BLOCK_ROWS_X])

            @printer.generate_loop(OrderBy(Row(TILE_DIM//BLOCK_ROWS_Y, TILE_DIM//BLOCK_ROWS_X)).TileBy([TILE_DIM//BLOCK_ROWS_Y, TILE_DIM//BLOCK_ROWS_X]))
            def matmul_body(pargs, _):
                j, i = pargs
                TB[wby, wbx, i, j, rty, rtx] = TSmem[j, i, wty, wtx]

        # TA.print_matrix_kernel(a_order, NX, NY)
        # TB.print_matrix_kernel(b_order, NY, NX)
