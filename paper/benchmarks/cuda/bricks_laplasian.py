from lego import *
import jinja2
from lego.lego_c import *

a, i, j, k, radius, i_diff, j_diff, k_diff, bx, by, bz = symbols(
    'a i j k radius i_diff j_diff k_diff bx by bz', integer=True, positive=True)

N = 384
B = 8
normal = OrderBy(Row(N, N, N)).TileBy([N//B, N//B, N//B], [B, B, B])
bricks = OrderBy(Row(N // B, N//B, N//B), Row(B, B, B)
                 ).TileBy([N//B, N//B, N//B], [B, B, B])
const = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])


def get_computation_indices(l):
    return l[bx, by, bz, i+a, j, k], l[bx, by, bz, i, j+a, k], l[bx, by, bz, i, j, k+a], l[bx, by, bz, i, j, k-a], l[bx, by, bz, i, j-a, k], l[bx, by, bz, i-a, j, k]

normal_indices = get_computation_indices(normal)
normal_in0_idx = normal_indices[0]
normal_in1_idx = normal_indices[1]
normal_in2_idx = normal_indices[2]
normal_in3_idx = normal_indices[3]
normal_in4_idx = normal_indices[4]
normal_in5_idx = normal_indices[5]

bricks_indices = get_computation_indices(bricks)
bricks_in0_idx = bricks_indices[0]
bricks_in1_idx = bricks_indices[1]
bricks_in2_idx = bricks_indices[2]
bricks_in3_idx = bricks_indices[3]
bricks_in4_idx = bricks_indices[4]
bricks_in5_idx = bricks_indices[5]

normal_out_idx = normal[bx, by, bz, i, j, k]
bricks_out_idx = bricks[bx, by, bz, i, j, k]

# Read the file content
with open('bricks-laplace/main_tmp.cu', 'r') as file:
    input_content = file.read()

kernel_template = jinja2.Template(input_content)

params = {
    'normal_in0_idx': normal_in0_idx,
    'normal_in1_idx': normal_in1_idx,
    'normal_in2_idx': normal_in2_idx,
    'normal_in3_idx': normal_in3_idx,
    'normal_in4_idx': normal_in4_idx,
    'normal_in5_idx': normal_in5_idx,
    'bricks_in0_idx': bricks_in0_idx,
    'bricks_in1_idx': bricks_in1_idx,
    'bricks_in2_idx': bricks_in2_idx,
    'bricks_in3_idx': bricks_in3_idx,
    'bricks_in4_idx': bricks_in4_idx,
    'bricks_in5_idx': bricks_in5_idx,
    'normal_out_idx': normal_out_idx,
    'bricks_out_idx': bricks_out_idx,
    'N': N,
}
# Create a C code printer
printer = LEGOCCodePrinter()

render_params = {key: printer.doprint(sp.simplify(value))
                 for key, value in params.items()}

kernel_code = kernel_template.render(**render_params)
with open('bricks-laplace/main.cu', 'w') as file:
    file.write(kernel_code)
