from lego import *
import jinja2
from lego_c import *

i, j, k, radius, i_diff, j_diff, k_diff, bx, by, bz = symbols(
    'i j k radius i_diff j_diff k_diff bx by bz', integer=True, positive=True)

N = 384
B = 8
normal = OrderBy(Row(N, N, N)).TileBy([N//B, N//B, N//B], [B, B, B])
bricks = OrderBy(Row(N // B, N//B, N//B), Row(B, B, B)
                 ).TileBy([N//B, N//B, N//B], [B, B, B])
const = OrderBy(Row(8, 8, 8)).TileBy([8, 8, 8])

normal_in_idx = normal[bx, by, bz, i + i_diff, j + j_diff, k + k_diff]
bricks_in_idx = bricks[bx, by, bz, i + i_diff, j + j_diff, k + k_diff]
const_idx = const[i_diff + radius, j_diff + radius, k_diff + radius]
const_out_idx = const[radius, radius, radius]

normal_out_idx = normal[bx, by, bz, i, j, k]
bricks_out_idx = bricks[bx, by, bz, i, j, k]


# Read the file content
with open('bricks/main_tmp.cu', 'r') as file:
    input_content = file.read()

kernel_template = jinja2.Template(input_content)

params = {
    'normal_in_idx': normal_in_idx,
    'bricks_in_idx': bricks_in_idx,
    'const_idx': const_idx,
    'const_out_idx': const_out_idx,
    'normal_out_idx': normal_out_idx,
    'bricks_out_idx': bricks_out_idx,
    'N': N,
}
# Create a C code printer
printer = LEGOCCodePrinter()

render_params = {key: printer.doprint(sp.simplify(value))
                 for key, value in params.items()}

kernel_code = kernel_template.render(**render_params)
with open('bricks/main.cu', 'w') as file:
    file.write(kernel_code)
