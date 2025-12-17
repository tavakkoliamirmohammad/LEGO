from lego import *
import jinja2
from lego_c import *

R, T, ii, jj, tid, BLOCK_SIZE = symbols(
    'R T ii jj tid BLOCK_SIZE', integer=True, positive=True)
expr = (ii * R + jj) * T * T + tid
constraints = [le_constraint(ii, R), le_constraint(
    jj, R), le_constraint(tid, T * T), le_constraint(0, tid)]

l = OrderBy(Row(R * T, R * T)).GroupBy([(R, R), (T, T)], constraints)
i, j, tidx, tidy = l.inv(expr)

# Read the file content
with open('lud/lud_kernel_temp.cu', 'r') as file:
    input_content = file.read()

kernel_template = jinja2.Template(input_content)

params = {
    'i': i,
    'j': j,
    'tidx': tidx,
    'tidy': tidy,
}
# Create a C code printer
printer = LEGOCCodePrinter()

render_params = {key: printer.doprint(sp.simplify(value))
                 for key, value in params.items()}

kernel_code = kernel_template.render(**render_params)
with open('lud/lud_kernel.cu', 'w') as file:
    file.write(kernel_code)
