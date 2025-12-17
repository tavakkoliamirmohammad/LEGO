import sympy as sp
import jinja2
from lego_c import *


n, i, j = sp.symbols('n i j', integer=True, postive=True)


anti_diag = OrderBy(
    GenP([n, n], lambda x: antidiag(n, x), None)).TileBy((n, n))
expr = anti_diag[i, j]
# Read the file content
with open('nw/needle_kernel_temp.cu', 'r') as file:
    input_content = file.read()

kernel_template = jinja2.Template(input_content)


params = {
    'expr': expr,
}
# Create a C code printer
printer = LEGOCCodePrinter()

render_params = {key: printer.doprint(sp.simplify(value))
                 for key, value in params.items()}

kernel_code = kernel_template.render(**render_params)

with open('nw/needle_kernel.cu', 'w') as f:
    f.write(kernel_code)
