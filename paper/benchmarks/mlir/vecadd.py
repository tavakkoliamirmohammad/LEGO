"""Vector addition C = A + B — @gpu_kernel DSL (all GPU backends).

Layout: OrderBy(Row(N)).TileBy([N // WG], [WG])
  - 1st index = block (workgroup) index
  - 2nd index = thread index within workgroup
  - lego.apply computes: block * WG + thread
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python vecadd.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = 256

# --- Layout: 1D vector tiled by workgroup ---
vec_layout = OrderBy(Row(N)).TileBy([N // WG], [WG])


@gpu_kernel(grid=(N // WG,), block=(WG,))
def vecadd(A: Buffer(vec_layout, N), B: Buffer(vec_layout, N),
           C: Buffer(vec_layout, N)):
    bx = block_id.x
    tx = thread_id.x
    C[bx, tx] = A[bx, tx] + B[bx, tx]


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + inputs[1]).astype(np.float32)

    run_benchmark(
        vecadd, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
    )
