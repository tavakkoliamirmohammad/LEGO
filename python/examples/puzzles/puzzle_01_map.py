# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 1 — Map: add 10 to each element of a vector.

LEGO layout: OrderBy(Row(N)).TileBy([N // WG], [WG])
  - Flat index = block_id.x * WG + thread_id.x
  - Layout handles the arithmetic; kernel just writes A[bx, tx].

Mojo equivalent:
    var i = thread_idx.x
    output[i] = a[i] + 10.0
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_01_map.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0

layout = OrderBy(Row(N)).TileBy([N // WG], [WG])


@gpu_kernel(grid=(N // WG,), block=(WG,))
def map_add10(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    Out[bx, tx] = A[bx, tx] + 10.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + 10.0).astype(np.float32)

    run_benchmark(
        map_add10, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"N={N}",
    )
