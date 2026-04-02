"""Puzzle 8 — Shared Memory: add 10 using shared memory as intermediate.

Pattern: global → shared → compute → global.
Demonstrates the shared memory load/barrier/compute pattern.

Mojo equivalent:
    shared = stack_allocation[TPB, dtype]()
    var i = thread_idx.x
    if i < size:
        shared[i] = a[i]
    barrier()
    if i < size:
        output[i] = shared[i] + 10.0
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_08_shared_mem.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0
num_blocks = N // WG

layout = OrderBy(Row(N)).TileBy([num_blocks], [WG])
smem_layout = Row(WG)


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def add_10_shared(A: Buffer(layout, N), Out: Buffer(layout, N),
                  smem: Shared(smem_layout, WG)):
    bx = block_id.x
    tx = thread_id.x
    # Load global → shared
    smem[tx] = A[bx, tx]
    barrier()
    # Compute from shared → global
    Out[bx, tx] = smem[tx] + 10.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + 10.0).astype(np.float32)

    run_benchmark(
        add_10_shared, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
    )
