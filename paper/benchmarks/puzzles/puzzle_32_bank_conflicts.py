"""Puzzle 32 — Bank Conflicts: shared memory access pattern analysis.

Mojo puzzle: (input + 10) * 2 via shared memory with stride-1 access.
Both the no-conflict and 2-way-conflict versions produce the same result.
This is a profiling exercise — the puzzle is about NSight Compute analysis.

We implement the no-conflict version faithfully.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_32_bank_conflicts.py N")
    sys.exit(1)

N = int(sys.argv[1])
TPB = 256
assert N % TPB == 0
num_blocks = N // TPB

layout = OrderBy(Row(N)).TileBy([num_blocks], [TPB])
smem_layout = Row(TPB)


@gpu_kernel(grid=(num_blocks,), block=(TPB,))
def no_conflict(A: Buffer(layout, N), Out: Buffer(layout, N),
                smem: Shared(smem_layout, TPB)):
    bx = block_id.x
    tx = thread_id.x
    # Stride-1 access: no bank conflicts
    smem[tx] = A[bx, tx] + 10.0
    barrier()
    Out[bx, tx] = smem[tx] * 2.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return ((inputs[0] + 10.0) * 2.0).astype(np.float32)

    run_benchmark(
        no_conflict, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
    )
