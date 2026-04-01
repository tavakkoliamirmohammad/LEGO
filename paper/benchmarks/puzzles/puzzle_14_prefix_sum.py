"""Puzzle 14 — Prefix Sum: compute inclusive prefix sum (running total).

Uses Hillis-Steele algorithm: stride doubles each step.
LEGO's compile-time while loop handles this naturally.

output[i] = sum(a[0..i])

Mojo equivalent:
    shared[i] = a[i]
    barrier()
    stride = 1
    while stride < TPB:
        if i >= stride:
            shared[i] += shared[i - stride]
        barrier()
        stride *= 2
    output[i] = shared[i]
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_14_prefix_sum.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
TPB = SIZE  # single block for simplicity

layout = Row(SIZE)
smem_layout = Row(TPB)


@gpu_kernel(grid=(1,), block=(TPB,))
def prefix_sum(A: Buffer(layout, SIZE), Out: Buffer(layout, SIZE),
               smem: Shared(smem_layout, TPB)):
    tx = thread_id.x
    smem[tx] = A[tx]
    barrier()
    # Hillis-Steele: stride doubles each step
    stride = 1
    while stride < TPB:
        if tx >= stride:
            smem[tx] = smem[tx] + smem[tx - stride]
        barrier()
        stride = stride * 2
    Out[tx] = smem[tx]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return np.cumsum(inputs[0][:SIZE]).astype(np.float32)

    run_benchmark(
        prefix_sum, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"SIZE={SIZE}",
        init_mod=10,
        atol=1e-4,
    )
