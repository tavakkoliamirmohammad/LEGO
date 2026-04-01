"""Puzzle 10 — Memory Error Detection: demonstrate guard patterns preventing OOB.

The Mojo puzzle teaches memory violation detection. Here we demonstrate
the guard pattern: launching more threads than data elements, with
bounds checking to prevent out-of-bounds access.

This is the same pattern as Puzzle 3 (Guards) but with 2D indexing
and explicit boundary checking on both dimensions.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_10_memory_errors.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
# Over-allocate: TPB > SIZE to demonstrate guard necessity
TPB = 1
while TPB < SIZE:
    TPB *= 2

layout = Row(TPB, TPB)


@gpu_kernel(grid=(1,), block=(TPB, TPB))
def guarded_2d(A: Buffer(layout, TPB, TPB), Out: Buffer(layout, TPB, TPB)):
    row = thread_id.y
    col = thread_id.x
    # Guard: only valid threads (within SIZE x SIZE) write
    if row < SIZE:
        if col < SIZE:
            Out[row, col] = A[row, col] * 2.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(TPB, TPB)
        out = np.zeros((TPB, TPB), dtype=np.float32)
        out[:SIZE, :SIZE] = a[:SIZE, :SIZE] * 2.0
        return out.ravel()

    run_benchmark(
        guarded_2d, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"SIZE={SIZE},TPB={TPB}",
    )
