"""Puzzle 3 — Guards: add 10 with boundary checking.

Launch more threads than data elements. Threads beyond SIZE must not
write. In LEGO, SIZE is a compile-time constant captured from Python scope.

Mojo equivalent:
    var i = thread_idx.x
    if i < size:
        output[i] = a[i] + 10.0
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_03_guards.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
# Round up to next power of 2 for TPB
TPB = 1
while TPB < SIZE:
    TPB *= 2

# Buffers are TPB-sized but only SIZE elements are valid
layout = Row(TPB)


@gpu_kernel(grid=(1,), block=(TPB,))
def add_10_guard(A: Buffer(layout, TPB), Out: Buffer(layout, TPB)):
    i = thread_id.x
    if i < SIZE:
        Out[i] = A[i] + 10.0


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        out = np.zeros(TPB, dtype=np.float32)
        out[:SIZE] = inputs[0][:SIZE] + 10.0
        return out

    run_benchmark(
        add_10_guard, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"SIZE={SIZE},TPB={TPB}",
    )
