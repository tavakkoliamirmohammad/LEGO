# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 16
# REQUIRES: gpu
"""Puzzle 14 — Prefix Sum: compute inclusive prefix sum (running total).

Hillis-Steele algorithm with double-barrier per step to avoid read-write
hazards (matches the Mojo solution exactly):
  1. Read smem[tx - offset] into temp variable
  2. Barrier (ensure all reads complete)
  3. Write smem[tx] += temp
  4. Barrier (ensure all writes complete before next step)

output[i] = sum(a[0..i])
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
    # Hillis-Steele with double-barrier (matches Mojo solution):
    # read phase → barrier → write phase → barrier
    offset = 1
    while offset < TPB:
        # Read phase: capture neighbor value before anyone writes
        current_val = 0.0
        if tx >= offset:
            current_val = smem[tx - offset]
        barrier()
        # Write phase: safe to write since all reads are done
        if tx >= offset:
            smem[tx] = smem[tx] + current_val
        barrier()
        offset = offset * 2
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
