# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 4 16
# REQUIRES: gpu
"""Puzzle 18 — Softmax: compute softmax along rows of a 2D matrix.

softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))

Three phases per row (matches Mojo solution):
  1. Find max via tree reduction (double-barrier pattern)
  2. Compute exp(x - max) and sum via tree reduction
  3. Normalize: output = exp_val / sum
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 3:
    print("Usage: python puzzle_18_softmax.py BATCH SIZE")
    sys.exit(1)

BATCH = int(sys.argv[1])
SIZE = int(sys.argv[2])
assert SIZE > 0 and (SIZE & (SIZE - 1)) == 0, "SIZE must be power of 2"
TPB = SIZE

A_layout = Row(BATCH, SIZE)
out_layout = Row(BATCH, SIZE)
smem_layout = Row(TPB)


@gpu_kernel(grid=(BATCH,), block=(TPB,))
def softmax(A: Buffer(A_layout, BATCH, SIZE), Out: Buffer(out_layout, BATCH, SIZE),
            smem: Shared(smem_layout, TPB)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    # Phase 1: Find max via tree reduction
    # Use smem for max; write directly to avoid nested-if yield issues
    smem[tx] = val
    barrier()
    stride = TPB // 2
    while stride > 0:
        if tx < stride:
            if smem[tx + stride] > smem[tx]:
                smem[tx] = smem[tx + stride]
        barrier()
        stride = stride // 2
    # max is in smem[0]; all threads read it
    barrier()
    # Phase 2: Compute exp(val - max) and store in output + smem
    Out[bx, tx] = exp(val - smem[0])
    smem[tx] = Out[bx, tx]
    barrier()
    # Sum reduction (double-barrier)
    stride = TPB // 2
    while stride > 0:
        if tx < stride:
            smem[tx] = smem[tx] + smem[tx + stride]
        barrier()
        stride = stride // 2
    # Phase 3: Normalize
    barrier()
    Out[bx, tx] = Out[bx, tx] / smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(BATCH, SIZE)
        max_vals = a.max(axis=1, keepdims=True)
        exp_vals = np.exp(a - max_vals)
        sum_exp = exp_vals.sum(axis=1, keepdims=True)
        return (exp_vals / sum_exp).ravel().astype(np.float32)

    run_benchmark(
        softmax, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "vulkan", "webgpu", "webgl", "metal"],
        label=f"{BATCH}x{SIZE}",
        init_mod=10,
        atol=1e-4,
    )
