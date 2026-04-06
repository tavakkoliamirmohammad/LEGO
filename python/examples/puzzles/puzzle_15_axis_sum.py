# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 4 256
# REQUIRES: gpu
"""Puzzle 15 — Axis Sum: sum each row of a 2D matrix.

Each block handles one row. Tree reduction within the block.

Mojo equivalent:
    shared[thread_idx.x] = a[block_idx.x, thread_idx.x]
    barrier()
    # tree reduction
    stride = TPB // 2
    while stride > 0:
        if thread_idx.x < stride:
            shared[thread_idx.x] += shared[thread_idx.x + stride]
        barrier()
        stride //= 2
    if thread_idx.x == 0:
        output[block_idx.x] = shared[0]
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 3:
    print("Usage: python puzzle_15_axis_sum.py BATCH SIZE")
    sys.exit(1)

BATCH = int(sys.argv[1])
SIZE = int(sys.argv[2])
TPB = SIZE
assert SIZE > 0 and (SIZE & (SIZE - 1)) == 0, "SIZE must be power of 2"

# 2D layout: A[batch, elem]
A_layout = Row(BATCH, SIZE)
out_layout = Row(BATCH)
smem_layout = Row(TPB)


@gpu_kernel(grid=(BATCH,), block=(TPB,))
def axis_sum(A: Buffer(A_layout, BATCH, SIZE), Out: Buffer(out_layout, BATCH),
             smem: Shared(smem_layout, TPB)):
    bx = block_id.x
    tx = thread_id.x
    # Load row into shared
    smem[tx] = A[bx, tx]
    barrier()
    # Tree reduction with double-barrier (matches Mojo: read → barrier → write → barrier)
    stride = TPB // 2
    while stride > 0:
        temp_val = 0.0
        if tx < stride:
            temp_val = smem[tx + stride]
        barrier()
        if tx < stride:
            smem[tx] = smem[tx] + temp_val
        barrier()
        stride = stride // 2
    # Thread 0 writes row sum
    if tx == 0:
        Out[bx] = smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(BATCH, SIZE)
        return a.sum(axis=1).astype(np.float32)

    run_benchmark(
        axis_sum, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "vulkan", "webgpu", "webgl", "metal"],
        label=f"{BATCH}x{SIZE}",
        init_mod=10,
        atol=1e-4,
    )
