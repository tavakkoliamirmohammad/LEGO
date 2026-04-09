# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 12 — Dot Product: compute dot product via parallel tree reduction.

Each thread multiplies a[i]*b[i], then tree-reduce in shared memory.
Only thread 0 writes the final scalar result.

Mojo equivalent:
    shared[i] = a[i] * b[i]
    barrier()
    stride = TPB // 2
    while stride > 0:
        if i < stride:
            shared[i] += shared[i + stride]
        barrier()
        stride //= 2
    if i == 0:
        output[block_idx.x] = shared[0]
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_12_dot_product.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = min(256, N)
assert N % BLOCK == 0
num_blocks = N // BLOCK

A_layout = OrderBy(Row(N)).TileBy([num_blocks], [BLOCK])
out_layout = Row(num_blocks)
smem_layout = Row(BLOCK)


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def dot_product(A: Buffer(A_layout, N), B: Buffer(A_layout, N),
                Out: Buffer(out_layout, num_blocks),
                smem: Shared(smem_layout, BLOCK)):
    tx = thread_id.x
    bx = block_id.x
    # Multiply and load to shared
    smem[tx] = A[bx, tx] * B[bx, tx]
    barrier()
    # Tree reduction
    stride = BLOCK // 2
    while stride > 0:
        if tx < stride:
            smem[tx] = smem[tx] + smem[tx + stride]
        barrier()
        stride = stride // 2
    # Thread 0 writes result
    if tx == 0:
        Out[bx] = smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, BLOCK)
        b = inputs[1].reshape(-1, BLOCK)
        return (a * b).sum(axis=1).astype(np.float32)

    run_benchmark(
        dot_product, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-4,
    )
