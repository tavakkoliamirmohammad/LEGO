# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 8
# REQUIRES: nvidia-gpu
"""Puzzle 11 — Pooling: compute sliding window sum of last 3 elements.

output[i] = a[i] + a[i-1] + a[i-2]  (clamped at boundaries)

Uses shared memory to avoid redundant global reads.

Mojo equivalent:
    shared[i] = a[i]
    barrier()
    total = shared[i]
    if i >= 1: total += shared[i-1]
    if i >= 2: total += shared[i-2]
    output[i] = total
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_11_pooling.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
TPB = SIZE  # single block

layout = Row(SIZE)
smem_layout = Row(TPB)


@gpu_kernel(grid=(1,), block=(TPB,))
def pooling(A: Buffer(layout, SIZE), Out: Buffer(layout, SIZE),
            smem: Shared(smem_layout, TPB)):
    tx = thread_id.x
    # Load to shared
    smem[tx] = A[tx]
    barrier()
    # Sliding window sum (natural pattern — scf.if yields modified locals)
    total = smem[tx]
    if tx > 0:
        total = total + smem[tx - 1]
    if tx > 1:
        total = total + smem[tx - 2]
    Out[tx] = total


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0][:SIZE]
        out = np.zeros(SIZE, dtype=np.float32)
        for i in range(SIZE):
            out[i] = a[i]
            if i >= 1:
                out[i] += a[i - 1]
            if i >= 2:
                out[i] += a[i - 2]
        return out

    run_benchmark(
        pooling, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"SIZE={SIZE}",
        init_mod=10,
    )
