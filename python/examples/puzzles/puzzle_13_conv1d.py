# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 15 4
# REQUIRES: gpu
"""Puzzle 13 — 1D Convolution: convolve input with a small kernel.

output[i] = sum(a[i+k] * b[k] for k in range(CONV_SIZE)) where valid.

Uses shared memory to cache the input block.

Mojo equivalent:
    shared_a[i] = a[i]
    shared_b[i] = b[i]  (if i < CONV_SIZE)
    barrier()
    acc = 0.0
    for k in range(CONV_SIZE):
        if i + k < size:
            acc += shared_a[i + k] * shared_b[k]
    output[i] = acc
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 3:
    print("Usage: python puzzle_13_conv1d.py SIZE CONV_SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
CONV_SIZE = int(sys.argv[2])
# TPB must cover SIZE + CONV_SIZE - 1 for halo, but keep simple for single block
TPB = SIZE

input_layout = Row(TPB)
conv_layout = Row(CONV_SIZE)
out_layout = Row(SIZE)
smem_layout = Row(TPB)
smem_conv_layout = Row(CONV_SIZE)


@gpu_kernel(grid=(1,), block=(TPB,))
def conv_1d(a: Buffer(input_layout, TPB), b: Buffer(conv_layout, CONV_SIZE),
            out: Buffer(out_layout, SIZE),
            smem_a: Shared(smem_layout, TPB),
            smem_b: Shared(smem_conv_layout, CONV_SIZE)):
    tx = thread_id.x
    # Load input to shared
    smem_a[tx] = a[tx]
    if tx < CONV_SIZE:
        smem_b[tx] = b[tx]
    barrier()
    # Convolve (natural pattern — scf.if yields modified locals)
    acc = 0.0
    for k in range(CONV_SIZE):
        if tx + k < SIZE:
            acc += smem_a[tx + k] * smem_b[k]
    out[tx] = acc


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0][:SIZE]
        b = inputs[1][:CONV_SIZE]
        out = np.zeros(SIZE, dtype=np.float32)
        for i in range(SIZE):
            for k in range(CONV_SIZE):
                if i + k < SIZE:
                    out[i] += a[i + k] * b[k]
        return out

    run_benchmark(
        conv_1d, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"SIZE={SIZE},CONV={CONV_SIZE}",
        init_mod=10,
        atol=1e-4,
    )
