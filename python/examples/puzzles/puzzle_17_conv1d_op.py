"""Puzzle 17 — 1D Convolution Op: package convolution as a reusable kernel.

The Mojo puzzle packages a conv kernel as a MAX Graph custom op.
In LEGO, the kernel IS the reusable unit — compile once, dispatch anywhere.

Same convolution as Puzzle 13, but structured to demonstrate that LEGO
kernels compile to CUDA, Vulkan, WebGPU, Metal from a single source.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 3:
    print("Usage: python puzzle_17_conv1d_op.py SIZE CONV_SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
CONV_SIZE = int(sys.argv[2])
TPB = SIZE

input_layout = Row(TPB)
conv_layout = Row(CONV_SIZE)
out_layout = Row(SIZE)
smem_layout = Row(TPB)
smem_conv_layout = Row(CONV_SIZE)


@gpu_kernel(grid=(1,), block=(TPB,))
def conv1d_op(a: Buffer(input_layout, TPB), b: Buffer(conv_layout, CONV_SIZE),
              out: Buffer(out_layout, SIZE),
              smem_a: Shared(smem_layout, TPB),
              smem_b: Shared(smem_conv_layout, CONV_SIZE)):
    tx = thread_id.x
    smem_a[tx] = a[tx]
    if tx < CONV_SIZE:
        smem_b[tx] = b[tx]
    barrier()
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
        conv1d_op, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"SIZE={SIZE},CONV={CONV_SIZE}",
        init_mod=10,
        atol=1e-4,
    )
