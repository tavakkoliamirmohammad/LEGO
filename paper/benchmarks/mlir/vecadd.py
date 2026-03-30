"""Vector addition C = A + B — unified KernelBuilder API (all GPU backends).

Demonstrates the simplest element-wise kernel with layout-aware buffers.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.compiler import DType
from lego.backend.gpu_builder import KernelBuilder, LayoutBuffer

if len(sys.argv) != 2:
    print("Usage: python vecadd.py N")
    sys.exit(1)

N = int(sys.argv[1])

# --- Buffers: 1D with Row(N) layout ---
A = LayoutBuffer(Row(N), shape=(N,), dtype=DType.f32)
B = LayoutBuffer(Row(N), shape=(N,), dtype=DType.f32)
C = LayoutBuffer(Row(N), shape=(N,), dtype=DType.f32)


def vecadd_kernel(ctx):
    gid = KernelBuilder.global_id_1d(ctx)
    va = ctx.load_flat(0, gid)
    vb = ctx.load_flat(1, gid)
    ctx.store_flat(ctx.addf(va, vb), 2, gid)


builder = KernelBuilder(
    buffers=[A, B, C],
    kernel_body=vecadd_kernel,
    name="vecadd",
    workgroup_size=256,
)


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + inputs[1]).astype(np.float32)

    run_benchmark(
        builder, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        atol=0,
    )
