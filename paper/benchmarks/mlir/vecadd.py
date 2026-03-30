"""Vector addition C = A + B — @gpu_kernel DSL (all GPU backends).

Demonstrates the simplest element-wise kernel with pure Python syntax.
"""
import sys
import numpy as np
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python vecadd.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = 256


@gpu_kernel(grid=(N // WG,), block=(WG,))
def vecadd(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    gid = block_id.x * block_dim.x + thread_id.x
    C[gid] = A[gid] + B[gid]


from bench_utils import run_benchmark


if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + inputs[1]).astype(np.float32)

    run_benchmark(
        vecadd, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        atol=0,
    )
