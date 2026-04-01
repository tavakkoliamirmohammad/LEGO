"""Puzzle 34 — GPU Cluster Programming: coordinate computation across SMs.

NOTE: GPU cluster operations (gpu.cluster_id, gpu.cluster_dim) require
SM90+ (NVIDIA Hopper) hardware. This puzzle demonstrates the PATTERN
of multi-block coordination — multiple blocks cooperating on a shared
reduction.

Pattern: each block reduces its local tile, then block 0 collects partial
sums from all blocks. In a real cluster, distributed shared memory would
allow direct cross-block communication.

For now, this implements multi-block reduction where each block writes
its partial sum, demonstrating the coordination pattern.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_34_clusters.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = min(256, N)
assert N % BLOCK == 0
num_blocks = N // BLOCK

layout = OrderBy(Row(N)).TileBy([num_blocks], [BLOCK])
partial_layout = Row(num_blocks)
smem_layout = Row(BLOCK)


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def cluster_reduce(A: Buffer(layout, N),
                   Partials: Buffer(partial_layout, num_blocks),
                   smem: Shared(smem_layout, BLOCK)):
    bx = block_id.x
    tx = thread_id.x
    # Phase 1: Each block reduces its tile (local reduction)
    smem[tx] = A[bx, tx]
    barrier()
    stride = BLOCK // 2
    while stride > 0:
        if tx < stride:
            smem[tx] = smem[tx] + smem[tx + stride]
        barrier()
        stride = stride // 2
    # Phase 2: Block 0 thread 0 writes partial sum
    # In a cluster, this would use distributed shared memory
    if tx == 0:
        Partials[bx] = smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return inputs[0].reshape(-1, BLOCK).sum(axis=1).astype(np.float32)

    run_benchmark(
        cluster_reduce, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1.0,
    )
