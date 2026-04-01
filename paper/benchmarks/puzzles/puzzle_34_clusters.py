"""Puzzle 34 — GPU Cluster Programming: multi-block scaled reduction.

Mojo puzzle: each block scales input by (block_id + 1), then sums.
Cluster coordination (cluster_arrive/wait) is used for inter-block sync.
We implement the per-block computation faithfully; cluster sync is
replaced by independent block execution (same result since blocks
don't exchange data in the first sub-test).

  output[block_id] = sum(input[block_start..block_end] * (block_id + 1))
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
def cluster_scaled_reduce(A: Buffer(layout, N),
                          Out: Buffer(partial_layout, num_blocks),
                          smem: Shared(smem_layout, BLOCK)):
    bx = block_id.x
    tx = thread_id.x
    # Scale by (block_id + 1) — matches Mojo's cluster_coordination_basics
    smem[tx] = A[bx, tx] * (bx + 1)
    barrier()
    # Thread 0 sums sequentially (matches Mojo's sequential accumulation)
    if tx == 0:
        block_sum = smem[0]
        i = 1
        while i < BLOCK:
            block_sum = block_sum + smem[i]
            i = i + 1
        Out[bx] = block_sum


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, BLOCK)
        out = np.zeros(num_blocks, dtype=np.float32)
        for b in range(num_blocks):
            out[b] = (a[b] * (b + 1)).sum()
        return out.astype(np.float32)

    run_benchmark(
        cluster_scaled_reduce, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-2,
    )
