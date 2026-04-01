"""Puzzle 33 — Tensor Core Operations: matrix multiply using MMA intrinsics.

NOTE: Tensor core ops (gpu.subgroup_mma_*) require specific hardware
(NVIDIA Volta+) and careful fragment type management. This puzzle
demonstrates the PATTERN — the tiled matmul that tensor cores accelerate.

The tensor core pattern:
  1. Load A tile fragment (16x16 from A)
  2. Load B tile fragment (16x16 from B)
  3. MMA: D = A * B + C   (done in one hardware instruction)
  4. Store D fragment to output

For now, this implements the equivalent computation using regular
arithmetic, structured exactly as tensor core code would be organized.
When MMA DSL builtins land, only the inner loop changes.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) == 2:
    M = N = K = int(sys.argv[1])
elif len(sys.argv) == 4:
    M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
else:
    print("Usage: python puzzle_33_tensor_cores.py N")
    sys.exit(1)

# MMA tile dimensions (16x16x16 is a common tensor core tile)
MMA_M = 16
MMA_N = 16
MMA_K = 16
assert M % MMA_M == 0 and N % MMA_N == 0 and K % MMA_K == 0

TILE = MMA_M  # = 16, matching tensor core tile

A_layout = Row(M, K)
B_layout = Row(K, N)
C_layout = OrderBy(Row(M, N)).TileBy([M // TILE, N // TILE], [TILE, TILE])
smem_layout = Row(TILE, TILE)


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def tensor_matmul(A: Buffer(A_layout, M, K), B: Buffer(B_layout, K, N),
                  C: Buffer(C_layout, M, N),
                  sA: Shared(smem_layout, TILE, TILE),
                  sB: Shared(smem_layout, TILE, TILE)):
    # This has the same structure as tensor core code:
    # outer loop tiles over K, inner loop is the MMA
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for t in range(K // TILE):
        # Load fragments into shared memory (= mma_load_a / mma_load_b)
        sA[thread_id.y, thread_id.x] = A[row, t * TILE + thread_id.x]
        sB[thread_id.y, thread_id.x] = B[t * TILE + thread_id.y, col]
        barrier()
        # MMA computation (= mma_compute)
        for kk in range(TILE):
            acc += sA[thread_id.y, kk] * sB[kk, thread_id.x]
        barrier()
    # Store result (= mma_store)
    C[block_id.y, block_id.x, thread_id.y, thread_id.x] = acc


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel().astype(np.float32)

    run_benchmark(
        tensor_matmul, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
        atol=1.0,
    )
