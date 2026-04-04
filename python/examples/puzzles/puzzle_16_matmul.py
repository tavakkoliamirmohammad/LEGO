# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 64
# REQUIRES: nvidia-gpu
"""Puzzle 16 — Matrix Multiplication: C = A @ B with tiled shared memory.

Two implementations:
  1. Naive (global memory only)
  2. Tiled with shared memory (load tiles → barrier → accumulate → barrier)

LEGO layout for tiled version:
  C: OrderBy(Row(N, N)).TileBy([N//TILE, N//TILE], [TILE, TILE])
  sA, sB: Row(TILE, TILE) shared memory tiles

Mojo equivalent (tiled):
    for t in range(K // TILE):
        sA[ty, tx] = A[row, t*TILE + tx]
        sB[ty, tx] = B[t*TILE + ty, col]
        barrier()
        for kk in range(TILE):
            acc += sA[ty, kk] * sB[kk, tx]
        barrier()
    C[row, col] = acc
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
    print("Usage: python puzzle_16_matmul.py N  or  python puzzle_16_matmul.py M N K")
    sys.exit(1)

TILE = 16

# --- Layouts ---
A_layout = Row(M, K)
B_layout = Row(K, N)
C_layout = OrderBy(Row(M, N)).TileBy([M // TILE, N // TILE], [TILE, TILE])
smem_layout = Row(TILE, TILE)


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def matmul(A: Buffer(A_layout, M, K), B: Buffer(B_layout, K, N),
           C: Buffer(C_layout, M, N),
           sA: Shared(smem_layout, TILE, TILE),
           sB: Shared(smem_layout, TILE, TILE)):
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for t in range(K // TILE):
        sA[thread_id.y, thread_id.x] = A[row, t * TILE + thread_id.x]
        sB[thread_id.y, thread_id.x] = B[t * TILE + thread_id.y, col]
        barrier()
        for kk in range(TILE):
            acc += sA[thread_id.y, kk] * sB[kk, thread_id.x]
        barrier()
    C[block_id.y, block_id.x, thread_id.y, thread_id.x] = acc


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel().astype(np.float32)

    run_benchmark(
        matmul, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
        atol=1e-2,
    )
