# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 64
# REQUIRES: gpu
"""Puzzle 16 — Matrix Multiplication: C = A @ B.

Three implementations matching the Mojo GPU Puzzle:
  1. naive_matmul: each thread computes one C element via a loop (global memory only)
  2. smem_matmul: single-block matmul using shared memory
  3. tiled_matmul: multi-block tiled matmul with shared memory

Mojo equivalent (naive):
    row, col = thread_id.y, thread_id.x
    if row < size and col < size:
        for k in range(size): acc += a[row, k] * b[k, col]
        c[row, col] = acc

Mojo equivalent (tiled):
    for t in range(K // TILE):
        sA[ty, tx] = A[row, t*TILE + tx]
        sB[ty, tx] = B[t*TILE + ty, col]
        barrier()
        for kk in range(TILE): acc += sA[ty, kk] * sB[kk, tx]
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
# Small size for naive/smem sub-tests (must fit in a single block)
S = min(M, TILE)

# ===================================================================
# Sub-test 1: naive_matmul — no shared memory, each thread does a full
# dot product via global memory reads. Single block, small matrix.
# ===================================================================
A_layout_naive = Row(S, S)
B_layout_naive = Row(S, S)
C_layout_naive = Row(S, S)


@gpu_kernel(grid=(1,), block=(S, S))
def naive_matmul(A: Buffer(A_layout_naive, S, S),
                 B: Buffer(B_layout_naive, S, S),
                 C: Buffer(C_layout_naive, S, S)):
    row = thread_id.y
    col = thread_id.x
    acc = 0.0
    for k in range(S):
        acc += A[row, k] * B[k, col]
    C[row, col] = acc


# ===================================================================
# Sub-test 2: smem_matmul — single-block matmul with shared memory.
# Load A and B tiles into shared memory, barrier, then accumulate.
# ===================================================================
smem_layout_s = Row(S, S)


@gpu_kernel(grid=(1,), block=(S, S))
def smem_matmul(A: Buffer(A_layout_naive, S, S),
                B: Buffer(B_layout_naive, S, S),
                C: Buffer(C_layout_naive, S, S),
                sA: Shared(smem_layout_s, S, S),
                sB: Shared(smem_layout_s, S, S)):
    row = thread_id.y
    col = thread_id.x
    sA[row, col] = A[row, col]
    sB[row, col] = B[row, col]
    barrier()
    acc = 0.0
    for kk in range(S):
        acc += sA[row, kk] * sB[kk, col]
    C[row, col] = acc


# ===================================================================
# Sub-test 3: tiled_matmul — multi-block tiled matmul with shared memory.
# Each block computes a TILE x TILE tile of C.
# ===================================================================
A_layout_tiled = Row(M, K)
B_layout_tiled = Row(K, N)
smem_layout = Row(TILE, TILE)
C_tiled_layout = OrderBy(Row(M, N)).TileBy([M // TILE, N // TILE], [TILE, TILE])


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def tiled_matmul(A: Buffer(A_layout_tiled, M, K),
                 B: Buffer(B_layout_tiled, K, N),
                 C: Buffer(C_tiled_layout, M, N),
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

    def compute_expected_small(inputs):
        a = inputs[0].reshape(S, S)
        b = inputs[1].reshape(S, S)
        return (a @ b).ravel().astype(np.float32)

    def compute_expected_tiled(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel().astype(np.float32)

    print("Sub-test 1: naive matmul (global memory only)", file=sys.stderr)
    run_benchmark(
        naive_matmul, compute_expected_small,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"naive {S}x{S}",
        init_mod=10, atol=1e-2,
    )

    print("Sub-test 2: shared memory matmul (single block)", file=sys.stderr)
    run_benchmark(
        smem_matmul, compute_expected_small,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"smem {S}x{S}",
        init_mod=10, atol=1e-2,
    )

    print("Sub-test 3: tiled matmul (multi-block)", file=sys.stderr)
    run_benchmark(
        tiled_matmul, compute_expected_tiled,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"tiled {M}x{N}x{K}",
        init_mod=10, atol=1e-2,
    )
