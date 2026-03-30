"""End-to-end GPU verification of @gpu_kernel AST DSL — all four patterns."""
import sys
import numpy as np
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared
from bench_utils import run_benchmark

# ======== vecadd ========
N = 1024

@gpu_kernel(grid=(N // 256,), block=(256,))
def vecadd(A: Buffer[N], B: Buffer[N], C: Buffer[N]):
    gid = block_id.x * block_dim.x + thread_id.x
    C[gid] = A[gid] + B[gid]

print("=== vecadd (DSL) ===", file=sys.stderr)
run_benchmark(vecadd, lambda inp: (inp[0] + inp[1]).astype(np.float32),
              targets=["cuda"], label="N=1024")

# ======== matmul_naive ========
M_m = N_m = K_m = 64
TILE = 16

@gpu_kernel(grid=(N_m // TILE, M_m // TILE), block=(TILE, TILE))
def matmul_naive(A: Buffer[M_m, K_m], B: Buffer[K_m, N_m], C: Buffer[M_m, N_m]):
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for k in range(K_m):
        acc += A[row, k] * B[k, col]
    C[row, col] = acc

print("\n=== matmul_naive (DSL) ===", file=sys.stderr)
run_benchmark(matmul_naive,
              lambda inp: (inp[0].reshape(M_m, K_m) @ inp[1].reshape(K_m, N_m)).ravel(),
              targets=["cuda"], label=f"{M_m}x{N_m}x{K_m}", init_mod=10)

# ======== matmul_smem ========
@gpu_kernel(grid=(N_m // TILE, M_m // TILE), block=(TILE, TILE))
def matmul_smem(A: Buffer[M_m, K_m], B: Buffer[K_m, N_m], C: Buffer[M_m, N_m],
                sA: Shared[TILE, TILE], sB: Shared[TILE, TILE]):
    row = block_id.y * TILE + thread_id.y
    col = block_id.x * TILE + thread_id.x
    acc = 0.0
    for t in range(K_m // TILE):
        sA[thread_id.y, thread_id.x] = A[row, t * TILE + thread_id.x]
        sB[thread_id.y, thread_id.x] = B[t * TILE + thread_id.y, col]
        barrier()
        for kk in range(TILE):
            acc += sA[thread_id.y, kk] * sB[kk, thread_id.x]
        barrier()
    C[row, col] = acc

print("\n=== matmul_smem (DSL) ===", file=sys.stderr)
run_benchmark(matmul_smem,
              lambda inp: (inp[0].reshape(M_m, K_m) @ inp[1].reshape(K_m, N_m)).ravel(),
              targets=["cuda"], label=f"{M_m}x{N_m}x{K_m}", init_mod=10)

# ======== reduce_sum ========
N_r = 1024
BLOCK = 256

@gpu_kernel(grid=(N_r // BLOCK,), block=(BLOCK,))
def reduce_sum(A: Buffer[N_r], Out: Buffer[N_r // BLOCK], smem: Shared[BLOCK]):
    tx = thread_id.x
    bx = block_id.x
    gid = bx * BLOCK + tx
    smem[tx] = A[gid]
    barrier()
    stride = BLOCK // 2
    while stride > 0:
        if tx < stride:
            smem[tx] = smem[tx] + smem[tx + stride]
        barrier()
        stride = stride // 2
    if tx == 0:
        Out[bx] = smem[0]

print("\n=== reduce_sum (DSL) ===", file=sys.stderr)
run_benchmark(reduce_sum,
              lambda inp: inp[0].reshape(-1, BLOCK).sum(axis=1).astype(np.float32),
              targets=["cuda"], label=f"N={N_r}", init_mod=10)
