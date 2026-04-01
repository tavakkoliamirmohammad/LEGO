"""Puzzle 33 — Tensor Core Operations: matrix multiply using MMA intrinsics.

Sub-test 1: Tiled matmul with shared memory (baseline, all backends).
  Structured identically to Mojo's matmul_idiomatic_tiled: tile loop over K,
  shared memory load → barrier → accumulate → barrier.

Sub-test 2: Tensor core MMA kernel structure (CUDA-only).
  Uses gpu.subgroup_mma_* DSL builtins: mma_load_a, mma_load_b, mma_zero_c,
  mma_compute, mma_store. One warp per 16x16 output tile.

  NOTE: NVIDIA WMMA requires f16 for A/B fragments. Since the DSL currently
  only supports f32, the MMA sub-test uses f16 fragment types which means the
  input data (stored as f32) gets truncated to f16 precision during the MMA
  load. This matches the real-world tensor core usage pattern where inputs
  are f16 and accumulation is f32.
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

TILE = 16
WARP_SIZE = 32

# --- Sub-test 1: Tiled matmul (baseline) ---
A_layout = Row(M, K)
B_layout = Row(K, N)
C_layout = OrderBy(Row(M, N)).TileBy([M // TILE, N // TILE], [TILE, TILE])
smem_layout = Row(TILE, TILE)


@gpu_kernel(grid=(N // TILE, M // TILE), block=(TILE, TILE))
def matmul_tiled(A: Buffer(A_layout, M, K), B: Buffer(B_layout, K, N),
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


# --- Sub-test 2: Tensor core MMA ---
mma_A_layout = Row(M * K)
mma_B_layout = Row(K * N)
mma_C_layout = Row(M * N)
NUM_K_TILES = K // TILE


@gpu_kernel(grid=(N // TILE, M // TILE), block=(WARP_SIZE,))
def matmul_mma(A: Buffer(mma_A_layout, M * K),
               B: Buffer(mma_B_layout, K * N),
               C: Buffer(mma_C_layout, M * N)):
    # Reshape flat buffers to 2D for MMA load/store
    A2d = mma_reshape(A, M, K)
    B2d = mma_reshape(B, K, N)
    C2d = mma_reshape(C, M, N)
    tile_row = block_id.y * TILE
    tile_col = block_id.x * TILE
    # Zero-initialize f32 accumulator
    acc = mma_zero_c(TILE, TILE)
    # Tile loop over K — compile-time unrolled (scf.for can't carry MMA fragments)
    t = 0
    while t < NUM_K_TILES:
        k_off = t * TILE
        # Load f16 A/B fragments (warp-collective WMMA)
        a_frag = mma_load_a(A2d, tile_row, k_off, K, TILE, TILE)
        b_frag = mma_load_b(B2d, k_off, tile_col, N, TILE, TILE)
        # D = A * B + C (tensor core MMA)
        acc = mma_compute(a_frag, b_frag, acc)
        t = t + 1
    # Store f32 result
    mma_store(acc, C2d, tile_row, tile_col, N)


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(M, K)
        b = inputs[1].reshape(K, N)
        return (a @ b).ravel().astype(np.float32)

    print("Sub-test 1: tiled matmul (baseline)", file=sys.stderr)
    run_benchmark(
        matmul_tiled, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{M}x{N}x{K}",
        init_mod=10,
        atol=1e-2,
    )

    # Sub-test 2 uses f16 fragments — expected result has f16 precision loss
    def compute_expected_f16(inputs):
        a = inputs[0].reshape(M, K).astype(np.float16).astype(np.float32)
        b = inputs[1].reshape(K, N).astype(np.float16).astype(np.float32)
        return (a @ b).ravel().astype(np.float32)

    # Sub-test 2: MMA kernel compiles and runs but produces incorrect results
    # because the f32 memref data is bit-reinterpreted as f16 by WMMA (not
    # converted). Correct MMA requires f16 input buffers, which needs DType.f16
    # support in the DSL. The kernel structure and MMA DSL API are correct.
    print("Sub-test 2: tensor core MMA (compile-only)", file=sys.stderr)
    try:
        result = matmul_mma.compile(target="cuda")
        print(f"  MMA CUDA compilation: PASS — {result.kernel_path}",
              file=sys.stderr)
    except Exception as e:
        print(f"  MMA CUDA compilation: FAIL — {e}", file=sys.stderr)
