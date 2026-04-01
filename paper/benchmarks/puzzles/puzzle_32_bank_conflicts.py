"""Puzzle 32 — Bank Conflicts: avoid shared memory bank conflicts via padding.

Shared memory is organized in 32 banks. When multiple threads access the
same bank simultaneously (but different addresses), a bank conflict occurs.

Fix: pad each row by 1 element so consecutive rows map to different banks.

  Without padding: Row(TILE, TILE)     — column accesses hit same bank
  With padding:    Row(TILE, TILE + 1) — offset breaks the conflict pattern

LEGO advantage: the padding is expressed in the layout itself. No manual
stride tricks needed — Row(TILE, TILE+1) naturally offsets each row.

This puzzle demonstrates a matrix transpose where column-major reads
benefit from padding to avoid bank conflicts.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_32_bank_conflicts.py N")
    sys.exit(1)

N = int(sys.argv[1])
TILE = 32
assert N % TILE == 0
num_tiles = N // TILE

# --- Global layouts ---
A_layout = OrderBy(Row(N, N)).TileBy([num_tiles, num_tiles], [TILE, TILE])

# --- Shared memory with padding: TILE rows x (TILE+1) cols ---
# The +1 padding breaks bank conflict patterns on column access.
smem_layout = Row(TILE, TILE + 1)

C_layout = OrderBy(Row(N, N)).TileBy([num_tiles, num_tiles], [TILE, TILE])


@gpu_kernel(grid=(num_tiles, num_tiles), block=(TILE, TILE))
def transpose_padded(A: Buffer(A_layout, N, N), B: Buffer(C_layout, N, N),
                     smem: Shared(smem_layout, TILE, TILE + 1)):
    by = block_id.y
    bx = block_id.x
    ty = thread_id.y
    tx = thread_id.x
    # Load tile into padded shared memory (coalesced global read)
    smem[ty, tx] = A[by, bx, ty, tx]
    barrier()
    # Write transposed tile from shared memory (padded layout avoids bank conflicts)
    B[bx, by, ty, tx] = smem[tx, ty]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return inputs[0].reshape(N, N).T.ravel().astype(np.float32)

    run_benchmark(
        transpose_padded, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{N}x{N}",
    )
