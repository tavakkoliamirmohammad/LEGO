# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 23 — Functional Patterns: elementwise, tiled, and vectorized ops.

Demonstrates three GPU programming patterns using LEGO layouts:

1. Elementwise: each thread processes one element
2. Tiled: threads cooperate on blocks of data via shared memory
3. Vectorized: each thread processes multiple elements (simulated via loop)

In LEGO, these patterns differ only in the layout and loop structure —
the layout algebra handles the index math.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_23_functional.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0
num_blocks = N // WG

# --- Pattern 1: Elementwise (each thread = one element) ---
elem_layout = OrderBy(Row(N)).TileBy([num_blocks], [WG])


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def elementwise_add(A: Buffer(elem_layout, N), B: Buffer(elem_layout, N),
                    Out: Buffer(elem_layout, N)):
    bx = block_id.x
    tx = thread_id.x
    Out[bx, tx] = A[bx, tx] + B[bx, tx]


# --- Pattern 2: Tiled (shared memory cooperation) ---
TILE = WG
smem_layout = Row(TILE)


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def tiled_add(A: Buffer(elem_layout, N), B: Buffer(elem_layout, N),
              Out: Buffer(elem_layout, N),
              sA: Shared(smem_layout, TILE), sB: Shared(smem_layout, TILE)):
    bx = block_id.x
    tx = thread_id.x
    # Load into shared memory
    sA[tx] = A[bx, tx]
    sB[tx] = B[bx, tx]
    barrier()
    # Compute from shared
    Out[bx, tx] = sA[tx] + sB[tx]


# --- Pattern 3: Vectorized (each thread processes VEC elements) ---
# 3-level TileBy: [num_blocks], [VWG], [VEC]
VEC = 4
VWG = WG // VEC  # fewer threads, each handles VEC elements
assert WG % VEC == 0
vec_layout = OrderBy(Row(N)).TileBy([num_blocks], [VWG], [VEC])


@gpu_kernel(grid=(num_blocks,), block=(VWG,))
def vectorized_add(A: Buffer(vec_layout, N), B: Buffer(vec_layout, N),
                   Out: Buffer(vec_layout, N)):
    bx = block_id.x
    tx = thread_id.x
    for v in range(VEC):
        Out[bx, tx, v] = A[bx, tx, v] + B[bx, tx, v]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return (inputs[0] + inputs[1]).astype(np.float32)

    for name, kernel in [("elementwise", elementwise_add),
                          ("tiled", tiled_add),
                          ("vectorized", vectorized_add)]:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Pattern: {name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        run_benchmark(
            kernel, compute_expected,
            targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
            label=f"{name} N={N}",
        )
