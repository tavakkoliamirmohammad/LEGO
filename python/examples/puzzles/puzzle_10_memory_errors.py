# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 3
# REQUIRES: gpu
"""Puzzle 10 — Memory Error Detection: guard patterns preventing OOB.

Two sub-tests from the Mojo puzzle:
  1. add_10_2d: 2D add-10 with bounds guard (TPB=3x3, SIZE=2x2)
  2. shared_memory_race: Sum all elements, broadcast to output (thread 0 only)

Mojo uses SIZE=2, TPB=3. We match that exactly.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_10_memory_errors.py SIZE")
    sys.exit(1)

SIZE = int(sys.argv[1])
TPB = SIZE + 1  # Mojo uses TPB=3 for SIZE=2

layout = Row(TPB, TPB)
smem_layout = Row(1)

# --- Sub-test 1: Guarded 2D add-10 (match Mojo exactly: +10.0) ---


@gpu_kernel(grid=(1,), block=(TPB, TPB))
def add_10_2d(A: Buffer(layout, TPB, TPB), Out: Buffer(layout, TPB, TPB)):
    row = thread_id.y
    col = thread_id.x
    if row < SIZE:
        if col < SIZE:
            Out[row, col] = A[row, col] + 10.0


# --- Sub-test 2: Race-free sum + broadcast (thread 0 accumulates) ---
# NOTE: LEGO DSL does not support nested for-loops over compile-time ranges
# accessing 2D buffers with runtime indices easily, so we implement the
# single-block sum using shared memory tree reduction instead.
# The key lesson is the same: avoid concurrent writes to shared memory.

out_layout_2 = Row(TPB, TPB)
smem_layout_2 = Row(1)


@gpu_kernel(grid=(1,), block=(TPB, TPB))
def shared_memory_race_fix(A: Buffer(layout, TPB, TPB),
                           Out: Buffer(out_layout_2, TPB, TPB),
                           smem: Shared(smem_layout_2, 1)):
    row = thread_id.y
    col = thread_id.x
    # Thread (0,0) accumulates sequentially from A (same as Mojo solution)
    if row == 0:
        if col == 0:
            total = 0.0
            for r in range(SIZE):
                for c in range(SIZE):
                    total += A[r, c]
            smem[0] = total
    barrier()
    # Broadcast sum to all valid output positions
    if row < SIZE:
        if col < SIZE:
            Out[row, col] = smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    # Sub-test 1: add_10_2d
    def compute_expected_add10(inputs):
        a = inputs[0].reshape(TPB, TPB)
        out = np.zeros((TPB, TPB), dtype=np.float32)
        out[:SIZE, :SIZE] = a[:SIZE, :SIZE] + 10.0
        return out.ravel()

    print("Sub-test 1: add_10_2d (guarded +10)", file=sys.stderr)
    run_benchmark(
        add_10_2d, compute_expected_add10,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"add10 SIZE={SIZE}",
    )

    # Sub-test 2: shared_memory_race_fix (sum + broadcast)
    def compute_expected_race(inputs):
        a = inputs[0].reshape(TPB, TPB)
        total = a[:SIZE, :SIZE].sum()
        out = np.zeros((TPB * TPB,), dtype=np.float32)
        for r in range(SIZE):
            for c in range(SIZE):
                out[r * TPB + c] = total
        return out.astype(np.float32)

    print("Sub-test 2: shared_memory_race_fix (sum+broadcast)", file=sys.stderr)
    run_benchmark(
        shared_memory_race_fix, compute_expected_race,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"race SIZE={SIZE}",
        atol=1e-4,
    )
