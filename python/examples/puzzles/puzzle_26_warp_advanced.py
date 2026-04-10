# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 26 — Advanced Warp Patterns: XOR shuffle, butterfly, prefix sum.

Five sub-tests matching the Mojo puzzle:
  1. pair_swap: shuffle_xor(val, 1) — swap adjacent pairs
  2. butterfly_max: XOR tree reduction for broadcast max
  3. butterfly_sum: XOR tree reduction for broadcast sum
  4. warp_prefix_sum: inclusive prefix sum within warp
  5. warp_partition: quicksort-style partition using prefix_sum + shuffle_xor
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_26_warp_advanced.py N")
    sys.exit(1)

N = int(sys.argv[1])
WARP_SIZE = 32
assert N % WARP_SIZE == 0
num_warps = N // WARP_SIZE

layout = OrderBy(Row(N)).TileBy([num_warps], [WARP_SIZE])

# --- Sub-test 1: pair_swap ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def pair_swap(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    Out[bx, tx] = shuffle_xor(A[bx, tx], 1)

# --- Sub-test 2: butterfly_max ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def butterfly_max(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    max_val = A[bx, tx]
    offset = WARP_SIZE // 2
    while offset > 0:
        other = shuffle_xor(max_val, offset)
        if other > max_val:
            max_val = other
        offset = offset // 2
    Out[bx, tx] = max_val

# --- Sub-test 3: butterfly_sum ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def butterfly_sum(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    mask = 1
    while mask < WARP_SIZE:
        other = shuffle_xor(val, mask)
        val = val + other
        mask = mask * 2
    Out[bx, tx] = val

# --- Sub-test 4: warp_prefix_sum (inclusive) ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def warp_prefix(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    Out[bx, tx] = warp_prefix_sum(val)

# --- Sub-test 5: warp_partition ---
# Quicksort-style partition: elements < pivot go left, >= pivot go right.
# Uses exclusive prefix sum to compute write positions, butterfly xor
# to get total count.
PIVOT = 5.0


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def warp_partition(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    # Phase 1: predicates — use arithmetic instead of if to avoid scf.if yield issues
    # is_left = 1.0 if val < PIVOT else 0.0 (computed via comparison + select)
    # Since DSL doesn't have ternary, use: store predicate in output buffer
    # then compute prefix sums on it.
    #
    # Simplified: compute both prefix sums and pick the right one at the end.
    # is_left = 1.0 when val < PIVOT, 0.0 otherwise
    # We compute this as: is_left = max(0, sign(PIVOT - val)) but that's complex.
    # Instead, use two separate prefix sums on complementary predicates.
    #
    # Write predicate to output first, read it back for prefix sum
    Out[bx, tx] = 0.0  # default: right
    if val < PIVOT:
        Out[bx, tx] = 1.0  # left
    is_left = Out[bx, tx]
    is_right = 1.0 - is_left
    # Phase 2: exclusive prefix sums for write positions
    left_pos = warp_prefix_sum_exclusive(is_left)
    right_pos = warp_prefix_sum_exclusive(is_right)
    # Phase 3: total left count via butterfly reduction
    left_total = is_left
    mask = 1
    while mask < WARP_SIZE:
        left_total = left_total + shuffle_xor(left_total, mask)
        mask = mask * 2
    # Phase 4: write position indices to output
    if val < PIVOT:
        Out[bx, tx] = left_pos
    if val >= PIVOT:
        Out[bx, tx] = left_total + right_pos


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected_swap(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            for i in range(0, WARP_SIZE, 2):
                out[w, i] = a[w, i + 1]
                out[w, i + 1] = a[w, i]
        return out.ravel().astype(np.float32)

    print("Sub-test 1: pair_swap", file=sys.stderr)
    run_benchmark(
        pair_swap, compute_expected_swap,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"swap N={N}",
    )

    def compute_expected_max(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        maxes = a.max(axis=1, keepdims=True)
        return np.broadcast_to(maxes, a.shape).ravel().astype(np.float32)

    print("Sub-test 2: butterfly_max", file=sys.stderr)
    run_benchmark(
        butterfly_max, compute_expected_max,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"max N={N}", atol=1e-4,
    )

    def compute_expected_sum(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        sums = a.sum(axis=1, keepdims=True)
        return np.broadcast_to(sums, a.shape).ravel().astype(np.float32)

    print("Sub-test 3: butterfly_sum", file=sys.stderr)
    run_benchmark(
        butterfly_sum, compute_expected_sum,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"sum N={N}", init_mod=10, atol=1e-4,
    )

    def compute_expected_prefix(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.cumsum(a, axis=1)
        return out.ravel().astype(np.float32)

    print("Sub-test 4: warp_prefix_sum", file=sys.stderr)
    run_benchmark(
        warp_prefix, compute_expected_prefix,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"prefix N={N}", init_mod=10, atol=1e-4,
    )

    # Sub-test 5: partition — verify partition property (not exact positions)
    # The kernel writes positions (float indices), we verify the partition
    # property: all left positions < left_count, all right positions >= left_count
    def compute_expected_partition(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            left_count = np.sum(a[w] < PIVOT)
            left_idx = 0
            right_idx = 0
            for i in range(WARP_SIZE):
                if a[w, i] < PIVOT:
                    out[w, i] = float(left_idx)
                    left_idx += 1
                else:
                    out[w, i] = float(left_count + right_idx)
                    right_idx += 1
        return out.ravel().astype(np.float32)

    print("Sub-test 5: warp_partition", file=sys.stderr)
    run_benchmark(
        warp_partition, compute_expected_partition,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"partition N={N}", init_mod=10, atol=1e-4,
    )
