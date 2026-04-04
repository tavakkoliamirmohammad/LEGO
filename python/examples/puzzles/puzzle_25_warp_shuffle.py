# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 25 — Warp Communication: all 5 Mojo sub-tests.

  1. neighbor_difference: output[i] = input[i+1] - input[i]
  2. moving_average_3: 3-point moving average via shuffle_down
  3. basic_broadcast: lane 0 sums first 4 elems, broadcasts, each adds own
  4. conditional_broadcast: lane 0 finds max of first 8, broadcasts threshold
  5. broadcast_shuffle_coordination: broadcast scale + pairwise sum
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_25_warp_shuffle.py N")
    sys.exit(1)

N = int(sys.argv[1])
WARP_SIZE = 32
assert N % WARP_SIZE == 0
num_warps = N // WARP_SIZE

layout = OrderBy(Row(N)).TileBy([num_warps], [WARP_SIZE])

# --- Sub-test 1: neighbor_difference ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def neighbor_difference(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    next_val = shuffle_down(val, 1)
    if tx < WARP_SIZE - 1:
        Out[bx, tx] = next_val - val

# --- Sub-test 2: moving_average_3 ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def moving_average_3(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    next1 = shuffle_down(val, 1)
    next2 = shuffle_down(val, 2)
    if tx < WARP_SIZE - 2:
        Out[bx, tx] = (val + next1 + next2) / 3.0
    if tx == WARP_SIZE - 2:
        Out[bx, tx] = (val + next1) / 2.0
    if tx == WARP_SIZE - 1:
        Out[bx, tx] = val

# --- Sub-test 3: basic_broadcast ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def basic_broadcast(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    bcast_val = 0.0
    if tx == 0:
        bcast_val = A[bx, 0] + A[bx, 1] + A[bx, 2] + A[bx, 3]
    bcast_val = broadcast(bcast_val)
    Out[bx, tx] = bcast_val + val

# --- Sub-test 4: conditional_broadcast ---
# Lane 0 finds max of first 8 elements, broadcasts; threshold = max/2
# If val >= threshold: double it, else halve it


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def conditional_broadcast(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    decision = 0.0
    if tx == 0:
        decision = A[bx, 0]
        if A[bx, 1] > decision:
            decision = A[bx, 1]
        if A[bx, 2] > decision:
            decision = A[bx, 2]
        if A[bx, 3] > decision:
            decision = A[bx, 3]
        if A[bx, 4] > decision:
            decision = A[bx, 4]
        if A[bx, 5] > decision:
            decision = A[bx, 5]
        if A[bx, 6] > decision:
            decision = A[bx, 6]
        if A[bx, 7] > decision:
            decision = A[bx, 7]
    decision = broadcast(decision)
    threshold = decision / 2.0
    if val >= threshold:
        Out[bx, tx] = val * 2.0
    if val < threshold:
        Out[bx, tx] = val / 2.0

# --- Sub-test 5: broadcast_shuffle_coordination ---


@gpu_kernel(grid=(num_warps,), block=(WARP_SIZE,))
def broadcast_shuffle_coord(A: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    val = A[bx, tx]
    scale = 0.0
    if tx == 0:
        scale = (A[bx, 0] + A[bx, 1] + A[bx, 2] + A[bx, 3]) / 4.0
    scale = broadcast(scale)
    next_val = shuffle_down(val, 1)
    if tx < WARP_SIZE - 1:
        Out[bx, tx] = (val + next_val) * scale
    if tx == WARP_SIZE - 1:
        Out[bx, tx] = val * scale


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected_diff(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        out[:, :-1] = np.diff(a, axis=1)
        return out.ravel().astype(np.float32)

    print("Sub-test 1: neighbor_difference", file=sys.stderr)
    run_benchmark(neighbor_difference, compute_expected_diff,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"neighbor N={N}", init_mod=100, atol=1e-4)

    def compute_expected_avg(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            for i in range(WARP_SIZE):
                if i < WARP_SIZE - 2:
                    out[w, i] = (a[w, i] + a[w, i+1] + a[w, i+2]) / 3.0
                elif i == WARP_SIZE - 2:
                    out[w, i] = (a[w, i] + a[w, i+1]) / 2.0
                else:
                    out[w, i] = a[w, i]
        return out.ravel().astype(np.float32)

    print("Sub-test 2: moving_average_3", file=sys.stderr)
    run_benchmark(moving_average_3, compute_expected_avg,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"avg3 N={N}", init_mod=100, atol=1e-4)

    def compute_expected_broadcast(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            out[w] = a[w, :4].sum() + a[w]
        return out.ravel().astype(np.float32)

    print("Sub-test 3: basic_broadcast", file=sys.stderr)
    run_benchmark(basic_broadcast, compute_expected_broadcast,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"broadcast N={N}", init_mod=10, atol=1e-4)

    def compute_expected_conditional(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            max8 = a[w, :8].max()
            threshold = max8 / 2.0
            for i in range(WARP_SIZE):
                if a[w, i] >= threshold:
                    out[w, i] = a[w, i] * 2.0
                else:
                    out[w, i] = a[w, i] / 2.0
        return out.ravel().astype(np.float32)

    print("Sub-test 4: conditional_broadcast", file=sys.stderr)
    run_benchmark(conditional_broadcast, compute_expected_conditional,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"cond N={N}", init_mod=10, atol=1e-4)

    def compute_expected_coord(inputs):
        a = inputs[0].reshape(-1, WARP_SIZE)
        out = np.zeros_like(a)
        for w in range(a.shape[0]):
            scale = a[w, :4].mean()
            for i in range(WARP_SIZE):
                if i < WARP_SIZE - 1:
                    out[w, i] = (a[w, i] + a[w, i+1]) * scale
                else:
                    out[w, i] = a[w, i] * scale
        return out.ravel().astype(np.float32)

    print("Sub-test 5: broadcast_shuffle_coordination", file=sys.stderr)
    run_benchmark(broadcast_shuffle_coord, compute_expected_coord,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"coord N={N}", init_mod=10, atol=1e-4)
