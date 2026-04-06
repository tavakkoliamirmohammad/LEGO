# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 1024
# REQUIRES: gpu
"""Puzzle 29 — GPU Synchronization: multi-stage pipeline + double-buffered stencil.

Sub-test 1 (Mojo 29A): 3-stage image processing pipeline
  Stage 1 (threads 0-127): Preprocess: P[i] = I[i] * 1.1
  Stage 2 (threads 128-255): Blur: B[i] = avg(P[i-2..i+2])
  Stage 3 (all threads): Smooth: F[i] = neighbor weighted average * 0.6

Sub-test 2 (Mojo 29B): Double-buffered iterative 3-point stencil
  3 iterations of: S[j] = avg(S[j-1], S[j], S[j+1])
  Uses barrier() as portable synchronization (Mojo uses mbarrier).
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_29_pipeline.py N")
    sys.exit(1)

N = int(sys.argv[1])
BLOCK = 256
assert N % BLOCK == 0
num_blocks = N // BLOCK

layout = OrderBy(Row(N)).TileBy([num_blocks], [BLOCK])
smem_a_layout = Row(BLOCK)
smem_b_layout = Row(BLOCK)

# --- Sub-test 1: 3-stage pipeline ---


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def pipeline_3stage(A: Buffer(layout, N), Out: Buffer(layout, N),
                    sA: Shared(smem_a_layout, BLOCK),
                    sB: Shared(smem_b_layout, BLOCK)):
    bx = block_id.x
    tx = thread_id.x
    # Stage 1: threads 0-127 preprocess (P[i] = I[i] * 1.1), 2 elems each
    if tx < 128:
        sA[tx * 2] = A[bx, tx * 2] * 1.1
        sA[tx * 2 + 1] = A[bx, tx * 2 + 1] * 1.1
    barrier()
    # Stage 2: threads 128-255 blur (radius 2), 2 elems each
    if tx >= 128:
        idx_base = (tx - 128) * 2
        for elem in range(2):
            idx = idx_base + elem
            total = sA[idx]
            count = 1.0
            if idx > 0:
                total = total + sA[idx - 1]
                count = count + 1.0
            if idx > 1:
                total = total + sA[idx - 2]
                count = count + 1.0
            if idx < BLOCK - 1:
                total = total + sA[idx + 1]
                count = count + 1.0
            if idx < BLOCK - 2:
                total = total + sA[idx + 2]
                count = count + 1.0
            sB[idx] = total * (1.0 / count)
    barrier()
    # Stage 3: all threads smooth (cascading neighbor average * 0.6)
    if tx == 0:
        Out[bx, tx] = (sB[0] + sB[1]) * 0.6
    if tx == BLOCK - 1:
        Out[bx, tx] = (sB[BLOCK - 1] + sB[BLOCK - 2]) * 0.6
    if tx > 0:
        if tx < BLOCK - 1:
            Out[bx, tx] = ((sB[tx] + sB[tx - 1]) * 0.6 + sB[tx + 1]) * 0.6


# --- Sub-test 2: Double-buffered 3-point stencil (3 iterations) ---
# Uses barrier() as portable mbarrier substitute.


@gpu_kernel(grid=(num_blocks,), block=(BLOCK,))
def stencil_double_buf(A: Buffer(layout, N), Out: Buffer(layout, N),
                       sA: Shared(smem_a_layout, BLOCK),
                       sB: Shared(smem_b_layout, BLOCK)):
    bx = block_id.x
    tx = thread_id.x
    # Load input into buffer A
    sA[tx] = A[bx, tx]
    barrier()
    # Iteration 1: read A, write B
    if tx == 0:
        sB[tx] = (sA[0] + sA[1]) * 0.5
    if tx == BLOCK - 1:
        sB[tx] = (sA[BLOCK - 2] + sA[BLOCK - 1]) * 0.5
    if tx > 0:
        if tx < BLOCK - 1:
            sB[tx] = (sA[tx - 1] + sA[tx] + sA[tx + 1]) * 0.333333343267441
    barrier()
    # Iteration 2: read B, write A
    if tx == 0:
        sA[tx] = (sB[0] + sB[1]) * 0.5
    if tx == BLOCK - 1:
        sA[tx] = (sB[BLOCK - 2] + sB[BLOCK - 1]) * 0.5
    if tx > 0:
        if tx < BLOCK - 1:
            sA[tx] = (sB[tx - 1] + sB[tx] + sB[tx + 1]) * 0.333333343267441
    barrier()
    # Iteration 3: read A, write B
    if tx == 0:
        sB[tx] = (sA[0] + sA[1]) * 0.5
    if tx == BLOCK - 1:
        sB[tx] = (sA[BLOCK - 2] + sA[BLOCK - 1]) * 0.5
    if tx > 0:
        if tx < BLOCK - 1:
            sB[tx] = (sA[tx - 1] + sA[tx] + sA[tx + 1]) * 0.333333343267441
    barrier()
    # Final output from buffer B
    Out[bx, tx] = sB[tx]


from bench_utils import run_benchmark

if __name__ == "__main__":

    # Sub-test 1: 3-stage pipeline
    def compute_expected_pipeline(inputs):
        a = inputs[0]
        out = np.zeros(N, dtype=np.float32)
        for blk in range(num_blocks):
            P = a[blk*BLOCK:(blk+1)*BLOCK].copy() * 1.1
            B = np.zeros(BLOCK, dtype=np.float32)
            for i in range(BLOCK):
                lo = max(0, i - 2)
                hi = min(BLOCK - 1, i + 2)
                B[i] = P[lo:hi+1].mean()
            F = np.zeros(BLOCK, dtype=np.float32)
            F[0] = (B[0] + B[1]) * 0.6
            F[BLOCK-1] = (B[BLOCK-1] + B[BLOCK-2]) * 0.6
            for i in range(1, BLOCK-1):
                F[i] = ((B[i] + B[i-1]) * 0.6 + B[i+1]) * 0.6
            out[blk*BLOCK:(blk+1)*BLOCK] = F
        return out.astype(np.float32)

    print("Sub-test 1: 3-stage pipeline", file=sys.stderr)
    run_benchmark(
        pipeline_3stage, compute_expected_pipeline,
        targets=["cuda", "rocm", "llvmspirv", "vulkan", "webgpu", "webgl", "metal"],
        label=f"pipeline N={N}",
        init_mod=10,
        atol=1e-2,
    )

    # Sub-test 2: Double-buffered stencil (3 iterations)
    def compute_expected_stencil(inputs):
        a = inputs[0]
        out = np.zeros(N, dtype=np.float32)
        for blk in range(num_blocks):
            buf = a[blk*BLOCK:(blk+1)*BLOCK].copy()
            for _ in range(3):
                new = np.zeros(BLOCK, dtype=np.float32)
                new[0] = (buf[0] + buf[1]) * 0.5
                new[-1] = (buf[-2] + buf[-1]) * 0.5
                for j in range(1, BLOCK-1):
                    new[j] = (buf[j-1] + buf[j] + buf[j+1]) / 3.0
                buf = new
            out[blk*BLOCK:(blk+1)*BLOCK] = buf
        return out.astype(np.float32)

    print("Sub-test 2: Double-buffered stencil", file=sys.stderr)
    run_benchmark(
        stencil_double_buf, compute_expected_stencil,
        targets=["cuda", "rocm", "llvmspirv", "vulkan", "webgpu", "webgl", "metal"],
        label=f"stencil N={N}",
        init_mod=10,
        atol=1e-2,
    )
