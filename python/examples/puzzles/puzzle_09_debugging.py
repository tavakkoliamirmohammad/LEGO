"""Puzzle 9 — GPU Debugging: demonstrate correct vs incorrect barrier placement.

The Mojo puzzle teaches debugging GPU race conditions. Here we demonstrate
the PATTERN: what happens when barriers are missing.

Correct version: barrier between shared memory load and compute phases.
Bug version: missing barrier → race condition (reads stale shared memory).

This puzzle runs the CORRECT version and verifies it.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_09_debugging.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0
num_blocks = N // WG

layout = OrderBy(Row(N)).TileBy([num_blocks], [WG])
smem_layout = Row(WG)


# Correct version: barrier between load and compute
@gpu_kernel(grid=(num_blocks,), block=(WG,))
def correct_barrier(A: Buffer(layout, N), Out: Buffer(layout, N),
                    smem: Shared(smem_layout, WG)):
    bx = block_id.x
    tx = thread_id.x
    # Phase 1: Load to shared
    smem[tx] = A[bx, tx]
    # BARRIER: ensures all loads complete before any thread reads
    barrier()
    # Phase 2: Read neighbor from shared (shifted by 1, wrap around)
    # Thread tx reads from thread (tx + 1) % WG
    if tx < WG - 1:
        Out[bx, tx] = smem[tx + 1]
    if tx == WG - 1:
        Out[bx, tx] = smem[0]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        a = inputs[0].reshape(-1, WG)
        out = np.zeros_like(a)
        out[:, :-1] = a[:, 1:]
        out[:, -1] = a[:, 0]
        return out.ravel().astype(np.float32)

    print("Running CORRECT version (with barrier):", file=sys.stderr)
    run_benchmark(
        correct_barrier, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
    )
