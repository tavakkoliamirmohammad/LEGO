# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 22 — Kernel Fusion: combine forward and backward pass in one kernel.

Demonstrates fusing two operations into a single kernel to avoid
intermediate global memory traffic:

  Forward:  y = x * w     (element-wise multiply)
  Backward: dx = dy * w   (gradient w.r.t. input)
             dw = dy * x   (gradient w.r.t. weight)

Fused kernel reads x, w, dy once and writes y, dx, dw — 3 reads, 3 writes
instead of 6 reads, 6 writes for two separate kernels.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 2:
    print("Usage: python puzzle_22_fusion.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0
num_blocks = N // WG

layout = OrderBy(Row(N)).TileBy([num_blocks], [WG])


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def fused_fwd_bwd(X: Buffer(layout, N), W: Buffer(layout, N),
                  DY: Buffer(layout, N),
                  Y: Buffer(layout, N), DX: Buffer(layout, N),
                  DW: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    # Read inputs once
    x_val = X[bx, tx]
    w_val = W[bx, tx]
    dy_val = DY[bx, tx]
    # Forward: y = x * w
    Y[bx, tx] = x_val * w_val
    # Backward: dx = dy * w, dw = dy * x
    DX[bx, tx] = dy_val * w_val
    DW[bx, tx] = dy_val * x_val


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        x, w, dy = inputs[0], inputs[1], inputs[2]
        y = x * w
        dx = dy * w
        dw = dy * x
        # Output is the last buffer (DW). But run_benchmark expects
        # last buffer = output. We have 6 buffers: X, W, DY, Y, DX, DW
        # The last (DW) is the output buffer.
        return dw.astype(np.float32)

    run_benchmark(
        fused_fwd_bwd, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"N={N}",
        init_mod=10,
        atol=1e-4,
    )
