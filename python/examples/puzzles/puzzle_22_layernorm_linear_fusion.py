# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 256
# REQUIRES: gpu
"""Puzzle 22 — Kernel Fusion: fuse LayerNorm + Linear into a single kernel.

Three implementations matching the Mojo GPU Puzzle:
  1. layernorm: standalone LayerNorm (mean/var → normalize → scale+shift)
  2. linear: standalone y = x * w + b (element-wise, like a diagonal linear)
  3. fused_layernorm_linear: both in one kernel, avoiding intermediate writes

Demonstrates reducing global memory traffic: unfused = 4 reads + 2 writes
per element; fused = 2 reads + 1 write.

Mojo equivalent (fused):
    mean, var = compute_stats(x)
    x_norm = (x - mean) / sqrt(var + eps)
    y = x_norm * w + b
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 2:
    print("Usage: python puzzle_22_fusion.py N")
    sys.exit(1)

N = int(sys.argv[1])
WG = min(256, N)
assert N % WG == 0
num_blocks = N // WG

layout = OrderBy(Row(N)).TileBy([num_blocks], [WG])
smem_layout = Row(WG)

EPS = 1e-5

# ===================================================================
# Sub-test 1: layernorm — per-block LayerNorm using shared memory.
# Each block normalizes WG elements: mean → variance → normalize → scale.
# ===================================================================


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def layernorm(X: Buffer(layout, N), Gamma: Buffer(layout, N),
              Beta: Buffer(layout, N), Out: Buffer(layout, N),
              smem: Shared(smem_layout, WG)):
    bx = block_id.x
    tx = thread_id.x
    x_val = X[bx, tx]
    smem[tx] = x_val
    barrier()
    # Compute mean (all threads scan — small WG)
    mean_val = 0.0
    for i in range(WG):
        mean_val = mean_val + smem[i]
    mean_val = mean_val * (1.0 / WG)
    # Compute variance
    diff = x_val - mean_val
    smem[tx] = diff * diff
    barrier()
    var_val = 0.0
    for i in range(WG):
        var_val = var_val + smem[i]
    var_val = var_val * (1.0 / WG)
    # Normalize and apply affine: out = (x - mean) / sqrt(var + eps) * gamma + beta
    std = sqrt(var_val + EPS)
    x_norm = diff * (1.0 / std)
    Out[bx, tx] = x_norm * Gamma[bx, tx] + Beta[bx, tx]


# ===================================================================
# Sub-test 2: linear — element-wise y = x * w + b
# Separate kernel so we can show fusion benefit.
# ===================================================================


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def linear(X: Buffer(layout, N), W: Buffer(layout, N),
           B: Buffer(layout, N), Out: Buffer(layout, N)):
    bx = block_id.x
    tx = thread_id.x
    Out[bx, tx] = X[bx, tx] * W[bx, tx] + B[bx, tx]


# ===================================================================
# Sub-test 3: fused_layernorm_linear — LayerNorm + Linear in one kernel.
# Avoids writing normalized x to global memory between the two ops.
# ===================================================================


@gpu_kernel(grid=(num_blocks,), block=(WG,))
def fused_layernorm_linear(X: Buffer(layout, N),
                           Gamma: Buffer(layout, N),
                           Beta: Buffer(layout, N),
                           W: Buffer(layout, N),
                           B: Buffer(layout, N),
                           Out: Buffer(layout, N),
                           smem: Shared(smem_layout, WG)):
    bx = block_id.x
    tx = thread_id.x
    x_val = X[bx, tx]
    smem[tx] = x_val
    barrier()
    # Mean
    mean_val = 0.0
    for i in range(WG):
        mean_val = mean_val + smem[i]
    mean_val = mean_val * (1.0 / WG)
    # Variance
    diff = x_val - mean_val
    smem[tx] = diff * diff
    barrier()
    var_val = 0.0
    for i in range(WG):
        var_val = var_val + smem[i]
    var_val = var_val * (1.0 / WG)
    # Fused: normalize + scale + shift + linear — all in registers
    std = sqrt(var_val + EPS)
    x_norm = diff * (1.0 / std)
    ln_out = x_norm * Gamma[bx, tx] + Beta[bx, tx]
    Out[bx, tx] = ln_out * W[bx, tx] + B[bx, tx]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected_ln(inputs):
        x = inputs[0].reshape(num_blocks, WG)
        gamma = inputs[1].reshape(num_blocks, WG)
        beta = inputs[2].reshape(num_blocks, WG)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + EPS)
        out = x_norm * gamma + beta
        return out.ravel().astype(np.float32)

    print("Sub-test 1: layernorm (standalone)", file=sys.stderr)
    run_benchmark(
        layernorm, compute_expected_ln,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"layernorm N={N}",
        init_mod=10, atol=1e-3,
    )

    def compute_expected_linear(inputs):
        x, w, b = inputs[0], inputs[1], inputs[2]
        return (x * w + b).astype(np.float32)

    print("Sub-test 2: linear (standalone)", file=sys.stderr)
    run_benchmark(
        linear, compute_expected_linear,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"linear N={N}",
        init_mod=10, atol=1e-4,
    )

    def compute_expected_fused(inputs):
        x = inputs[0].reshape(num_blocks, WG)
        gamma = inputs[1].reshape(num_blocks, WG)
        beta = inputs[2].reshape(num_blocks, WG)
        w = inputs[3].reshape(num_blocks, WG)
        b = inputs[4].reshape(num_blocks, WG)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + EPS)
        ln_out = x_norm * gamma + beta
        out = ln_out * w + b
        return out.ravel().astype(np.float32)

    print("Sub-test 3: fused layernorm+linear", file=sys.stderr)
    run_benchmark(
        fused_layernorm_linear, compute_expected_fused,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"fused N={N}",
        init_mod=10, atol=1e-3,
    )
