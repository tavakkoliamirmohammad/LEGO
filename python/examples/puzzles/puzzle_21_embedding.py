# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 4 8
# REQUIRES: gpu
"""Puzzle 21 — Embedding: gather rows from a weight matrix by index.

  output[batch, dim] = weights[indices[batch], dim]

Two implementations matching the Mojo GPU Puzzle:
  1. embedding_coalesced: 1D thread layout, threads read consecutive dim elements
     of the same embedding row → coalesced memory access
  2. embedding_2d: 2D thread layout (batch on x, dim on y), each block handles
     one embedding lookup → per-row parallelism

For simplicity, indices are identity: index[i] = i.
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 3:
    print("Usage: python puzzle_21_embedding.py BATCH DIM")
    sys.exit(1)

BATCH = int(sys.argv[1])
DIM = int(sys.argv[2])

weights_layout = Row(BATCH, DIM)
out_layout = Row(BATCH, DIM)

# ===================================================================
# Sub-test 1: embedding_coalesced — 1D flat thread layout
# Total threads = BATCH * DIM arranged as a single flat block.
# Threads within a warp read consecutive memory elements → coalesced.
# Uses a flat layout to express the 1D→2D index mapping via LEGO.
# ===================================================================
TOTAL = BATCH * DIM
flat_layout = Row(TOTAL)


@gpu_kernel(grid=(1,), block=(TOTAL,))
def embedding_coalesced(Weights: Buffer(flat_layout, TOTAL),
                        Out: Buffer(flat_layout, TOTAL)):
    tx = thread_id.x
    Out[tx] = Weights[tx]


# ===================================================================
# Sub-test 2: embedding_2d — per-row blocks
# block_id.x = batch index, thread_id.x = dim index.
# Each block handles one embedding lookup.
# ===================================================================


@gpu_kernel(grid=(BATCH,), block=(DIM,))
def embedding_2d(Weights: Buffer(weights_layout, BATCH, DIM),
                 Out: Buffer(out_layout, BATCH, DIM)):
    row = block_id.x
    col = thread_id.x
    Out[row, col] = Weights[row, col]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        return inputs[0].copy()

    print("Sub-test 1: coalesced embedding (1D thread layout)", file=sys.stderr)
    run_benchmark(
        embedding_coalesced, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"coalesced {BATCH}x{DIM}",
    )

    print("Sub-test 2: 2D embedding (per-row blocks)", file=sys.stderr)
    run_benchmark(
        embedding_2d, compute_expected,
        targets=["cuda", "rocm", "llvmspirv", "intel", "vulkan", "webgpu", "webgl", "metal"],
        label=f"2d {BATCH}x{DIM}",
    )
