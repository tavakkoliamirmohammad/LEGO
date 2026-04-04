# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 4 8
# REQUIRES: gpu
"""Puzzle 21 — Embedding: demonstrate coalesced vs non-coalesced memory access.

An embedding lookup: given indices, gather rows from a weight matrix.
  output[i] = weights[indices[i]]

Two patterns:
  - Coalesced: threads read consecutive elements of the same row
  - Non-coalesced: threads read from different rows (strided access)

This implements the coalesced version where each block handles one
embedding lookup, and threads cooperate to read the embedding dimension.
"""
import sys
import numpy as np
from lego.core import Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer

if len(sys.argv) != 3:
    print("Usage: python puzzle_21_embedding.py BATCH DIM")
    sys.exit(1)

BATCH = int(sys.argv[1])
DIM = int(sys.argv[2])

# Weights: [BATCH, DIM] — each row is an embedding vector
# For simplicity, indices are identity (index i → row i)
# The kernel gathers row block_id.x and writes it to output row block_id.x
weights_layout = Row(BATCH, DIM)
out_layout = Row(BATCH, DIM)


@gpu_kernel(grid=(BATCH,), block=(DIM,))
def embedding_gather(Weights: Buffer(weights_layout, BATCH, DIM),
                     Out: Buffer(out_layout, BATCH, DIM)):
    # Coalesced access: threads within a block read consecutive elements
    # of the same row — DIM threads read DIM elements in parallel
    row = block_id.x
    col = thread_id.x
    Out[row, col] = Weights[row, col]


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        # Identity embedding lookup: output = input
        return inputs[0].copy()

    run_benchmark(
        embedding_gather, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"{BATCH}x{DIM}",
    )
