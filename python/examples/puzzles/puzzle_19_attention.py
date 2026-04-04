# RUN: env PYTHONPATH=%{pythonpath} MLIR_BUILD_DIR=%{mlir_build_dir} %{python} %s 4 8
# REQUIRES: gpu
"""Puzzle 19 — Attention: compute scaled dot-product attention.

  attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Three kernels matching the Mojo GPU Puzzle:
  1. attention_scores: compute Q @ K^T / sqrt(d)  → scores [SEQ, SEQ]
  2. softmax: row-wise softmax on scores             → weights [SEQ, SEQ]
  3. weighted_sum: weights @ V                       → output [SEQ, DIM]

Simplified single-head attention for small SEQ/DIM:
  - Q, K, V: [SEQ, DIM]
  - Out: [SEQ, DIM]
"""
import sys
import numpy as np
from lego.core import OrderBy, Row
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared

if len(sys.argv) != 3:
    print("Usage: python puzzle_19_attention.py SEQ DIM")
    sys.exit(1)

SEQ = int(sys.argv[1])
DIM = int(sys.argv[2])
assert DIM > 0 and (DIM & (DIM - 1)) == 0, "DIM must be power of 2"

# ===================================================================
# Sub-test 1: attention_scores — Q @ K^T / sqrt(d)
# Each block computes one row of scores. block_id.x = query index.
# ===================================================================
QK_layout = Row(SEQ, DIM)
scores_layout = Row(SEQ, SEQ)


@gpu_kernel(grid=(SEQ,), block=(SEQ,))
def attention_scores(Q: Buffer(QK_layout, SEQ, DIM),
                     K: Buffer(QK_layout, SEQ, DIM),
                     Scores: Buffer(scores_layout, SEQ, SEQ)):
    qi = block_id.x
    ki = thread_id.x
    acc = 0.0
    for d in range(DIM):
        acc += Q[qi, d] * K[ki, d]
    inv_sqrt_dim = 1.0 / (DIM ** 0.5)
    Scores[qi, ki] = acc * inv_sqrt_dim


# ===================================================================
# Sub-test 2: softmax — row-wise softmax using shared memory reduction.
# Each block handles one row of scores. Thread tx handles element tx.
# Steps: find max → subtract max → exp → sum → normalize
# ===================================================================
smem_scores = Row(SEQ)


@gpu_kernel(grid=(SEQ,), block=(SEQ,))
def softmax(Scores: Buffer(scores_layout, SEQ, SEQ),
            Weights: Buffer(scores_layout, SEQ, SEQ),
            smem: Shared(smem_scores, SEQ)):
    row = block_id.x
    tx = thread_id.x
    val = Scores[row, tx]
    # Load into shared memory for reductions
    smem[tx] = val
    barrier()
    # Find max (thread 0 scans — small SEQ)
    max_val = smem[0]
    for i in range(1, SEQ):
        v = smem[i]
        if v > max_val:
            max_val = v
    # Compute exp(val - max)
    exp_val = exp(val - max_val)
    smem[tx] = exp_val
    barrier()
    # Sum (thread 0 scans)
    sum_val = 0.0
    for i in range(SEQ):
        sum_val = sum_val + smem[i]
    # Normalize
    Weights[row, tx] = exp_val / sum_val


# ===================================================================
# Sub-test 3: weighted_sum — weights @ V → output [SEQ, DIM]
# Each block computes one row of the output.
# ===================================================================
V_layout = Row(SEQ, DIM)
out_layout = Row(SEQ, DIM)


@gpu_kernel(grid=(SEQ,), block=(DIM,))
def weighted_sum(Weights: Buffer(scores_layout, SEQ, SEQ),
                 V: Buffer(V_layout, SEQ, DIM),
                 Out: Buffer(out_layout, SEQ, DIM)):
    qi = block_id.x  # which query row
    di = thread_id.x  # which dim element
    acc = 0.0
    for s in range(SEQ):
        acc += Weights[qi, s] * V[s, di]
    Out[qi, di] = acc


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected_scores(inputs):
        q = inputs[0].reshape(SEQ, DIM)
        k = inputs[1].reshape(SEQ, DIM)
        scores = (q @ k.T) / np.sqrt(DIM)
        return scores.ravel().astype(np.float32)

    print("Sub-test 1: attention_scores (Q @ K^T / sqrt(d))", file=sys.stderr)
    run_benchmark(
        attention_scores, compute_expected_scores,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"scores SEQ={SEQ},DIM={DIM}",
        init_mod=10, atol=1e-2,
    )

    def compute_expected_softmax(inputs):
        scores = inputs[0].reshape(SEQ, SEQ)
        # Row-wise softmax
        max_vals = scores.max(axis=1, keepdims=True)
        exp_vals = np.exp(scores - max_vals)
        weights = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        return weights.ravel().astype(np.float32)

    print("Sub-test 2: softmax (row-wise)", file=sys.stderr)
    run_benchmark(
        softmax, compute_expected_softmax,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"softmax SEQ={SEQ}",
        init_mod=10, atol=1e-4,
    )

    def compute_expected_weighted(inputs):
        weights = inputs[0].reshape(SEQ, SEQ)
        v = inputs[1].reshape(SEQ, DIM)
        out = weights @ v
        return out.ravel().astype(np.float32)

    print("Sub-test 3: weighted_sum (weights @ V)", file=sys.stderr)
    run_benchmark(
        weighted_sum, compute_expected_weighted,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"weighted SEQ={SEQ},DIM={DIM}",
        init_mod=10, atol=1e-2,
    )
