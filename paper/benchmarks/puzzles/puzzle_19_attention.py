"""Puzzle 19 — Attention: compute scaled dot-product attention.

attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

Simplified single-head attention for a small sequence length:
  - Q: [SEQ, DIM]  (query)
  - K: [SEQ, DIM]  (key)
  - V: [SEQ, DIM]  (value)
  - Out: [SEQ, DIM]

This implements the Q @ K^T part (attention scores) as a tiled matmul.
Each block computes one row of the score matrix.
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

Q_layout = Row(SEQ, DIM)
K_layout = Row(SEQ, DIM)
# Scores: [SEQ, SEQ] — each row is attention weights for one query
scores_layout = Row(SEQ, SEQ)
smem_layout = Row(DIM)

@gpu_kernel(grid=(SEQ,), block=(SEQ,))
def attention_scores(Q: Buffer(Q_layout, SEQ, DIM),
                     K: Buffer(K_layout, SEQ, DIM),
                     Scores: Buffer(scores_layout, SEQ, SEQ),
                     smem: Shared(smem_layout, DIM)):
    # Each block computes one row of Q @ K^T
    # block_id.x = query index, thread_id.x = key index
    qi = block_id.x
    ki = thread_id.x
    # Dot product Q[qi] . K[ki] via loop over DIM
    acc = 0.0
    for d in range(DIM):
        acc += Q[qi, d] * K[ki, d]
    # Scale by 1/sqrt(DIM) — ** is compile-time evaluated
    inv_sqrt_dim = 1.0 / (DIM ** 0.5)
    Scores[qi, ki] = acc * inv_sqrt_dim


from bench_utils import run_benchmark

if __name__ == "__main__":

    def compute_expected(inputs):
        q = inputs[0].reshape(SEQ, DIM)
        k = inputs[1].reshape(SEQ, DIM)
        scores = (q @ k.T) / np.sqrt(DIM)
        return scores.ravel().astype(np.float32)

    run_benchmark(
        attention_scores, compute_expected,
        targets=["cuda", "llvmspirv", "vulkan", "webgpu", "metal"],
        label=f"SEQ={SEQ},DIM={DIM}",
        init_mod=10,
        atol=1e-2,
    )
