"""Triton fused-attention baseline (tutorial kernel, padded to a power-of-two head dim) vs PyTorch."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402
from lego_synth.bench.triton_attention import triton_attention, head_dim_pad  # noqa: E402


def test_head_dim_pad_is_next_power_of_two():
    assert [head_dim_pad(d) for d in (64, 80, 96, 112, 128)] == [64, 128, 128, 128, 128]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [64, 80, 128])
def test_triton_attention_matches_pytorch(D, causal):
    BH, N = 2, 256
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    o = triton_attention(q, k, v, causal=causal, config=dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2))
    assert o.dtype == torch.float16 and o.shape == q.shape
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
    assert torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2), (o.float() - ref).abs().max()


def test_triton_attention_tuned_path_runs():
    from lego_synth.bench.triton_attention import triton_attention_tuned
    BH, N, D = 2, 256, 64
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    o = triton_attention_tuned(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    assert torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2)
