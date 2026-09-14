"""End-to-end attention through CASTLE with synthesized K/V shared layouts, checked against PyTorch."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cupy")
pytest.importorskip("lego.mlir.ir")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402
from lego_synth.problems import problem_for  # noqa: E402
from lego_synth.search import synthesize  # noqa: E402
from lego_synth.machine import ampere  # noqa: E402
from lego_synth.export import to_lego_tile  # noqa: E402
from lego_synth.kernels.attn_mma import build_attention  # noqa: E402
from lego_synth.kernels.launch import CastleKernel  # noqa: E402


def kv_layout(block_n, d):
    return to_lego_tile(synthesize(problem_for("staging", block_n, d, 2, ampere()), "full").layout)


@pytest.mark.parametrize("D", [64, 80, 128])
def test_attention_matches_pytorch(D):
    BH, N = 2, 256
    L = kv_layout(32, D)
    k = CastleKernel.from_builder(build_attention(BH, N, D, L, L, BLOCK_M=64, BLOCK_N=32, stages=2))
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float32)
    k.launch(q, kk, v, o)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float())
    assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (o - ref).abs().max()


@pytest.mark.parametrize("D", [64, 80])
def test_causal_attention_matches_pytorch(D):
    BH, N = 2, 256
    L = kv_layout(32, D)
    k = CastleKernel.from_builder(build_attention(BH, N, D, L, L, BLOCK_M=64, BLOCK_N=32, stages=2, causal=True))
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float32)
    k.launch(q, kk, v, o)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float(), is_causal=True)
    assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (o - ref).abs().max()
