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


@pytest.mark.parametrize("D", [64, 80])
def test_attention_f16_output(D):
    BH, N = 2, 256
    L = kv_layout(32, D)
    k = CastleKernel.from_builder(build_attention(BH, N, D, L, L, BLOCK_M=64, BLOCK_N=32, stages=2, out_dtype="f16"))
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk = torch.randn_like(q); v = torch.randn_like(q)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float16)
    k.launch(q, kk, v, o)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float())
    assert torch.allclose(o.float(), ref, atol=2e-2, rtol=2e-2), (o.float() - ref).abs().max()


@pytest.mark.parametrize("D,causal,bn", [(64, False, 32), (80, True, 32), (128, False, 64)])
def test_attention_two_mtiles_per_warp(D, causal, bn):
    """rows_per_warp=32: each warp owns two 16-row m-tiles, so every K/V fragment feeds four mma."""
    BH, N = 2, 256
    L = kv_layout(bn, D)
    k = CastleKernel.from_builder(build_attention(BH, N, D, L, L, BLOCK_M=64, BLOCK_N=bn, stages=2, causal=causal,
                                                  rows_per_warp=32))
    assert k.block == (64, 1, 1)
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk = torch.randn_like(q); v = torch.randn_like(q)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float32)
    k.launch(q, kk, v, o)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float(), is_causal=causal)
    assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (o - ref).abs().max()


@pytest.mark.parametrize("D", [64, 128])
def test_attention_q_reloaded_from_shared(D):
    BH, N = 2, 256
    L = kv_layout(32, D)
    k = CastleKernel.from_builder(build_attention(BH, N, D, L, L, BLOCK_M=64, BLOCK_N=32, stages=2, q_in_regs=False,
                                                  rows_per_warp=32))
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk = torch.randn_like(q); v = torch.randn_like(q)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float32)
    k.launch(q, kk, v, o)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float())
    assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (o - ref).abs().max()


def _ptx_of(builder):
    from lego_synth.kernels.target import detect_target
    from lego_synth.kernels.launch import extract_ptx
    t = detect_target()
    res = builder.compile(target="cuda", chip=t.chip, features=t.features, format="assembly")
    return extract_ptx(open(res.kernel_path).read())


def test_attention_exp2_is_a_single_approx_instruction():
    """exp2 must lower to ex2.approx, not the 4-instruction libdevice __nv_exp2f (compare + 2 muls + ex2)."""
    L = kv_layout(32, 64)
    ptx = _ptx_of(build_attention(2, 256, 64, L, L, BLOCK_M=64, BLOCK_N=32, stages=2))
    assert "ex2.approx.ftz.f32" in ptx
    assert ptx.count("setp.") <= 4, ptx.count("setp.")     # only loop-control compares remain in the non-causal kernel

    assert "fma.rn.f32" in ptx                             # scale-and-subtract fused


@pytest.mark.parametrize("D", [64, 80])
def test_attention_with_swizzled_q_tile(D):
    BH, N = 2, 256
    L = kv_layout(32, D)
    LQ = kv_layout(64, D)
    b = build_attention(BH, N, D, L, L, BLOCK_M=64, BLOCK_N=32, stages=2, sQ_layout=LQ, causal=True)
    k = CastleKernel.from_builder(b)
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk = torch.randn_like(q); v = torch.randn_like(q)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float32)
    k.launch(q, kk, v, o)
    torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float(), is_causal=True)
    assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (o - ref).abs().max()


def test_attention_row_sum_is_reduced_across_the_quad_only_once():
    """The running row sum stays a per-lane partial (FlashAttention-2's trick): the quad shuffle-reduce
    of the sum happens once at the end, only the row max is reduced per key block."""
    L = kv_layout(32, 64)
    ptx = _ptx_of(build_attention(2, 256, 64, L, L, BLOCK_M=64, BLOCK_N=32, stages=2))
    # per unrolled key block: 2 rows x 2 shuffles for the max; epilogue: 2 rows x 2 shuffles for the sum
    assert ptx.count("shfl.sync.bfly") <= 2 * 4 + 4, ptx.count("shfl.sync.bfly")
