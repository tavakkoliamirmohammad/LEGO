"""Delta addressing: tile addresses at (lane + constant) coordinates as (base ^ K) + C, verified against the layout."""
import pytest
from lego_synth.layouts import XorSwizzle, RowMajor
from lego_synth.kernels.addr import delta_constants


def ldmatrix_lanes():
    rows = [(t % 32) % 8 + ((t % 32) // 16) * 8 for t in range(32)]
    cols = [(((t % 32) // 8) % 2) * 8 for t in range(32)]
    return rows, cols


@pytest.mark.parametrize("layout", [XorSwizzle(64, 80, shift=2, mask=1, unit=8), XorSwizzle(64, 64, shift=0, mask=7, unit=8),
                                    XorSwizzle(64, 96, shift=1, mask=3, unit=8), XorSwizzle(128, 32, shift=1, mask=3, unit=8),
                                    RowMajor(64, 80), RowMajor(64, 64, pad=8)])
@pytest.mark.parametrize("dr,dc", [(0, 0), (16, 0), (32, 16), (48, 48), (16, 32)])
def test_delta_constants_reproduce_the_layout_exactly(layout, dr, dc):
    rows, cols = ldmatrix_lanes()
    if dr + 16 > layout.rows or dc + 16 > layout.cols:
        pytest.skip("delta outside the tile")
    kc = delta_constants(layout, rows, cols, dr, dc)
    assert kc is not None, "every synthesized staging layout must admit the rewrite for ldmatrix lane sets"
    K, C = kc
    for r, c in zip(rows, cols):
        assert (int(layout.flat(r, c)) ^ K) + C == int(layout.flat(r + dr, c + dc))


def test_delta_constants_returns_none_when_no_rewrite_exists():
    # A rotation-like target: row deltas that are not a multiple of the swizzle period on a masked
    # mapping where the lanes straddle the period (row 7 + 1 carries into the swizzle bits).
    layout = XorSwizzle(64, 64, shift=0, mask=7, unit=8)
    rows = [t % 8 for t in range(32)]; cols = [(t // 8) * 8 for t in range(32)]
    assert delta_constants(layout, rows, cols, 1, 0) is None


torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or not pytest.importorskip("lego.mlir.ir", reason="needs the native build"):
    pytest.skip("needs a CUDA GPU", allow_module_level=True)


def _attn(delta, **kw):
    from lego_synth.problems import problem_for
    from lego_synth.search import synthesize
    from lego_synth.machine import ampere
    from lego_synth.export import to_lego_tile
    from lego_synth.kernels.attn_mma import build_attention
    from lego_synth.kernels.launch import CastleKernel
    kvl = lambda bn, d: to_lego_tile(synthesize(problem_for("staging", bn, d, 2, ampere()), "full").layout)
    BH, N, D = kw.pop("BH", 2), kw.pop("N", 256), kw.pop("D", 64)
    bn, bm = kw.get("BLOCK_N", 32), kw.get("BLOCK_M", 64)
    return CastleKernel.from_builder(build_attention(BH, N, D, kvl(bn, D), kvl(bn, D), sQ_layout=kvl(bm, D),
                                                    delta_addr=delta, **kw)), (BH, N, D)


@pytest.mark.parametrize("D,causal", [(64, False), (80, True), (128, False)])
def test_attention_with_delta_addressing_matches_pytorch(D, causal):
    import torch.nn.functional as F
    k, (BH, N, D) = _attn(True, D=D, causal=causal, BLOCK_M=64, BLOCK_N=32, stages=2)
    q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16); kk = torch.randn_like(q); v = torch.randn_like(q)
    o = torch.empty(BH, N, D, device="cuda", dtype=torch.float32)
    k.launch(q, kk, v, o); torch.cuda.synchronize()
    ref = F.scaled_dot_product_attention(q.float(), kk.float(), v.float(), is_causal=causal)
    assert torch.allclose(o, ref, atol=2e-2, rtol=2e-2), (o - ref).abs().max()


def test_delta_addressing_shares_one_base_register_per_tile():
    """Without the rewrite every ldmatrix carries its own hoisted 64-bit address; with it the accesses
    of a tile share a handful of base registers and differ by an XOR and an immediate offset."""
    import re
    hoisted, _ = _attn(False, BH=32, N=2048, D=64, BLOCK_M=64, BLOCK_N=64, stages=2)
    delta, _ = _attn(True, BH=32, N=2048, D=64, BLOCK_M=64, BLOCK_N=64, stages=2)
    def bases(k):
        ptx = open(k.ptx_path).read()
        ld = re.findall(r"ldmatrix[^\n]*\[(%rd\d+)(\+\d+)?\]", ptx)
        return len(ld), len({r for r, _ in ld}), sum(1 for _, imm in ld if imm)
    n_h, distinct_h, imm_h = bases(hoisted)
    n_d, distinct_d, imm_d = bases(delta)
    assert n_h == n_d and distinct_h == n_h and imm_h == 0, (n_h, distinct_h, imm_h)
    assert distinct_d <= 16 and imm_d >= n_d // 2, (n_d, distinct_d, imm_d)
