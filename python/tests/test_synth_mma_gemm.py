"""End-to-end tensor-core GEMM through CASTLE with synthesized shared-memory layouts."""
import os
import re
import pytest

pytest.importorskip("lego.mlir.ir")
from lego_synth.kernels.target import detect_target  # noqa: E402
_T = detect_target()
import numpy as np  # noqa: E402
from lego_synth.layouts import RowMajor, XorSwizzle  # noqa: E402
from lego_synth.export import to_lego_tile  # noqa: E402
from lego_synth.kernels.gemm_mma import build_gemm  # noqa: E402
from lego_synth.kernels.runner import run_kernel, init_values  # noqa: E402

M = N = K = 128
BM, BN, BK = 64, 64, 32
SW = XorSwizzle(64, 32, shift=0, mask=3, unit=8)     # 64 B rows: 4 chunks, chunk ^= row & 3


def test_gemm_kernel_compiles_to_ptx_with_tensor_core_instructions():
    b = build_gemm(M, N, K, BM, BN, BK, to_lego_tile(SW), to_lego_tile(XorSwizzle(32, 64, 0, 7, unit=8)), stages=2)
    res = b.compile(target="cuda", chip=_T.chip, features=_T.features, format="assembly")
    ptx = open(res.kernel_path).read()
    for ins in ("cp.async", "ldmatrix", "mma.sync"):
        assert ins in ptx, ins


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.environ.get("MLIR_BUILD_DIR", ""), "bin", "mlir-runner")),
                    reason="needs mlir-runner")
def test_gemm_result_is_exact_for_small_integer_inputs():
    for sA, sB in ((RowMajor(64, 32), RowMajor(32, 64)), (to_lego_tile(SW), to_lego_tile(XorSwizzle(32, 64, 0, 7, unit=8)))):
        b = build_gemm(M, N, K, BM, BN, BK, sA if not isinstance(sA, RowMajor) else to_lego_tile(sA),
                       sB if not isinstance(sB, RowMajor) else to_lego_tile(sB), stages=2, packed_out=False)
        out = run_kernel(b)
        A = init_values(M * K).reshape(M, K); B = init_values(K * N).reshape(K, N)
        expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32).ravel()
        assert np.array_equal(out, expected), np.abs(out - expected).max()


def test_gemm_f16_output_matches_reference():
    """out_dtype="f16": accumulation stays f32, the store rounds to f16 (cuBLAS's output type)."""
    torch = pytest.importorskip("torch"); pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from lego_synth.kernels.launch import CastleKernel, extract_ptx
    b = build_gemm(M, N, K, BM, BN, BK, to_lego_tile(SW), to_lego_tile(XorSwizzle(32, 64, 0, 7, unit=8)), stages=2,
                   out_dtype="f16")
    res = b.compile(target="cuda", chip=_T.chip, features=_T.features, format="assembly")
    ptx = extract_ptx(open(res.kernel_path).read())
    assert "cvt.rn.f16" in ptx
    assert "st.global.b16" not in ptx and ("st.global.b32" in ptx or "st.global.u32" in ptx), "f16 pairs must be stored as one 32-bit word"
    k = CastleKernel.from_builder(b)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16); B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    k.launch(A, B, C); torch.cuda.synchronize()
    assert torch.allclose(C.float(), (A.float() @ B.float()), atol=1e-1, rtol=1e-2)


def test_gemm_delta_addressing_matches_hoisted_addressing():
    torch = pytest.importorskip("torch"); pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from lego_synth.kernels.launch import CastleKernel
    A = torch.randn(256, 256, device="cuda", dtype=torch.float16); B = torch.randn_like(A)
    ref = A.float() @ B.float()
    for delta in (False, True):
        for BM_, BN_, BK_, wm, wn in ((64, 64, 32, 2, 2), (128, 128, 32, 2, 2), (128, 128, 32, 4, 2)):
            sA = to_lego_tile(XorSwizzle(BM_, BK_, 1, 3, unit=8)); sB = to_lego_tile(XorSwizzle(BK_, BN_, 0, 7, unit=8))
            k = CastleKernel.from_builder(build_gemm(256, 256, 256, BM_, BN_, BK_, sA, sB, warps_m=wm, warps_n=wn, stages=2,
                                                     delta_addr=delta, out_dtype="f16"))
            C = torch.empty(256, 256, device="cuda", dtype=torch.float16)
            k.launch(A, B, C); torch.cuda.synchronize()
            assert torch.allclose(C.float(), ref, atol=1e-1, rtol=1e-2), (delta, BM_, BN_)


def test_gemm_f32_output_stores_pairs_as_64_bit_words():
    b = build_gemm(M, N, K, BM, BN, BK, to_lego_tile(SW), to_lego_tile(XorSwizzle(32, 64, 0, 7, unit=8)), stages=2)
    res = b.compile(target="cuda", chip=_T.chip, features=_T.features, format="assembly")
    from lego_synth.kernels.launch import extract_ptx
    ptx = extract_ptx(open(res.kernel_path).read())
    # the pair is one 64-bit word in the IR; NVPTX may legalize it to two 32-bit stores, never scalar f32 stores
    assert "st.global.f32" not in ptx and any(t in ptx for t in ("st.global.b64", "st.global.u64", "st.global.v2", "st.global.b32"))
