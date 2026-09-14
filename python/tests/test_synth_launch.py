"""Launch CASTLE-compiled PTX on device tensors through CuPy: correctness against cuBLAS and event timing."""
import pytest

torch = pytest.importorskip("torch")
cupy = pytest.importorskip("cupy")
pytest.importorskip("lego.mlir.ir")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from lego_synth.layouts import XorSwizzle  # noqa: E402
from lego_synth.export import to_lego_tile  # noqa: E402
from lego_synth.kernels.gemm_mma import build_gemm  # noqa: E402
from lego_synth.kernels.launch import CastleKernel  # noqa: E402

M = N = K = 256
BM, BN, BK = 64, 64, 32


@pytest.fixture(scope="module")
def kernel():
    b = build_gemm(M, N, K, BM, BN, BK, to_lego_tile(XorSwizzle(64, 32, 0, 3, unit=8)),
                   to_lego_tile(XorSwizzle(32, 64, 0, 7, unit=8)), stages=4)
    return CastleKernel.from_builder(b)


def test_launched_gemm_matches_cublas_on_device_tensors(kernel):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    kernel.launch(A, B, C)
    torch.cuda.synchronize()
    ref = (A.float() @ B.float())
    assert torch.allclose(C, ref, atol=1e-2, rtol=1e-3), (C - ref).abs().max()


def test_timing_returns_positive_milliseconds(kernel):
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    ms = kernel.time(A, B, C, iters=20, warmup=5)
    assert 0 < ms < 10
