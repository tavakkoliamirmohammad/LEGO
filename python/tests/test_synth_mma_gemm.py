"""End-to-end tensor-core GEMM through CASTLE with synthesized shared-memory layouts."""
import os
import re
import pytest

pytest.importorskip("lego.mlir.ir")
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
    res = b.compile(target="cuda", chip="sm_86", features="+ptx78", format="assembly")
    ptx = open(res.kernel_path).read()
    for ins in ("cp.async", "ldmatrix", "mma.sync"):
        assert ins in ptx, ins


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.environ.get("MLIR_BUILD_DIR", ""), "bin", "mlir-runner")),
                    reason="needs mlir-runner")
def test_gemm_result_is_exact_for_small_integer_inputs():
    for sA, sB in ((RowMajor(64, 32), RowMajor(32, 64)), (to_lego_tile(SW), to_lego_tile(XorSwizzle(32, 64, 0, 7, unit=8)))):
        b = build_gemm(M, N, K, BM, BN, BK, sA if not isinstance(sA, RowMajor) else to_lego_tile(sA),
                       sB if not isinstance(sB, RowMajor) else to_lego_tile(sB), stages=2)
        out = run_kernel(b)
        A = init_values(M * K).reshape(M, K); B = init_values(K * N).reshape(K, N)
        expected = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32).ravel()
        assert np.array_equal(out, expected), np.abs(out - expected).max()
