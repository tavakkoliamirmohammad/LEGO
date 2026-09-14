"""Kernels whose shared tiles exceed the 48 KB static limit must use dynamic shared memory."""
import re
import pytest

pytest.importorskip("lego.mlir.ir")
from lego_synth.kernels.target import detect_target  # noqa: E402
_T = detect_target()
from lego_synth.export import to_lego_tile  # noqa: E402
from lego_synth.layouts import XorSwizzle  # noqa: E402
from lego_synth.kernels.gemm_mma import build_gemm  # noqa: E402
from lego_synth.kernels.launch import extract_ptx  # noqa: E402


def test_gemm_with_64kb_of_shared_tiles_compiles_with_dynamic_shared_memory():
    # 128x128x32 tiles, 4 stages: (128*32 + 32*128) * 2 B * 4 = 64 KB.
    b = build_gemm(256, 256, 256, 128, 128, 32, to_lego_tile(XorSwizzle(128, 32, 0, 3, unit=8)),
                   to_lego_tile(XorSwizzle(32, 128, 0, 7, unit=8)), stages=4, dynamic_shared=True)
    assert b.dynamic_shared_bytes == 64 * 1024
    res = b.compile(target="cuda", chip=_T.chip, features=_T.features, format="assembly")
    ptx = extract_ptx(open(res.kernel_path).read())
    # NVPTX declares dynamic shared memory as an unsized symbol: `.shared .align 16 .b8 __dynamic_shmem__0;`
    assert re.search(r"\.shared \.align \d+ \.b8 __dynamic_shmem__\d*;", ptx), "expected a dynamic shared symbol"
    assert not re.search(r"\.shared \.align \d+ \.b8 \w+\[\d+\];", ptx), "no static shared arrays expected"
