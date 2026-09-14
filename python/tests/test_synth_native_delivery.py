"""Deliver a synthesized layout through the real LEGO pipeline: native @gpu_kernel with Shared(layout).

Needs the native MLIR extension (skipped until the build exists). Measures what the
pipeline makes of the exported XOR swizzle (index-arithmetic cost) and, when a
GPU is available, checks the transpose result.
"""
import re
import pytest

pytest.importorskip("lego.mlir.ir")

import numpy as np  # noqa: E402
from lego.core import OrderBy, Row  # noqa: E402
from lego.backend.gpu_dsl import gpu_kernel, Buffer, Shared  # noqa: E402
from lego_synth.layouts import RowMajor, XorSwizzle  # noqa: E402
from lego_synth.export import to_lego_tile  # noqa: E402

N, T = 256, 32
GLOB = OrderBy(Row(N, N)).TileBy([N // T, N // T], [T, T])
SMEM = None   # set by make_transpose; annotations are evaluated at module scope by the DSL


def make_transpose(smem_layout):
    global SMEM
    SMEM = smem_layout

    @gpu_kernel(grid=(N // T, N // T), block=(T, 8))
    def transpose(A: Buffer(GLOB, N, N), Out: Buffer(GLOB, N, N), tile: Shared(SMEM, T, T)):
        bx = block_id.x
        by = block_id.y
        tx = thread_id.x
        ty = thread_id.y
        for k in range(4):
            r = ty + 8 * k
            tile[r, tx] = A[by, bx, r, tx]
        barrier()
        for k in range(4):
            r = ty + 8 * k
            Out[bx, by, r, tx] = tile[tx, r]

    return transpose


def _lowered_arith_ops(builder) -> dict:
    from lego.mlir.ir import Context, Module
    from lego.mlir.passmanager import PassManager
    from lego.backend.dialects.lego_dialect import register as register_lego
    ctx, mod = builder.build_module()
    with ctx:
        src = str(mod)
    ctx2 = Context(); register_lego(ctx2); ctx2.load_all_available_dialects()
    with ctx2:
        m = Module.parse(src)
        PassManager.parse("builtin.module(lego-lower)").run(m.operation)
        text = str(m)
    return {op: len(re.findall(rf"arith\.{op}\b", text)) for op in
            ("muli", "addi", "subi", "divui", "remui", "shli", "shrui", "andi", "xori", "ori")}


def test_native_kernel_with_synthesized_smem_layout_compiles_for_cuda():
    for L in (RowMajor(T, T), XorSwizzle(T, T, shift=0, mask=31)):
        b = make_transpose(to_lego_tile(L))
        result = b.compile(target="cuda", chip="sm_86", features="+ptx78", format="assembly")
        assert result.kernel_path


def test_pipeline_index_arithmetic_cost_of_exported_swizzle_is_reported():
    naive = _lowered_arith_ops(make_transpose(to_lego_tile(RowMajor(T, T))))
    swz = _lowered_arith_ops(make_transpose(to_lego_tile(XorSwizzle(T, T, shift=0, mask=31))))
    print("naive:", naive)
    print("xor  :", swz)
    assert swz["xori"] >= 1                          # the swizzle lowers to a native XOR
    assert sum(swz.values()) <= sum(naive.values()) + 6   # not the 26-op bit decomposition


def test_native_kernel_with_synthesized_smem_layout_runs_correctly_on_the_gpu():
    import shutil, os
    if not os.path.isfile(os.path.join(os.environ.get("MLIR_BUILD_DIR", ""), "bin", "mlir-runner")):
        pytest.skip("needs MLIR_BUILD_DIR with mlir-runner")
    from bench_utils import run_cuda_verify
    init = (np.arange(N * N, dtype=np.int32) * 37 + 17) % 10000
    expected = init.reshape(N, N).T.ravel()
    for L in (RowMajor(T, T), XorSwizzle(T, T, shift=0, mask=31)):
        b = make_transpose(to_lego_tile(L))
        assert run_cuda_verify(b, expected, label=type(L).__name__) is True
