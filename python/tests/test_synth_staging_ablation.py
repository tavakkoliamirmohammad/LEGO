"""Staging ablation: b16 tiles written by cp.async (dense, 16 B per lane) and read by ldmatrix.x4."""
from lego_synth.ablation import ablation_rows
from lego_synth.layouts import Rotation
from lego_synth.eval_np import conflicts
from lego_synth.machine import ampere
from lego_synth.patterns import ldmatrix_x4, row_vector


def test_chunk_rotation_keeps_16_byte_chunks_contiguous():
    L = Rotation(16, 32, r=1, unit=8)          # rotate 16 B chunks by row
    f = L.flat(__import__("numpy").full(8, 3), __import__("numpy").arange(8))
    assert f.tolist() == list(range(f[0], f[0] + 8))
    assert L.flat(__import__("numpy").array([1]), __import__("numpy").array([0]))[0] == 1 * 32 + 8


def test_staging_ablation_on_80_wide_b16_tile():
    rows = ablation_rows(widths=[80], rows=64, elem_bytes=2, kind="staging")
    by = {r["tier"]: r for r in rows}
    assert by["f2"]["footprint"] == 64 * 128          # block shape padded to the next power of two
    # Padding alone cannot serve both cp.async stores and ldmatrix loads at width 80
    # (an odd chunk stride fixes ldmatrix but wraps the dense store into conflicts).
    assert by["affine_xor"]["footprint"] is None
    assert by["full"]["footprint"] == 64 * 80          # exact: in-bounds chunk swizzle
    for r in rows:
        assert r["conflicts"] in (0, None)


def test_dense_transpose_problem_uses_dense_patterns():
    from lego_synth.problems import problem_for
    from lego_synth.machine import ampere
    masked = problem_for("transpose", 32, 24, 4, ampere())
    dense = problem_for("transpose_dense", 32, 24, 4, ampere())
    assert masked.patterns[0].n_instr == 32            # one instruction per 24-wide row
    assert dense.patterns[0].n_instr == 32 * 24 // 32  # 32 lanes cover 32 consecutive elements
    assert dense.kind == "transpose_dense"
