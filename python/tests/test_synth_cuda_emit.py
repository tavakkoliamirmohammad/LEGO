"""Tests for emitting a concrete layout's element-index expression as CUDA C."""
import numpy as np

from lego_synth.layouts import RowMajor, XorSwizzle, Rotation
from lego_synth.harness.cuda_emit import index_expr


def _c_eval(expr: str, i: int, j: int) -> int:
    # C integer semantics on non-negative ints coincide with Python's for >>, &, ^, %, *, +.
    return eval(expr, {"__builtins__": {}}, {"i": i, "j": j})


def test_emitted_expression_matches_layout_for_all_families():
    layouts = [RowMajor(32, 32, pad=1), XorSwizzle(32, 32, shift=1, mask=7), Rotation(24, 24, r=5)]
    for L in layouts:
        expr = index_expr(L)
        for i in range(0, L.rows, 5):
            for j in range(0, L.cols, 3):
                assert _c_eval(expr, i, j) == int(L.flat(np.array([i]), np.array([j]))[0]), (L, i, j)


def test_emitted_expression_uses_only_the_names_i_and_j():
    expr = index_expr(XorSwizzle(32, 32, shift=0, mask=31))
    assert "(i)" in expr and "(j)" in expr
