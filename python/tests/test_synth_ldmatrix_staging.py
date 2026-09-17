"""GEMM staging: 16-bit tile written by cp.async (16 B per lane) and read by ldmatrix.x4."""
import numpy as np

from lego_synth.machine import ampere
from lego_synth.patterns import row_vector, ldmatrix_x4
from lego_synth.layouts import RowMajor, XorSwizzle
from lego_synth.templates import XorTemplate
from lego_synth.solve import solve_conflict_free
from lego_synth.eval_np import conflicts


def test_dense_row_vector_with_vector_width_covers_four_128B_rows_per_warp_instruction():
    p = row_vector(rows=64, cols=64, warp=32, vec=8, dense=True)
    assert p.n_instr == 64 * 64 // (32 * 8)     # 16 instructions, 4 rows each
    i, j = p.coords(1)
    assert i[0] == 4 and j[3] == 24 and i[8] == 5
    assert p.access_elems == 8


def test_ldmatrix_over_a_64x64_tile_issues_one_x4_per_16x16_block():
    p = ldmatrix_x4(rows=64, cols=64)
    assert p.n_instr == 16
    i, j = p.coords(5)                # block (row 1, col 1) in row-major block order
    assert i.min() == 16 and j.min() == 16


def test_row_major_64x64_b16_tile_conflicts_8_ways_on_ldmatrix_and_128B_swizzle_fixes_it():
    p = ldmatrix_x4(rows=64, cols=64)
    assert conflicts(RowMajor(64, 64), p, elem_bytes=2, machine=ampere()) == 16 * 4 * 7
    cutlass_sw128 = XorSwizzle(64, 64, shift=0, mask=7, unit=8)   # chunk ^= row % 8
    assert conflicts(cutlass_sw128, p, elem_bytes=2, machine=ampere()) == 0
    assert conflicts(cutlass_sw128, row_vector(64, 64, 32, vec=8, dense=True), elem_bytes=2, machine=ampere()) == 0


def test_solver_rediscovers_the_128B_swizzle_for_cp_async_plus_ldmatrix():
    pats = [row_vector(64, 64, 32, vec=8, dense=True), ldmatrix_x4(64, 64)]
    sol = solve_conflict_free(XorTemplate(64, 64, unit=8), pats, elem_bytes=2, machine=ampere())
    assert isinstance(sol, XorSwizzle) and sol.unit == 8
    assert all(conflicts(sol, p, 2, ampere()) == 0 for p in pats)
    # The chunk permutation must be a row-dependent XOR over the 8 chunks of a 128-byte row.
    assert sol.mask & 7 == 7 or sol.mask == 7
