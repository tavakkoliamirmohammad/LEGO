"""Tests for the exhaustive NumPy evaluator of layout hardware properties."""
import numpy as np
import pytest

from lego_synth.machine import ampere
from lego_synth.eval_np import bank_conflicts


def test_ampere_preset_has_nvidia_shared_memory_geometry():
    m = ampere()
    assert m.banks == 32
    assert m.bank_bytes == 4
    assert m.warp == 32


def test_column_read_of_row_major_f32_tile_conflicts_31_ways():
    # lane t reads smem[t][0] of a 32x32 f32 tile: every lane hits bank 0.
    lanes = np.arange(32)
    byte_addrs = (lanes * 32 + 0) * 4
    assert bank_conflicts(byte_addrs, elem_bytes=4, machine=ampere()) == 31


def test_padded_row_stride_33_makes_column_read_conflict_free():
    lanes = np.arange(32)
    assert bank_conflicts((lanes * 33) * 4, elem_bytes=4, machine=ampere()) == 0


def test_xor_swizzled_column_read_is_conflict_free():
    lanes = np.arange(32)
    assert bank_conflicts((lanes * 32 + (0 ^ lanes)) * 4, elem_bytes=4, machine=ampere()) == 0


def test_broadcast_of_one_word_is_conflict_free():
    assert bank_conflicts(np.zeros(32, dtype=np.int64), elem_bytes=4, machine=ampere()) == 0


def test_f64_column_read_is_processed_in_half_warps():
    # stride 33 doubles: lane t occupies banks (2*33t, 2*33t+1) mod 32 = (2t, 2t+1)
    # within each 16-lane half, so no conflicts.
    lanes = np.arange(32)
    assert bank_conflicts((lanes * 33) * 8, elem_bytes=8, machine=ampere()) == 0


def test_float4_loads_at_128_byte_stride_conflict_within_quarter_warps():
    # 16-byte loads: 8 lanes per wavefront; stride 128B puts all 8 lanes on banks 0-3.
    lanes = np.arange(32)
    assert bank_conflicts(lanes * 128, elem_bytes=16, machine=ampere()) == 4 * 7


def test_sub_word_f16_row_read_is_conflict_free():
    lanes = np.arange(32)
    assert bank_conflicts(lanes * 2, elem_bytes=2, machine=ampere()) == 0
